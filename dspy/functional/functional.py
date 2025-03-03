import inspect, os, openai, dspy, typing, pydantic
from typing import Annotated
import typing
from dsp.templates import passages2text
import json

from dspy.signatures.signature import ensure_signature


MAX_RETRIES = 3


def predictor(func):
    signature = _func_to_signature(func)
    *_, output_key = signature.output_fields.keys()
    return _StripOutput(TypedPredictor(signature), output_key)


def cot(func):
    signature = _func_to_signature(func)
    *_, output_key = signature.output_fields.keys()
    return _StripOutput(TypedChainOfThought(signature), output_key)


class _StripOutput(dspy.Module):
    def __init__(self, predictor, output_key):
        super().__init__()
        self.predictor = predictor
        self.output_key = output_key
    
    def copy(self):
        return _StripOutput(self.predictor.copy(), self.output_key)

    def forward(self, **kwargs):
        prediction = self.predictor(**kwargs)
        return prediction[self.output_key]


class FunctionalModule(dspy.Module):
    """ To use the @cot and @predictor decorators, your module needs to inheret form this class. """
    def __init__(self):
        super().__init__()
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, dspy.Module):
                self.__dict__[name] = attr.copy()


def TypedChainOfThought(signature, make_example=False):
    """ Just like TypedPredictor, but adds a ChainOfThought OutputField. """
    signature = ensure_signature(signature)
    output_keys = ", ".join(signature.output_fields.keys())
    return TypedPredictor(signature.prepend(
        "reasoning",
        dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the " + output_keys + "}. We ...",
        ),
    ), make_example)


class TypedPredictor(dspy.Module):
    def __init__(self, signature, make_example=False):
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
        self.make_example = make_example

    def copy(self):
        return TypedPredictor(self.signature, self.make_example)

    @staticmethod
    def _make_example(type_):
        # Note: DSPy will cache this call so we only pay the first time TypedPredictor is called.
        return dspy.Predict(
            dspy.Signature(
                "json_schema -> json_object",
                "Make a very succinct json object that validates with the following schema",
            )
        )(json_schema=json.dumps(type_.model_json_schema())).json_object
        # TODO: Another fun idea is to only (but automatically) do this if the output fails.
        # We could also have a more general "suggest solution" prompt that tries to fix the output
        # More directly.
        # TODO: Instead of using a language model to create the example, we can also just use a
        # library like https://pypi.org/project/polyfactory/ that's made exactly to do this.

    def _prepare_signature(self):
        """Add formats and parsers to the signature fields, based on the type
        annotations of the fields."""
        signature = self.signature
        for name, field in self.signature.fields.items():
            is_output = field.json_schema_extra["__dspy_field_type"] == "output"
            type_ = field.annotation
            if is_output:
                if type_ in (str, int, float, bool):
                    signature = signature.with_updated_fields(
                        name,
                        desc=field.json_schema_extra.get("desc", "")
                        + (f". Respond with a single {type_.__name__} value"),
                        format=lambda x: x if isinstance(x, str) else str(x),
                        parser=type_,
                    )
                else:
                    # Anything else we wrap in a pydantic object
                    unwrap = lambda x: x
                    if not (inspect.isclass(type_) and issubclass(type_, pydantic.BaseModel)):
                        type_ = pydantic.create_model(
                            "Output", value=(type_, ...), __base__=pydantic.BaseModel
                        )
                        unwrap = lambda x: x.value
                    signature = signature.with_updated_fields(
                        name,
                        desc=field.json_schema_extra.get("desc", "")
                        + (
                            f". Respond with a single JSON object using the schema "
                            + json.dumps(type_.model_json_schema())
                            + (". For example: " + self._make_example(type_) if self.make_example else "")
                        ),
                        format=lambda x: (
                            x if isinstance(x, str) else x.model_dump_json()
                        ),
                        parser=lambda x: unwrap(
                            type_.model_validate_json(_unwrap_json(x))
                        ),
                    )
            else:  # If input field
                format = lambda x: x if isinstance(x, str) else str(x)
                if type_ in (list[str], tuple[str]):
                    format = passages2text
                elif inspect.isclass(type_) and issubclass(type_, pydantic.BaseModel):
                    format = lambda x: x if isinstance(x, str) else x.model_dump_json()
                signature = signature.with_updated_fields(name, format=format)

        return signature

    def forward(self, **kwargs):
        modified_kwargs = kwargs.copy()
        # We have to re-prepare the signature on every forward call, because the base
        # signature might have been modified by an optimizer or something like that.
        signature = self._prepare_signature()
        for try_i in range(MAX_RETRIES):
            result = self.predictor(**modified_kwargs, new_signature=signature)
            errors = {}
            parsed_results = {}
            # Parse the outputs
            for name, field in signature.output_fields.items():
                try:
                    value = getattr(result, name)
                    parser = field.json_schema_extra.get("parser", lambda x: x)
                    parsed_results[name] = parser(value)
                except (pydantic.ValidationError, ValueError) as e:
                    errors[name] = e
            if errors:
                # Add new fields for each error
                for name, error in errors.items():
                    modified_kwargs[f"error_{name}_{try_i}"] = str(error)
                    signature = signature.append(
                        f"error_{name}_{try_i}",
                        dspy.InputField(
                            prefix=f"Past Error "
                            + (f"({name}):" if try_i == 0 else f"({name}, {try_i+1}):"),
                            desc="An error to avoid in the future",
                        ),
                    )
            else:
                # If there are no errors, we return the parsed results
                for name, value in parsed_results.items():
                    setattr(result, name, value)
                return result
        raise ValueError("Too many retries")


def _func_to_signature(func):
    """Make a dspy.Signature based on a function definition."""
    sig = inspect.signature(func)
    annotations = typing.get_type_hints(func)
    output_key = func.__name__
    instructions = func.__doc__
    fields = {}

    # Input fields
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        # We default to str as the type of the input
        annotation = annotations.get(param.name, str)
        kwargs = {}
        if typing.get_origin(annotation) is Annotated:
            annotation, kwargs["desc"] = typing.get_args(annotation)
        fields[param.name] = (annotation, dspy.InputField(**kwargs))

    # Output field
    kwargs = {}
    annotation = annotations.get("return", str)
    if typing.get_origin(annotation) is Annotated:
        annotation, kwargs["desc"] = typing.get_args(annotation)
    fields[output_key] = (annotation, dspy.OutputField(**kwargs))

    return dspy.Signature(fields, instructions)


def _unwrap_json(output):
    output = output.strip()
    if output.startswith("```"):
        if not output.startswith("```json"):
            raise ValueError("json output should start with ```json")
        if not output.endswith("```"):
            raise ValueError("json output should end with ```")
        output = output[7:-3].strip()
    if not output.startswith("{") or not output.endswith("}"):
        raise ValueError("json output should start and end with { and }")
    return output


################################################################################
# Example usage
################################################################################


def main():
    class Answer(pydantic.BaseModel):
        value: float
        certainty: float
        comments: list[str] = pydantic.Field(
            description="At least two comments about the answer"
        )

    class QA(dspy.Module):
        @predictor
        def hard_question(self, topic: str) -> str:
            """Think of a hard factual question about a topic. It should be answerable with a number."""

        @cot
        def answer(self, question: Annotated[str, "Question to answer"]) -> Answer:
            pass

        def forward(self, **kwargs):
            question = self.hard_question(**kwargs)
            return (question, self.answer(question=question))

    openai.api_key = os.getenv("OPENAI_API_KEY")
    lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    # lm = dspy.OpenAI(model="gpt-4", max_tokens=4000)
    # lm = dspy.OpenAI(model="gpt-4-preview-1106", max_tokens=4000)
    with dspy.context(lm=lm):
        qa = QA()
        question, answer = qa(topic="Physics")
        # lm.inspect_history(n=5)

        print("Question:", question)
        print("Answer:", answer)


################################################################################
# HotpotQA example with SimpleBaleen
################################################################################


def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred):
        return False
    if not dspy.evaluate.answer_passage_match(example, pred):
        return False

    hops = [example.question] + [
        outputs.query for *_, outputs in trace if "query" in outputs
    ]

    if max([len(h) for h in hops]) > 100:
        return False
    if any(
        dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8)
        for idx in range(2, len(hops))
    ):
        return False

    return True


def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
    found_titles = set(
        map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context])
    )

    return gold_titles.issubset(found_titles)


def hotpot():
    from dsp.utils import deduplicate
    import dspy.evaluate
    from dspy.datasets import HotPotQA
    from dspy.evaluate.evaluate import Evaluate
    from dspy.teleprompt.bootstrap import BootstrapFewShot

    print("Load the dataset.")
    dataset = HotPotQA(
        train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0
    )
    trainset = [x.with_inputs("question") for x in dataset.train]
    devset = [x.with_inputs("question") for x in dataset.dev]
    print("Done")

    class SimplifiedBaleen(FunctionalModule):
        def __init__(self, passages_per_hop=3, max_hops=1):
            super().__init__()
            self.retrieve = dspy.Retrieve(k=passages_per_hop)
            self.max_hops = max_hops

        @cot
        def generate_query(self, context: list[str], question) -> str:
            """Write a simple search query that will help answer a complex question."""
            pass

        @cot
        def generate_answer(self, context: list[str], question) -> str:
            """Answer questions with short factoid answers."""
            pass

        def forward(self, question):
            context = []

            for hop in range(self.max_hops):
                query = self.generate_query(context=context, question=question)
                passages = self.retrieve(query).passages
                context = deduplicate(context + passages)

            answer = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=answer)

    openai.api_key = os.getenv("OPENAI_API_KEY")
    rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=4000)
    dspy.settings.configure(lm=lm, rm=rm, trace=[])

    evaluate_on_hotpotqa = Evaluate(
        devset=devset, num_threads=10, display_progress=True, display_table=5
    )

    # uncompiled (i.e., zero-shot) program
    uncompiled_baleen = SimplifiedBaleen()
    print(
        "Uncompiled Baleen retrieval score:",
        evaluate_on_hotpotqa(uncompiled_baleen, metric=gold_passages_retrieved),
    )

    # compiled (i.e., few-shot) program
    compiled_baleen = BootstrapFewShot(
        metric=validate_context_and_answer_and_hops
    ).compile(
        SimplifiedBaleen(),
        teacher=SimplifiedBaleen(passages_per_hop=2),
        trainset=trainset,
    )
    print(
        "Compiled Baleen retrieval score:",
        evaluate_on_hotpotqa(compiled_baleen, metric=gold_passages_retrieved),
    )
    # lm.inspect_history(n=5)


if __name__ == "__main__":
    # main()
    hotpot()
