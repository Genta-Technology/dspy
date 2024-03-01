import requests

from typing import Any, Optional

from dsp.modules.lm import LM

class GentaLLM(LM):
    """Integration to call models hosted in Genta platform.

    Args:
        model (str, optional): Genta mode ID. Defaults to "Mistral-7B-Instruct".
        api_key (Optional[str], optional): Genta API token. Defaults to None.
        **kwargs: Additional arguments to pass to the API provider.
    Example:
        import dspy
        dspy.configure(lm=dspy.Clarifai(model=MODEL_URL,
                                        api_key=TOKEN,
                                        inference_params={"max_tokens":100,'temperature':0.6}))
    """
    __CHAT_URL = "https://api.genta.tech/chat/"

    def __init__(
        self,
        model: str = "Starstreak",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model)
        self.api_key = api_key
        self.history: list[dict[str, Any]] = []
        self.kwargs: dict[str, Any] = {
            "n": 1,
            "inference_params": {
                "token": api_key,
                "model_name": model,
                "parameters": {
                    "best_of": 1,
                    "decoder_input_details": False,
                    "details": False,
                    "do_sample": True,
                    "max_new_tokens": 128,
                    "repetition_penalty": 1.03,
                    "return_full_text": False,
                    "seed": None,
                    "stop": [],
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_n_tokens": None,
                    "top_p": 0.95,
                    "truncate": None,
                    "typical_p": 0.95,
                    "watermark": False,
                }
            },
            **kwargs
        }
        self.provider = "genta"
        self.headers = {"Content-Type": "application/json"}

    def basic_request(
        self,
        prompt: str,
        kwargs**
    ):
        user_params = self.kwargs['inference_params']
        if "parameters" in kwargs:
            user_param['parameters'] = {**user_params['parameters'], **kwargs['parameters']}

        user_params['inference_params']['inputs'] = [{'role': 'user', 'content': prompt}]
        response = requests.post(self.__CHAT_URL, json=user_params, headers=self.headers).json()[0][0][0]['generated_text']
        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": {**self.kwargs, **kwargs}
        }
        self.history.append(history)
        return response

    def request(self, prompt: str, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs
    ):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n = kwargs.pop("n", 1)
        completions = []
        for i in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response)

        return completions
