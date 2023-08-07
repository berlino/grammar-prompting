import abc
import time
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import openai
import google.generativeai as palm
import google.api_core.exceptions as palm_exceptions

import neural_lark.utils as utils
from neural_lark.flags import FLAGS
from neural_lark.train_utils import logger
from neural_lark.structs import LLMResponse


class LargeLanguageModel(abc.ABC):
    """A pretrained large language model."""

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this LLM.

        This identifier should include sufficient information so that
        querying the same model with the same prompt and same identifier
        should yield the same result (assuming temperature 0).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _sample_completions(self,
                            prompt: str,
                            temperature: float,
                            stop_token: str,
                            num_completions: int = 1) -> List[LLMResponse]:
        """This is the main method that subclasses must implement.

        This helper method is called by sample_completions(), which
        caches the prompts and responses to disk.
        """
        raise NotImplementedError("Override me!")
    
    def sample_completions(self,
                           prompt: str,
                           temperature: float,
                           stop_token: str,
                           num_completions: int = 1,
                           disable_cache: bool = False) -> List[LLMResponse]:
        """Sample one or more completions from a prompt.

        Higher temperatures will increase the variance in the responses.
        The seed may not be used and the results may therefore not be
        reproducible for LLMs where we only have access through an API
        that does not expose the ability to set a random seed. Responses
        are saved to disk.
        """

        # Set up the cache file.
        os.makedirs(FLAGS.llm_cache_dir, exist_ok=True)
        llm_id = self.get_id()
        prompt_id = utils.str_to_identifier(prompt)
        # If the temperature is 0, the seed does not matter.
        escaped_stop_token = stop_token.replace("\n", "\\n")
        if temperature == 0.0:
            config_id = f"most_likely_{num_completions}_{escaped_stop_token}_{FLAGS.freq_penalty}"
        else:
            config_id = f"{temperature}_{FLAGS.seed}_{num_completions}_{escaped_stop_token}_{FLAGS.freq_penalty}"
        cache_filename = f"{llm_id}_{config_id}_{prompt_id}.pkl"
        cache_filepath = Path(FLAGS.llm_cache_dir) / cache_filename
        if not os.path.exists(cache_filepath):
            os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
        if disable_cache or not os.path.exists(cache_filepath):
            logger.debug(f"Querying LLM {llm_id} with new prompt.")
            completions = self._sample_completions(prompt,
                                                   temperature,
                                                   stop_token, num_completions)
            # Cache the completions.
            with open(cache_filepath, 'wb') as f:
                pickle.dump(completions, f)
            logger.debug(f"Saved LLM response to {cache_filepath}.")
        
        # Load the saved completion.
        with open(cache_filepath, 'rb') as f:
            completions = pickle.load(f)
        logger.debug(f"Loaded LLM response from {cache_filepath}.")
        return completions
    
    def greedy_completion(self,
                          prompt: str,
                          stop_token: str) -> LLMResponse:
        """Sample a greedy completion from a prompt."""
        responses = self.sample_completions(prompt, 0.0, stop_token)
        assert len(responses) == 1
        return responses[0]

    #@abc.abstractmethod
    def _sample_next_token_with_logit_bias(self, prompt, logit_bias, temperature):
        """Sample the next token from the model with a logit bias."""
        raise NotImplementedError("Override me!")

    
class GPT(LargeLanguageModel):
    AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    def __init__(self, model_name: str, use_azure=True) -> None:
        self._model_name = model_name
        self.use_azure = use_azure
        if self.use_azure:
            openai.api_key = self.AZURE_API_KEY
            openai.api_base =  "https://symdistill.openai.azure.com/"
            openai.api_type = 'azure'
            openai.api_version = '2023-03-15-preview'
        else:
            openai.api_key = self.OPENAI_API_KEY

    def get_id(self) -> str:
        return f"gpt_{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            temperature: float,
            stop_token: str,
            num_completions: int = 1) -> List[LLMResponse]:  
        response = None
        for _ in range(6):
            try:
                response = openai.Completion.create(
                    engine=self._model_name,
                    prompt=prompt,
                    temperature=temperature,
                    stop=stop_token,
                    max_tokens=FLAGS.max_tokens,
                    frequency_penalty=FLAGS.freq_penalty,
                    n=num_completions)
                # Successfully queried, so break.
                break
            except (openai.error.RateLimitError,
                    openai.error.APIConnectionError, openai.error.APIError):
                # Wait for 60 seconds if this limit is reached. Hopefully rare.
                time.sleep(6)

        if response is None:
            raise RuntimeError("Failed to query OpenAI API.")
        
        assert len(response["choices"]) == num_completions
        return [
            self._raw_to_llm_response(r, prompt, temperature,stop_token, num_completions)
            for r in response["choices"]
        ]
    
    def _sample_next_token_with_logit_bias(self, prompt, logit_bias, temperature=0.0):
        response = None
        for _ in range(6):
            try:
                response = openai.Completion.create(
                    engine=self._model_name,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=2,
                    logit_bias=logit_bias)
                break
            except (openai.error.RateLimitError,
                    openai.error.APIConnectionError, openai.error.APIError):
                time.sleep(6)
        if response is None:
            raise RuntimeError("Failed to query OpenAI API.") 
        return response["choices"][0]["text"]

    @staticmethod
    def _raw_to_llm_response(raw_response: Dict[str, Any], 
                             prompt: str,
                             temperature: float, 
                             stop_token: str,
                             num_completions: int) -> LLMResponse:
        text = raw_response["text"]

        text = text.strip()
        text = text.replace("<|im_end|>", "")
        text = text.replace("<|im_sep|>", "")

        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        return LLMResponse(prompt,
                           text,
                           prompt_info=prompt_info,
                           other_info=raw_response.copy())

    def evaluate_completion(self, prefix: str, suffix:str, average=True) -> float:
        while True:
            try:
                return self._evaluate_gpt_completion(prefix, suffix, average)
            except Exception as runtime_error:
                if "This model's maximum context length is 8001 tokens" in str(runtime_error):
                    raise runtime_error
                else:
                    time.sleep(3)
                    logger.warning(str(runtime_error))
                    logger.info("retrying...") 

    def _evaluate_completion(self, 
                                prefix: str, 
                                suffix: str, 
                                average:bool) -> float:
        _prompt = f"{prefix}{suffix}"
        response = openai.Completion.create(
            engine=self._model_name,
            prompt=_prompt,
            echo=True,
            logprobs=1,
            temperature=0,
            max_tokens=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
        offsets = response["choices"][0]["logprobs"]["text_offset"]
        try:
            suffix_start_token_id = offsets.index(len(prefix))
        except ValueError:
            # find the closest token
            suffix_start_token_id = min(range(len(offsets)), key=lambda i: abs(offsets[i] - len(prefix)))
            logger.warning("suffix_start_token_id not found, using closest token")

        # corner case: prefix is empty
        if suffix_start_token_id == 0: 
            assert logprobs[0] is None
            logprobs[0] = 0

        if average:
            suffix_logit = sum(logprobs[suffix_start_token_id:]) / len(logprobs[suffix_start_token_id:])
        else:
            suffix_logit = sum(logprobs[suffix_start_token_id:])

        return suffix_logit


class PaLM(LargeLanguageModel):
    PALM_API_KEY = os.environ.get("PALM_API_KEY")

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        palm.configure(api_key=self.PALM_API_KEY)
    
    def get_id(self) -> str:
        return f"palm_{self._model_name}"
    
    def _sample_completions(self, 
                            prompt: str, 
                            temperature: float, 
                            stop_token: str, 
                            num_completions: int = 1) -> List[LLMResponse]:
        
        response = None
        for _ in range(12):
            try:
                response = palm.generate_text(prompt=prompt,
                                            model=self._model_name,
                                            max_output_tokens=FLAGS.max_tokens,
                                            temperature=temperature,
                                            stop_sequences=[stop_token],
                                            candidate_count=num_completions)
            except palm_exceptions.ResourceExhausted as e:
                logger.debug("ResourceExhausted, waiting..")
                # logger.warn(f"Error {str(e)}")
                time.sleep(120)
        
        if response is None:
            raise RuntimeError("Failed to query Palm API.")
        
        if len(response.candidates) == 0:
            logger.warning("Palm returned empty response.")
            return [LLMResponse(prompt, "", prompt_info={"temperature": temperature}, other_info={})]
        
        assert len(response.candidates) == num_completions
        return [
            self._raw_to_llm_response(r, prompt, temperature,stop_token, num_completions)
            for r in response.candidates
        ]

    @staticmethod
    def _raw_to_llm_response(raw_response: Dict[str, Any], 
                             prompt: str,
                             temperature: float, 
                             stop_token: str,
                             num_completions: int) -> LLMResponse:
        text = raw_response["output"]
        text = text.strip()
        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        return LLMResponse(prompt,
                           text,
                           prompt_info=prompt_info,
                           other_info=raw_response.copy())


class ChatGPT(LargeLanguageModel):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        openai.api_key = self.OPENAI_API_KEY

    def get_id(self) -> str:
        return f"chatgpt_{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            temperature: float,
            stop_token: str,
            num_completions: int = 1) -> List[LLMResponse]:  
        """
        Note that sys and user prompt are assumed to be separated by a newline.
        """
        
        chunks = prompt.split("\n")
        sys_prompt = chunks[0]
        user_prompt = "\n".join(chunks[1:])

        response = None
        for _ in range(6):
            try:
                response = openai.ChatCompletion.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    stop=stop_token,
                    max_tokens=FLAGS.max_tokens,
                    frequency_penalty=FLAGS.freq_penalty,
                    n=num_completions)
                # Successfully queried, so break.
                break
            except (openai.error.RateLimitError,
                    openai.error.APIConnectionError, openai.error.APIError):
                # Wait for 60 seconds if this limit is reached. Hopefully rare.
                time.sleep(6)

        if response is None:
            raise RuntimeError("Failed to query OpenAI API.")
        
        assert len(response["choices"]) == num_completions
        return [
            self._raw_to_llm_response(r, prompt, temperature,stop_token, num_completions)
            for r in response["choices"]
        ]

    @staticmethod
    def _raw_to_llm_response(raw_response: Dict[str, Any], 
                             prompt: str,
                             temperature: float, 
                             stop_token: str,
                             num_completions: int) -> LLMResponse:
        text = raw_response["message"]["content"]
        prompt_info = {
            "temperature": temperature,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        return LLMResponse(prompt,
                           text,
                           prompt_info=prompt_info,
                           other_info=raw_response.copy())


def setup_llm(engine):
    split_point = engine.index("/")
    platform, engine_short = engine[:split_point], engine[split_point+1:]
    if platform == "azure":
        llm = GPT(engine_short)
    elif platform == "google":
        llm = PaLM(engine_short)
    elif platform == "openai":
        if engine_short == "code-davinci-002":
            llm = GPT(engine_short, use_azure=False)
        else:
            llm = ChatGPT(engine_short)
    else:
        raise NotImplementedError(f"platform {platform} not supported")
    return llm