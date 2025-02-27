# Licensed under the MIT license.

import sys

sys.path.append(".")

from typing import List, Dict

try:
    from models.vLLM_API import generate_with_vLLM_model
except:
    pass

try:
    from models.OpenAI_API import generate_n_with_OpenAI_model, generate_prompts_with_OpenAI_model
except:
    pass

try:
    from models.HuggingFace_API import generate_with_HF_model
except:
    pass

class IO_System:
    """Input/Output system"""

    def __init__(self, args, tokenizer, model) -> None:
        self.api = args.api
        if self.api == "together":
            assert tokenizer is None and model is None
        elif self.api == "gpt3.5-turbo":
            assert tokenizer is None and isinstance(model, str)
        self.model_ckpt = args.model_ckpt
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.tokenizer = tokenizer
        self.model = model

        self.call_counter = 0
        self.token_counter = 0

    def generate(self, model_input, max_tokens: int, num_return: int, stop_tokens):
        if self.api == "gpt-4o" and len(stop_tokens) > 4:
            stop_tokens = stop_tokens[:4]

        if isinstance(model_input, str):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
                io_output_list = [o.text for o in vllm_response[0].outputs]
                self.call_counter += 1
                self.token_counter += sum([len(o.token_ids) for o in vllm_response[0].outputs])
            elif self.api == "gpt-4o":
                gpt_response = generate_n_with_OpenAI_model(
                    prompt=model_input,
                    n=num_return,
                    model_ckpt=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stop=stop_tokens,
                )
                io_output_list = gpt_response
                self.call_counter += num_return
                self.token_counter += 0
            elif self.api == "huggingface":
                hf_response = generate_with_HF_model(
                    self.tokenizer,
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    num_beams=num_return,
                    max_tokens=max_tokens
                )
                io_output_list = hf_response
                self.call_counter += num_return
                self.token_counter += 0
            elif self.api == "debug":
                io_output_list = ["Debug: The answer is generated with debug mode, 233." for _ in range(num_return)]
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")
        elif isinstance(model_input, list):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
                io_output_list = [
                    [o.text for o in resp_to_single_input.outputs] for resp_to_single_input in vllm_response
                ]
                self.call_counter += 1
                self.token_counter += sum(
                    [
                        sum([len(o.token_ids) for o in resp_to_single_input.outputs])
                        for resp_to_single_input in vllm_response
                    ]
                )
            elif self.api == "gpt-4o":
                io_output_list = generate_prompts_with_OpenAI_model(
                    prompts=model_input,
                    n=num_return,
                    model_ckpt=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stop=stop_tokens,
                )
                self.call_counter += num_return * len(model_input)
                self.token_counter += 0
            elif self.api == "debug":
                io_output_list = [
                    ["Debug: The answer is generated with debug mode, 233." for _ in range(num_return)]
                    for _ in model_input
                ]
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")

        return io_output_list
