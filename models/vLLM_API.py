# Licensed under the MIT license.

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import math


def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=True, max_num_seqs=256):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if half_precision:
        #llm = LLM(
        #    model=model_ckpt,
        #    dtype="half",
        #    tensor_parallel_size=2,
        #    seed=seed,
        #    speculative_model="/home/asperger/models/Qwen2.5-0.5B-Instruct",
        #    speculative_draft_tensor_parallel_size=1,
        #    num_speculative_tokens=4,
        #    trust_remote_code=True,
        #    max_num_seqs=max_num_seqs,
        #    swap_space=8,
        #)
        #llm = LLM(
        #    model=model_ckpt,
        #    dtype="half",
        #    tensor_parallel_size=1,
        #    seed=seed,
        #    speculative_model="[ngram]",
        #    speculative_draft_tensor_parallel_size=1,
        #    num_speculative_tokens=4,
        #    ngram_prompt_lookup_max=4,
        #    trust_remote_code=True,
        #    max_num_seqs=max_num_seqs,
        #    swap_space=8,
        #)
        llm = LLM(
            model=model_ckpt,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=8,
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=8,
        )

    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)
    return output


if __name__ == "__main__":
    model_ckpt = "mistralai/Mistral-7B-v0.1"
    tokenizer, model = load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False)
    input = "What is the meaning of life?"
    output = generate_with_vLLM_model(model, input)
    breakpoint()
    print(output[0].outputs[0].text)
    
