# Licensed under the MIT license.

import os
import os
import threading
import time
from tqdm import tqdm
import concurrent.futures
from openai import AzureOpenAI, OpenAI

thread_lock = threading.Lock()

openai_api_key = "your_openai_api_key"

client = OpenAI(
    api_key=openai_api_key,
)


max_threads = 10


def load_OpenAI_model(model):
    return None, model


def generate_with_OpenAI_model(
    prompt,
    n=1,
    model_ckpt="gpt-4o",
    max_tokens=512,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
):
    messages = [{"role": "user", "content": prompt}]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "n": n,
    }

    ans, timeout = "", 5
    while not ans:
        try:
            time.sleep(timeout)
            completion = client.chat.completions.create(messages=messages, **parameters)
            ans = [choice.message.content for choice in completion.choices]

        except Exception as e:
            print(e)
        if not ans:
            timeout = timeout * 2

            if timeout > 20:
                timeout = 1
            try:
                print(ans)
                print(messages)
                print(len(messages[0]['content']))
                print(f"Will retry after {timeout} seconds ...")
            except:
                pass
    return ans


def generate_n_with_OpenAI_model(
    prompt,
    n=1,
    model_ckpt="gpt-4o",
    max_tokens=512,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
    max_threads=10,
    disable_tqdm=True,
):
    preds = generate_with_OpenAI_model(prompt, n, model_ckpt, max_tokens, temperature, top_k, top_p, stop)
    return preds

def generate_prompts_with_OpenAI_model(
    prompts: list,
    n=1,
    model_ckpt="gpt-4o",
    max_tokens=512,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
    max_threads=10,
    disable_tqdm=True,
):
    preds = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(generate_with_OpenAI_model, prompt, n, model_ckpt, max_tokens, temperature, top_k, top_p, stop)
            for prompt in prompts
        ]
        for i, future in tqdm(
            enumerate(concurrent.futures.as_completed(futures)),
            total=len(futures),
            desc="running evaluate",
            disable=disable_tqdm,
        ):
            ans = future.result()
            preds.append(ans)
    return preds
