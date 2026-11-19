"""Does more tokenizer threads give more tokens/s (does encode release the GIL)?

`--renderer-num-workers N` only helps the Python API server if N threads can
tokenize at the same time. This encodes the same random-token text (like
`vllm bench serve` random prompts) with 1, 2, 4, 8 threads, each thread
encoding THREAD_REPEATS prompts, and prints the total tokens/s. If encode holds
the GIL, total tokens/s stays flat as threads grow.

Usage (CPU only, pin to >= 8 cores): python3 tokenizer_thread_scaling.py <model dir>
"""

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

PROMPT_TOKENS = 128000
THREAD_COUNTS = (1, 2, 4, 8)
THREAD_REPEATS = 4


def encode_repeatedly(tokenizer, text):
    encoded_tokens = 0
    for _ in range(THREAD_REPEATS):
        encoded_tokens += len(tokenizer(text)["input_ids"])
    return encoded_tokens


def main():
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
    special_ids = set(tokenizer.all_special_ids)
    vocab_ids = []
    for token_id in range(tokenizer.vocab_size):
        if token_id not in special_ids:
            vocab_ids.append(token_id)
    rng = random.Random(0)
    token_ids = []
    for _ in range(PROMPT_TOKENS):
        token_ids.append(rng.choice(vocab_ids))
    text = tokenizer.decode(token_ids)
    encode_repeatedly(tokenizer, text)

    import tokenizers

    print(f"tokenizers {tokenizers.__version__}, prompt {PROMPT_TOKENS} tokens")
    for thread_count in THREAD_COUNTS:
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            start = time.perf_counter()
            futures = []
            for _ in range(thread_count):
                futures.append(executor.submit(encode_repeatedly, tokenizer, text))
            total_tokens = 0
            for future in futures:
                total_tokens += future.result()
            seconds = time.perf_counter() - start
        print(
            f"{thread_count} threads: {total_tokens} tokens in {seconds:.2f} s, "
            f"{total_tokens / seconds:.0f} tokens/s total"
        )


if __name__ == "__main__":
    main()
