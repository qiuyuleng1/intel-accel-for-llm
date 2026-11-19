"""Single-thread tokenization speed of `vllm bench serve` random prompts.

vLLM 0.29's API server tokenizes text prompts in a ThreadPoolExecutor with
renderer_num_workers threads (default 1), so its prompt-token intake is bounded by
one thread's tokenizer speed. This times tokenizer(text) on random-token text like
the random dataset's (random token ids decoded to text), 32k / 128k / 256k tokens.

Usage (CPU only): python3 tokenizer_speed.py <model dir>
"""

import random
import sys
import time

from transformers import AutoTokenizer

LENGTHS = (32000, 128000, 256000)
REPEATS = 3


def main():
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
    special_ids = set(tokenizer.all_special_ids)
    vocab_ids = []
    for token_id in range(tokenizer.vocab_size):
        if token_id not in special_ids:
            vocab_ids.append(token_id)
    rng = random.Random(0)
    print(f"tokenizer {type(tokenizer).__name__}, vocab {tokenizer.vocab_size}")
    for length in LENGTHS:
        token_ids = []
        for _ in range(length):
            token_ids.append(rng.choice(vocab_ids))
        text = tokenizer.decode(token_ids)
        seconds = []
        encoded_len = 0
        for _ in range(REPEATS):
            start = time.perf_counter()
            encoded_len = len(tokenizer(text)["input_ids"])
            seconds.append(time.perf_counter() - start)
        best = min(seconds)
        print(
            f"{length} tokens: text {len(text)} chars, re-encoded {encoded_len} tokens, "
            f"encode {best * 1000:.0f} ms (min of {REPEATS}), {encoded_len / best:.0f} tokens/s"
        )


if __name__ == "__main__":
    main()
