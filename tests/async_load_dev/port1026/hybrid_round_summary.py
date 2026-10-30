"""Per-round summary of correctness-1026.sh results without the byte-identity criterion
(dev helper, not part of the PR). For GDN hybrid models VLLM_BATCH_INVARIANT cannot be
on, so hit outputs may legitimately differ from the reference; this prints, per round:
reference external tokens, hits, external tokens of the hits, how many hits equal the
reference, where the others first differ (character offset), how many outputs (reference
included) answer the prompt's question correctly (sensor and value of Record 0042, which
sits in the loaded prefix), U+FFFD / control characters, and the distinct output texts so
a human can judge whether they are reasonable.

usage: python3 hybrid_round_summary.py <correctness result dir>
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from async_load_correctness_check import bad_chars  # noqa: E402


def first_difference(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for index in range(limit):
        if a[index] != b[index]:
            return index
    return limit if len(a) != len(b) else -1


def main() -> int:
    out = sys.argv[1]
    rounds = sorted(name for name in os.listdir(out) if name.startswith("round") and
                    os.path.isdir(os.path.join(out, name)))
    for name in rounds:
        report = json.load(open(os.path.join(out, name, "report.json")))
        responses = json.load(open(os.path.join(out, name, "responses.json")))
        ref_text = responses["reference"]["choices"][0]["text"]
        record = re.search(r"Record 0042: sensor (\S+) measured (\d+) units\.",
                           responses["request"]["prompt"])
        expected = f"{record.group(1)} measured {record.group(2)} units"
        hit_texts = []
        for hit in responses["hits"]:
            hit_texts.append(hit["choices"][0]["text"])
        external = []
        for hit in report["hits"]:
            external.append(hit["externally_cached"])
        diffs = []
        for text in hit_texts:
            offset = first_difference(ref_text, text)
            if offset >= 0:
                diffs.append(offset)
        bad = len(bad_chars(ref_text))
        for text in hit_texts:
            bad += len(bad_chars(text))
        correct = 0
        for text in [ref_text] + hit_texts:
            # The model may prefix the answer with "sensor"; only the first line is the answer.
            if expected in text.strip().split("\n")[0]:
                correct += 1
        print(f"{name}: prompt_tokens={report['prompt_tokens']} "
              f"ref_external={report['reference']['externally_cached']} hits={len(hit_texts)} "
              f"hit_external={sorted(set(external))} identical={len(hit_texts) - len(diffs)} "
              f"first_diff_chars={sorted(diffs)} answer_correct={correct}/{len(hit_texts) + 1} "
              f"bad_chars={bad}")
        print(f"  expected answer: {expected!r}")
        print(f"  reference: {ref_text!r}")
        distinct = []
        for text in hit_texts:
            if text != ref_text and text not in distinct:
                distinct.append(text)
        for text in distinct:
            print(f"  hit (x{hit_texts.count(text)}): {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
