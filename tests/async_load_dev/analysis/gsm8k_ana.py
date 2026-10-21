# Usage: gsm8k_ana.py <dir> [<dir> ...]   (each dir has run1/ run2/ and run*.vllm.log)
# Per run: correct/total, external-cache hits and loaded tokens, selected N, garbled
# chars. Per dir: run1 vs run2 per-question comparison.
import glob
import json
import re
import sys

conn_re = re.compile(
    r"get_num_new_matched_tokens, req-\S+, externally-cached tokens: (\d+), .*async=(\w+), "
    r"selected_load_layers=(-?\d+)"
)
bad_ctrl = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def load_reviews(run_dir):
    path = glob.glob(f"{run_dir}/*/reviews/*/gsm8k_main.jsonl")[0]
    out = {}
    for line in open(path):
        r = json.loads(line)
        score = r["sample_score"]["score"]
        text = ""
        for m in r["messages"]:
            if m.get("role") == "assistant":
                text = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        out[r["index"]] = (score["value"]["acc"], str(score.get("extracted_prediction")), text)
    return out


def run_stats(d, run):
    reviews = load_reviews(f"{d}/{run}")
    correct = 0
    garbled = 0
    for acc, _, text in reviews.values():
        correct += acc == 1.0
        garbled += text.count("\ufffd") + len(bad_ctrl.findall(text))
    hits = []
    n_sel = {}
    lookups = 0
    for line in open(f"{d}/{run}.vllm.log", errors="replace"):
        m = conn_re.search(line)
        if not m:
            continue
        lookups += 1
        tokens = int(m.group(1))
        hits.append(tokens)
        key = m.group(3) if m.group(2) == "True" else "0(sync)"
        n_sel[key] = n_sel.get(key, 0) + 1
    n_hit = 0
    for t in hits:
        n_hit += t > 0
    print(
        f"  {run}: correct {correct}/{len(reviews)} acc={correct / len(reviews):.3f} "
        f"ext_hit {n_hit}/{lookups} tokens avg/min/max {sum(hits) / len(hits):.0f}/{min(hits)}/{max(hits)} "
        f"N={n_sel} garbled={garbled}"
    )
    return reviews


for d in sys.argv[1:]:
    print(d)
    r1 = run_stats(d, "run1")
    r2 = run_stats(d, "run2")
    same_ans = 0
    same_text = 0
    flips = []
    for idx in r1:
        same_ans += r1[idx][1] == r2[idx][1]
        same_text += r1[idx][2] == r2[idx][2]
        if r1[idx][0] != r2[idx][0]:
            flips.append(f"{idx}:{'r1' if r1[idx][0] == 1.0 else 'r2'}")
    print(f"  run1 vs run2: same final answer {same_ans}/200, identical text {same_text}/200, "
          f"correctness differs {len(flips)} ({' '.join(flips)}; r1/r2 = which run was right)")
