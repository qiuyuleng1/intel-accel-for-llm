import json
import os
import re
import sys

ZIP_SUMMARY = re.compile(r"\[zip_calibration\] (\w+) segment sizes by decompress GB/s: (.*), \d+ workers")
DSA_SUMMARY = re.compile(r"\[dsa_calibration\] crossover H2D (\d+) KB, D2H (\d+) KB")
KV_ZIP_LINE = re.compile(r"\[kv_zip\] block_size=(\d+) B segment_size=(\d+) B segments=(\d+)")
RANKED_ITEM = re.compile(r"(\d+) KB [\d.]+")
HIT_LINE = re.compile(r"get_num_new_matched_tokens, req-(\S+), externally-cached tokens: (\d+)")


def read_lines(path: str) -> list:
    with open(path, errors="replace") as log_file:
        return log_file.read().splitlines()


def expected_segment(block_bytes: int, ranked_kb: list) -> int:
    for segment_kb in ranked_kb:
        if segment_kb == 0:
            break
        if segment_kb * 1024 <= block_bytes:
            return segment_kb * 1024
    return block_bytes


def main() -> int:
    run_dir = sys.argv[1]
    expected_backend = sys.argv[2]
    serve_lines = read_lines(os.path.join(run_dir, "serve.log"))
    vllm_log_lines = read_lines(os.path.join(run_dir, "log.kvshrink-vllm"))

    zip_summary_index = -1
    zip_backend = ""
    ranked_kb = []
    dsa_summary_count = 0
    for index in range(len(serve_lines)):
        if DSA_SUMMARY.search(serve_lines[index]):
            dsa_summary_count += 1
        match = ZIP_SUMMARY.search(serve_lines[index])
        if match:
            zip_summary_index = index
            zip_backend = match.group(1)
            for item in RANKED_ITEM.finditer(match.group(2)):
                ranked_kb.append(int(item.group(1)))
    calibration_end = zip_summary_index
    for index in range(len(serve_lines)):
        if DSA_SUMMARY.search(serve_lines[index]) and index > calibration_end:
            calibration_end = index

    kv_zip_rows = []
    for line in serve_lines[calibration_end + 1:]:
        match = KV_ZIP_LINE.search(line)
        if match:
            kv_zip_rows.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))

    zip_summary_in_vllm_log = False
    for line in vllm_log_lines:
        if ZIP_SUMMARY.search(line):
            zip_summary_in_vllm_log = True

    hits_by_log_id = {}
    traceback_count = 0
    for line in serve_lines:
        if "Traceback" in line:
            traceback_count += 1
        match = HIT_LINE.search(line)
        if match:
            hits_by_log_id[match.group(1)] = int(match.group(2))
    with open(os.path.join(run_dir, "outputs.json")) as outputs_file:
        replies = json.load(outputs_file)
    for reply in replies:
        reply["hit_tokens"] = None
        for log_id, hit_tokens in hits_by_log_id.items():
            if log_id.startswith(reply["response_id"]):
                reply["hit_tokens"] = hit_tokens

    print(f"zip calibration backend: {zip_backend or '-'}, ranked KB: {ranked_kb}")
    print(f"dsa calibration summary lines: {dsa_summary_count}")
    print(f"vLLM [kv_zip] lines (block B, segment B, segments): {kv_zip_rows}")
    print(f"Traceback lines: {traceback_count}")
    print("round prompt prompt_tokens hit_tokens text")
    for reply in replies:
        print(f"  {reply['round']} {reply['prompt_index']} {reply['prompt_tokens']} "
              f"{reply['hit_tokens']} {reply['text']!r}")

    checks = []
    checks.append(("one DSA calibration summary line", dsa_summary_count == 1))
    if expected_backend == "none":
        checks.append(("no zip calibration line", zip_summary_index == -1))
    else:
        checks.append((f"zip calibration tested {expected_backend}", zip_backend == expected_backend))
        checks.append(("zip calibration ranked 8 sizes", sorted(ranked_kb) == [16, 32, 64, 128, 256, 512, 1024, 2048]))
        checks.append(("zip summary line also in log.kvshrink-vllm", zip_summary_in_vllm_log))
    segments_follow_rule = len(kv_zip_rows) > 0
    for block_bytes, segment_bytes, segments in kv_zip_rows:
        if segment_bytes != expected_segment(block_bytes, ranked_kb):
            segments_follow_rule = False
        if segments != max(1, block_bytes // segment_bytes):
            segments_follow_rule = False
    checks.append(("vLLM segment sizes follow the calibrated list", segments_follow_rule))
    last_round = replies[-1]["round"]
    every_prompt_hit = True
    correct = True
    for reply in replies:
        if reply["round"] == last_round and not reply["hit_tokens"]:
            every_prompt_hit = False
        expected_value = (reply["prompt_index"] * 7919 + 17 * 104729) % 100000
        if str(expected_value) not in reply["text"]:
            correct = False
    checks.append((f"every prompt hits in round {last_round}", every_prompt_hit))
    checks.append(("every reply contains the correct value", correct))
    checks.append(("no Traceback", traceback_count == 0))

    all_passed = True
    print("== checks")
    for description, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {description}")
        if not passed:
            all_passed = False
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
