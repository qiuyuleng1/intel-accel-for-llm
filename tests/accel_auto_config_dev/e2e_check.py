"""Step 3 check of one vLLM run started by examples/kvshrink-vllm-serve.sh: reads <run_dir>/serve.log and
<run_dir>/outputs.json. Lines before "vLLM configuration:" come from setvars.sh and accel_auto_config,
later ones from vLLM.
Usage: e2e_check.py <run_dir> <qat|iaa|none> <TP size>"""

import json
import os
import re
import sys

DSA_SUMMARY = re.compile(r"\[accel_auto_config\] DSA crossover H2D (\d+) KB, D2H (\d+) KB")
QAT_SUMMARY = re.compile(r"\[accel_auto_config\] QAT segment sizes by decompress GB/s: (.*), [\d.]+ s")
DONE_LINE = re.compile(r"\[accel_auto_config\] done in ([\d.]+) s")
PRINTED_VALUE = re.compile(r"^  (IAXL_DSA_H2D_CROSSOVER_KB|IAXL_DSA_D2H_CROSSOVER_KB|IAXL_ZIP_SEGMENT_SIZE_KB)=(\S+)$")
RANKED_ITEM = re.compile(r"(\d+) KB [\d.]+")
KV_ZIP_LINE = re.compile(r"\[kv_zip\] block_size=(\d+) B segment_size=(\d+) B segments=(\d+)")
BLOCK_SIZE_LINE = re.compile(r"copy_chunks_batch: contiguous block size = ([\d.]+) KB")
BACKEND_LINE = re.compile(r"copy_chunks_batch (H2D|D2H): using (Intel DSA|CUDA) backend")
HIT_LINE = re.compile(r"get_num_new_matched_tokens, req-(\S+), externally-cached tokens: (\d+)")


def read_lines(path: str) -> list[str]:
    with open(path, errors="replace") as log_file:
        return log_file.read().splitlines()


def expected_segment(block_bytes: int, ranked_kb: list[int]) -> int:
    for segment_kb in ranked_kb:
        if segment_kb * 1024 <= block_bytes:
            return segment_kb * 1024
    return block_bytes


def main() -> int:
    run_dir = sys.argv[1]
    zip_case = sys.argv[2]
    tp_size = int(sys.argv[3])
    serve_lines = read_lines(os.path.join(run_dir, "serve.log"))

    config_index = len(serve_lines)
    for index in range(len(serve_lines)):
        if serve_lines[index] == "vLLM configuration:":
            config_index = index
            break
    setvars_lines = serve_lines[:config_index]
    vllm_lines = serve_lines[config_index + 1:]

    dsa_summaries = []
    qat_ranked_kb = []
    qat_summary_count = 0
    done_seconds = ""
    for line in setvars_lines:
        match = DSA_SUMMARY.search(line)
        if match:
            dsa_summaries.append((int(match.group(1)), int(match.group(2))))
        match = QAT_SUMMARY.search(line)
        if match:
            qat_summary_count += 1
            for item in RANKED_ITEM.finditer(match.group(1)):
                qat_ranked_kb.append(int(item.group(1)))
        match = DONE_LINE.search(line)
        if match:
            done_seconds = match.group(1)
    printed = {}
    for line in vllm_lines:
        match = PRINTED_VALUE.search(line)
        if match:
            printed[match.group(1)] = match.group(2)

    segment_list_kb = []
    if printed.get("IAXL_ZIP_SEGMENT_SIZE_KB", "unset") != "unset":
        for item in printed["IAXL_ZIP_SEGMENT_SIZE_KB"].split(","):
            segment_list_kb.append(int(item))
    kv_zip_rows = []
    block_sizes_kb = []
    backends = {"H2D": [], "D2H": []}
    for line in vllm_lines:
        match = KV_ZIP_LINE.search(line)
        if match:
            kv_zip_rows.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
        match = BLOCK_SIZE_LINE.search(line)
        if match:
            block_sizes_kb.append(float(match.group(1)))
        match = BACKEND_LINE.search(line)
        if match:
            backends[match.group(1)].append(match.group(2))

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

    print(f"accel_auto_config done in: {done_seconds} s")
    print(f"DSA summaries (H2D KB, D2H KB): {dsa_summaries}")
    print(f"QAT ranked KB: {qat_ranked_kb}")
    print(f"printed by setvars.sh: {printed}")
    print(f"vLLM [kv_zip] lines (block B, segment B, segments): {kv_zip_rows}")
    print(f"vLLM contiguous block sizes (KB, one per process): {block_sizes_kb}")
    print(f"vLLM backends (direction: one per process): {backends}")
    print(f"Traceback lines: {traceback_count}")
    print("round prompt prompt_tokens hit_tokens text")
    for reply in replies:
        print(f"  {reply['round']} {reply['prompt_index']} {reply['prompt_tokens']} "
              f"{reply['hit_tokens']} {reply['text']!r}")

    checks = []
    checks.append(("one DSA summary line before vLLM", len(dsa_summaries) == 1))
    crossover_kb = {"H2D": 0, "D2H": 0}
    if len(dsa_summaries) == 1:
        crossover_kb["H2D"] = dsa_summaries[0][0]
        crossover_kb["D2H"] = dsa_summaries[0][1]
    checks.append(("printed crossovers match the DSA summary",
                   printed.get("IAXL_DSA_H2D_CROSSOVER_KB") == str(crossover_kb["H2D"])
                   and printed.get("IAXL_DSA_D2H_CROSSOVER_KB") == str(crossover_kb["D2H"])))
    if zip_case == "qat":
        ranked_text = []
        for segment_kb in qat_ranked_kb:
            ranked_text.append(str(segment_kb))
        checks.append(("one QAT summary line ranking 32..1024 KB",
                       qat_summary_count == 1 and sorted(qat_ranked_kb) == [32, 64, 128, 256, 512, 1024]))
        checks.append(("printed IAXL_ZIP_SEGMENT_SIZE_KB is the QAT ranking",
                       printed.get("IAXL_ZIP_SEGMENT_SIZE_KB") == ",".join(ranked_text)))
    elif zip_case == "iaa":
        checks.append(("no QAT summary line", qat_summary_count == 0))
        checks.append(("printed IAXL_ZIP_SEGMENT_SIZE_KB=2048", printed.get("IAXL_ZIP_SEGMENT_SIZE_KB") == "2048"))
    else:
        checks.append(("no QAT summary line", qat_summary_count == 0))
        checks.append(("printed IAXL_ZIP_SEGMENT_SIZE_KB=unset", printed.get("IAXL_ZIP_SEGMENT_SIZE_KB") == "unset"))
    segments_follow_rule = len(kv_zip_rows) > 0
    for block_bytes, segment_bytes, segments in kv_zip_rows:
        if segment_bytes != expected_segment(block_bytes, segment_list_kb):
            segments_follow_rule = False
        if segments != max(1, block_bytes // segment_bytes):
            segments_follow_rule = False
    checks.append(("vLLM segment sizes follow IAXL_ZIP_SEGMENT_SIZE_KB", segments_follow_rule))
    block_kb = 0.0
    if block_sizes_kb:
        block_kb = block_sizes_kb[0]
    for direction in ("H2D", "D2H"):
        expected = "Intel DSA"
        if crossover_kb[direction] > 0 and block_kb >= crossover_kb[direction]:
            expected = "CUDA"
        all_expected = len(backends[direction]) == tp_size
        for backend in backends[direction]:
            if backend != expected:
                all_expected = False
        checks.append((f"{direction}: {tp_size} ranks use {expected} (block {block_kb} KB, "
                       f"crossover {crossover_kb[direction]} KB)", all_expected))
    last_round = replies[-1]["round"]
    every_prompt_hit = True
    correct = True
    for reply in replies:
        if reply["round"] == last_round and not reply["hit_tokens"]:
            every_prompt_hit = False
        # e2e_requests.py asks for item 17 of report <prompt_index>; its value follows this formula.
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
