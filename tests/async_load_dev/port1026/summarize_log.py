"""Summarize a 1026 test server log (dev helper, not part of the PR).

usage: python3 summarize_log.py <vllm_log> [start_byte]
Prints, for the part of the log after start_byte:
  lookups: get_num_new_matched_tokens lines (one per scheduling attempt);
  reqs / ext_hit / async: distinct requests (last decision of each), those with
    externally-cached tokens > 0, and those with async=True;
  async_N: async_load_layers of async requests -> count;
  raise_lines / raised_reqs / with_sync_batch / moved_tasks: raise_trace lines of the
    traced connector, summed over TP ranks;
  errors: Traceback / RuntimeError / ValueError / "IAXL check failed" lines.
"""

import re
import sys

path = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
with open(path, "rb") as f:
    f.seek(start)
    text = f.read().decode(errors="replace")

decision_re = re.compile(
    r"req-(\S+), externally-cached tokens: (\d+), .*async=(\w+), selected_load_layers=(-?\d+), "
    r"async_load_layers=(-?\d+)"
)
raise_re = re.compile(r"raise_trace rank=(\d+) raised_reqs=(\d+) sync_batch=(\d) bundles=(\d+) "
                      r"moved=(\d+)")
lookups = 0
decisions = {}
raise_lines = 0
raised_reqs = 0
with_sync = 0
moved = 0
errors = 0
for line in text.split("\n"):
    if "get_num_new_matched_tokens" in line:
        m = decision_re.search(line)
        if m:
            lookups += 1
            decisions[m.group(1)] = (int(m.group(2)), m.group(3) == "True", int(m.group(5)))
    m = raise_re.search(line)
    if m:
        raise_lines += 1
        raised_reqs += int(m.group(2))
        with_sync += int(m.group(3))
        moved += int(m.group(5))
    if re.search(r"Traceback|RuntimeError|ValueError|IAXL check failed", line):
        errors += 1

ext_hit = 0
async_count = 0
async_n = {}
for external_tokens, is_async, layers in decisions.values():
    if external_tokens > 0:
        ext_hit += 1
    if is_async:
        async_count += 1
        async_n[layers] = async_n.get(layers, 0) + 1

print(f"lookups={lookups} reqs={len(decisions)} ext_hit={ext_hit} async={async_count} "
      f"async_N={dict(sorted(async_n.items()))} raise_lines={raise_lines} "
      f"raised_reqs={raised_reqs} with_sync_batch={with_sync} moved_tasks={moved} "
      f"errors={errors}")
