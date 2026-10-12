#!/bin/bash
# usage: run-queue-and-correctness.sh <scheme> <s1|s2> ...   (dev helper)
# For each scenario: queue test with VLLM_BATCH_INVARIANT=0, correctness with =1.
# Env: C = container (default iaxl.vllm.qiuyu), PREFIX = result dir prefix.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
HERE="$REPO/tests/async_load_dev"
C=${C:-iaxl.vllm.qiuyu}
SCHEME=$1; shift
cd "$REPO"
for S in "$@"; do
    if [ "$S" = s1 ]; then CFG="0 4 0-3:0,4-:4"; else CFG="1 -1 0-3:0,4-:4"; fi
    TAG=${PREFIX:-run}-${SCHEME%%_*}
    echo "===== $SCHEME $S queue (BI=0) connector_md5=$(md5sum kvshrink/kvshrink_connector.py | cut -c1-8) ====="
    VLLM_BATCH_INVARIANT=0 C=$C "$HERE/restart-vllm.sh" $CFG "$SCHEME" | tail -2
    docker exec -w "$REPO" "$C" bash -c "export no_proxy=localhost,127.0.0.1 PORT=8010 IAXL_API_CONTROLLER_PORT=18710 KVSHRINK_ASYNC_LOAD_SCHEME=$SCHEME; ./tests/async-load-queue-test.sh $S _data/queue-test/$TAG-$S > _data/queue-test/$TAG-$S.out 2>&1; echo queue_exit=\$?; sed -n '/^{/,\$p' _data/queue-test/$TAG-$S.out | grep -vE '^\s+\"(scheme|scenario)\"'"
    echo "  activate lines with held!='': $(grep -ah 'activate reqs' _data/queue-test/$TAG-$S/vllm.log | grep -acvE 'held=$')"
    echo "===== $SCHEME $S correctness (BI=1) connector_md5=$(md5sum kvshrink/kvshrink_connector.py | cut -c1-8) ====="
    VLLM_BATCH_INVARIANT=1 C=$C "$HERE/restart-vllm.sh" $CFG "$SCHEME" | tail -2
    docker exec -w "$REPO" "$C" bash -c "export no_proxy=localhost,127.0.0.1; python3 tests/async_load_correctness_check.py --out _data/queue-test/$TAG-$S-correctness > /dev/null; echo correctness_exit=\$?; python3 -c \"import json;r=json.load(open('_data/queue-test/$TAG-$S-correctness/report.json'));print(r['verdict'],'ref_ext',r['reference']['externally_cached'],'hits',[(h['externally_cached'],h['identical'],len(h['bad_chars'])) for h in r['hits']])\""
    echo "  activate lines with held!='': $(grep -a 'activate reqs' _data/queue-test/$TAG-$S-correctness/connector_log.txt | grep -acvE 'held=$')"
done
