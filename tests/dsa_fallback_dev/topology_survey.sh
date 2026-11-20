#!/bin/bash
# Read-only survey of GPU / DSA / PCIe placement on this host, to explain the 8-rank calibration speeds.
# Usage: bash tests/dsa_fallback_dev/topology_survey.sh > <file>
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

echo "=== nvidia-smi topo -m"
nvidia-smi topo -m

echo "=== GPU PCI path, NUMA node, link"
for bus_id in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
    # nvidia-smi prints 00000000:38:00.0, sysfs uses 0000:38:00.0
    bdf=$(echo "${bus_id:4}" | tr 'A-F' 'a-f')
    index=$(nvidia-smi --query-gpu=index --format=csv,noheader -i "$bus_id")
    echo "GPU$index $bdf numa=$(cat /sys/bus/pci/devices/$bdf/numa_node) path=$(readlink -f /sys/bus/pci/devices/$bdf)"
    echo "  link: $(cat /sys/bus/pci/devices/$bdf/current_link_speed) x$(cat /sys/bus/pci/devices/$bdf/current_link_width)"
done

echo "=== DSA / IAA devices: PCI path, NUMA node, WQs, engines"
for dev in /sys/bus/dsa/devices/dsa* /sys/bus/dsa/devices/iax*; do
    [[ -e "$dev" ]] || continue
    name=$(basename "$dev")
    pci=$(readlink -f "$dev/..")
    echo "$name numa=$(cat "$dev/numa_node" 2>/dev/null) state=$(cat "$dev/state" 2>/dev/null) path=$pci"
    for wq in "$dev"/wq*; do
        [[ -e "$wq" ]] || continue
        state=$(cat "$wq/state")
        [[ "$state" == enabled ]] || continue
        echo "  $(basename "$wq") mode=$(cat "$wq/mode") size=$(cat "$wq/size") group=$(cat "$wq/group_id") clients=$(cat "$wq/clients") name=$(cat "$wq/name" 2>/dev/null)"
    done
    for engine in "$dev"/engine*; do
        [[ -e "$engine" ]] || continue
        group=$(cat "$engine/group_id")
        [[ "$group" == -1 ]] && continue
        echo "  $(basename "$engine") group=$group"
    done
done

echo "=== CPU NUMA nodes"
lscpu | grep -E "^(Model name|Socket|NUMA node)"

echo "=== PCIe tree"
lspci -tv

echo "=== which DSA WQ each rank used in the vLLM runs"
for log in "$REPO_DIR"/tests/dsa_fallback_dev/results/tests_20261120_*/step4/*/fallback1/serve.log; do
    echo "--- $log"
    grep -a -m1 "KVSHRINK_DSA_DEVICES" "$log"
    grep -a -m1 "VLLM_CPU_OMP_THREADS_BIND" "$log"
    grep -a "dsa_init: mapped" "$log" | sort | uniq -c
done
