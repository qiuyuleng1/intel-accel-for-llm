#!/bin/bash
#
# check-acs.sh
#
# Fails when PCIe ACS is on at the switch port above any NVIDIA GPU. With ACS
# on, traffic between two GPUs under the same PCIe switch detours through the
# CPU: on h8263-6000d an 8-GPU 128 MiB NCCL all-reduce took 27.5 ms instead of
# 9.9 ms, and a 32k-token Qwen2.5-72B TP=8 prefill 21.6 s instead of 10.4 s.
# ACS is switched back on at every reboot; turn it off as described in
# PERFORMANCE_MODE_SETUP.md (tools/disable-acs.sh --apply).
#
# Read-only. Run on the HOST (reads PCI config space through sudo -n):
#   bash tools/check-acs.sh
set -uo pipefail

acs_on=0
for gpu_bus_id in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
    # nvidia-smi prints 00000000:38:00.0, sysfs names it 0000:38:00.0.
    gpu_bdf=$(echo "${gpu_bus_id: -12}" | tr 'A-F' 'a-f')
    port_bdf=$(basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/$gpu_bdf")")")
    if ! port_info=$(sudo -n lspci -vvv -s "$port_bdf" 2>&1); then
        echo "ERROR: cannot read PCI config of $port_bdf: $port_info" >&2
        exit 2
    fi
    acs_control=$(echo "$port_info" | grep -oE "ACSCtl:.*" | tr -s ' ')
    if [[ -z "$acs_control" ]]; then
        echo "GPU $gpu_bdf  port $port_bdf  no ACS capability"
        continue
    fi
    echo "GPU $gpu_bdf  port $port_bdf  $acs_control"
    # Every ACS control bit must be off ('-'), as tools/disable-acs.sh leaves it.
    if [[ "$acs_control" == *+* ]]; then
        acs_on=1
    fi
done

if [[ "$acs_on" == 1 ]]; then
    echo "ERROR: ACS is on above some GPU, so GPU-to-GPU traffic detours through the CPU." >&2
    echo "Turn it off first (see PERFORMANCE_MODE_SETUP.md): sudo bash tools/disable-acs.sh --apply" >&2
    exit 1
fi
echo "ACS is off on every GPU switch port."
