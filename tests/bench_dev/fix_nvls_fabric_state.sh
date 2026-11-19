#!/bin/bash
# Re-sync b300's NVLink fabric state after nvidia-fabricmanager was restarted without a GPU reset
# (11-14 06:43, needrestart after an unattended libc upgrade). Since then NCCL's NVLS fails:
# dmesg "NVLink fabric state cached by the driver is out of sync", fabric manager "status 135".
#
# Steps: stop nvidia-dcgm, nvidia-fabricmanager (also stops the NVLink Subnet Manager) and
# nvidia-persistenced, reset all 8 GPUs with nvidia-smi -r, start the three services again, then
# show the fabric state. It aborts before stopping anything if a process uses a GPU. If the reset
# fails it still starts the services again and exits 1 (the next step, a driver reload, is a
# separate decision).
#
# Runs on the b300 host (needs sudo): bash tests/bench_dev/fix_nvls_fabric_state.sh
set -uo pipefail

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

start_services() {
    local service
    for service in nvidia-persistenced nvidia-fabricmanager nvidia-dcgm; do
        sudo systemctl start "$service"
        log "start $service: $(systemctl is-active "$service")"
    done
}

log "step 0: check that no process uses a GPU"
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
if [[ -n "$apps" ]]; then
    log "ABORT: GPU compute processes: $apps"
    exit 1
fi
holders=$(sudo lsof /dev/nvidia* 2>/dev/null | awk 'NR > 1 {print $1}' | sort -u | tr '\n' ' ')
log "processes holding /dev/nvidia*: ${holders:-none}"
if [[ -n "$holders" && "$holders" != "nv-hosten " ]]; then
    log "ABORT: unexpected processes hold /dev/nvidia*"
    exit 1
fi

for service in nvidia-dcgm nvidia-fabricmanager nvidia-persistenced; do
    log "stop $service"
    sudo systemctl stop "$service"
    log "$service: $(systemctl is-active "$service")"
done
holders=$(sudo lsof /dev/nvidia* 2>/dev/null | awk 'NR > 1 {print $1}' | sort -u | tr '\n' ' ')
log "processes holding /dev/nvidia* after stopping: ${holders:-none}"

log "step 4: nvidia-smi -r (all GPUs)"
reset_start=$(date '+%Y-%m-%d %H:%M:%S')
sudo nvidia-smi -r
reset_status=$?
log "nvidia-smi -r exit $reset_status"

log "step 5: start the services again"
start_services
fm_start=$(systemctl show nvidia-fabricmanager -p ActiveEnterTimestamp --value)
log "fabric manager started at $fm_start"
sudo journalctl -u nvidia-fabricmanager --no-pager --since "$reset_start" | grep -a "nv-fabricmanager\[" \
    | tail -3

log "fabric state per GPU:"
nvidia-smi -q | grep -A2 "^    Fabric" | grep -E "State|Status" | paste - - | sort | uniq -c
log "kernel messages since the reset:"
sudo journalctl -k --no-pager --since "$reset_start" | grep -a -i "NVRM\|xid" | tail -10

if (( reset_status != 0 )); then
    log "RESET FAILED: services are running again, fabric state is still the old one"
    exit 1
fi
log "DONE"
