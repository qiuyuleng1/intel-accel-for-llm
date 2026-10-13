#!/bin/bash -e
# usage: dsa-numa1.sh enable|disable   (dev helper, run as root, e.g. in the
# privileged container)
# GPU4-7 are on NUMA1. Enable one dedicated user WQ on each NUMA1 DSA
# (dsa8/10/12/14), or put them back to the disabled state they were found in.
# Same settings as tools/setup_dsa.sh, but only for these four devices so the
# NUMA0 DSAs (possibly used by others) are not touched.
DEVS="dsa8 dsa10 dsa12 dsa14"
ACTION=$1

for DEV in $DEVS; do
    ID=${DEV#dsa}
    SYS=/sys/bus/dsa/devices/$DEV
    accel-config disable-wq "$DEV/wq$ID.0" 2>/dev/null || true
    accel-config disable-device "$DEV" 2>/dev/null || true
    if [ "$ACTION" = "disable" ]; then
        continue
    fi
    for ENGINE in "$SYS"/engine$ID.*; do
        accel-config config-engine "$DEV/$(basename "$ENGINE")" --group-id=0
    done
    accel-config config-wq "$DEV/wq$ID.0" \
        --group-id=0 \
        --wq-size="$(cat "$SYS/max_work_queues_size")" \
        --mode=dedicated \
        --type=user \
        --name=dsa_gpu \
        --priority=10 \
        --max-transfer-size="$(cat "$SYS/max_transfer_size")" \
        --max-batch-size="$(cat "$SYS/max_batch_size")" \
        --driver-name=user
    accel-config enable-device "$DEV"
    accel-config enable-wq "$DEV/wq$ID.0"
done

for DEV in $DEVS; do
    ID=${DEV#dsa}
    echo "$DEV state=$(cat /sys/bus/dsa/devices/$DEV/state) wq$ID.0=$(cat /sys/bus/dsa/devices/wq$ID.0/state) type=$(cat /sys/bus/dsa/devices/wq$ID.0/type)"
done
