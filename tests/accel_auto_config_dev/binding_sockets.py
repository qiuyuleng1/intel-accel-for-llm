"""Prints the socket of each rank's CPUs (first CPU of the entry) and DSA work queue (first WQ of the entry).
Usage: binding_sockets.py <VLLM_CPU_OMP_THREADS_BIND> <KVSHRINK_DSA_DEVICES>
Output: "cpu <socket of rank 0> <socket of rank 1> ..." and "dsa ..." lines."""

import sys


def socket_of_cpu(cpu: str) -> int:
    with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id") as package_file:
        return int(package_file.read())


def socket_of_numa_node(numa_node: int) -> int:
    with open(f"/sys/devices/system/node/node{numa_node}/cpulist") as cpulist_file:
        return socket_of_cpu(cpulist_file.read().split(",")[0].split("-")[0].strip())


def main() -> int:
    cpu_sockets = []
    for entry in sys.argv[1].split("|"):
        cpu_sockets.append(str(socket_of_cpu(entry.split(",")[0].split("-")[0].strip())))
    dsa_sockets = []
    for entry in sys.argv[2].split("|"):
        work_queue = entry.split(",")[0].strip()
        device = "dsa" + work_queue[len("wq"):].split(".")[0]
        with open(f"/sys/bus/dsa/devices/{device}/numa_node") as numa_file:
            dsa_sockets.append(str(socket_of_numa_node(int(numa_file.read()))))
    print("cpu " + " ".join(cpu_sockets))
    print("dsa " + " ".join(dsa_sockets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
