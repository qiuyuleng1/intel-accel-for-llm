"""Check of the remote_pool GET/PUT payload and SET_BUNDLES_PRIORITY plumbing in the 1026
PR branch (dev helper, not part of the PR; remote KVStore itself is not run).

Run inside the 1026 container: python3 rpc_check.py
"""

import sys

from iaxl.remote_pool import rpc
from iaxl.remote_pool.kvstore_remote import KVStoreRemote

failures = 0


def check(ok: bool, what: str) -> None:
    global failures
    print(("PASS" if ok else "FAIL") + ": " + what)
    if not ok:
        failures += 1


# Round trip with priorities and a bundle id.
payload = rpc.pack_blocks([5, 6, 7], ["h5", "h6", "h7"], [0, 2, 3], "req-1", "mamba",
                          [1, 2, 2], 42)
indices, hashes, layer_idx, desc, label, priorities, bundle_id = rpc.unpack_blocks(payload)
check(indices == [5, 6, 7] and hashes == ["h5", "h6", "h7"] and layer_idx == [0, 2, 3]
      and desc == "req-1" and label == "mamba" and priorities == [1, 2, 2] and bundle_id == 42,
      "GET payload with priorities and bundle id round-trips")

# Round trip without priorities (PUT, or GET with priorities=None).
payload = rpc.pack_blocks([1], ["h1"], [4], "", "")
indices, hashes, layer_idx, desc, label, priorities, bundle_id = rpc.unpack_blocks(payload)
check(indices == [1] and hashes == ["h1"] and layer_idx == [4] and desc == "" and label == ""
      and priorities == [] and bundle_id == 0, "payload without priorities round-trips")

check(rpc.SET_BUNDLES_PRIORITY == 14 and rpc.SORT_PRIORITY_LEVEL == 15 and rpc.UNREGISTER == 13
      and rpc.GET == 4, "SET_BUNDLES_PRIORITY / SORT_PRIORITY_LEVEL appended; old numbers unchanged")


class FakeStore:
    def __init__(self):
        self.calls = []

    def get(self, indices, hashes, names, desc, label=None, priorities=None, bundle_id=0):
        self.calls.append(("get", names, label, priorities, bundle_id))
        return {name: object() for name in names}

    def put(self, indices, hashes, names, desc, label=None):
        self.calls.append(("put", names, label))
        return {name: object() for name in names}

    def set_bundles_priority(self, bundle_ids, priority):
        self.calls.append(("raise", bundle_ids, priority))
        return 3

    def sort_priority_level(self, priority):
        self.calls.append(("sort", priority))


# Daemon handlers, without the NIXL transport.
service = rpc.KVStoreService.__new__(rpc.KVStoreService)
service.kvstore = FakeStore()
service.layer_names = ["l0", "l1", "l2", "l3"]
service.layer_idx = {"l0": 0, "l1": 1, "l2": 2, "l3": 3}
service.jobs = {}
service._next_job = 0

service.dispatch("peer", rpc.GET, rpc.pack_blocks([1, 2], ["a", "b"], [1, 3], "r", "",
                                                   [1, 2], 9))
check(service.kvstore.calls[-1] == ("get", ["l1", "l3"], None, {"l1": 1, "l3": 2}, 9),
      "daemon GET forwards per-layer priorities and bundle id")
service.dispatch("peer", rpc.GET, rpc.pack_blocks([1], ["a"], [0], "r", ""))
check(service.kvstore.calls[-1] == ("get", ["l0"], None, None, 0),
      "daemon GET without priorities forwards None / bundle id 0")
service.dispatch("peer", rpc.PUT, rpc.pack_blocks([1], ["a"], [2], "r", ""))
check(service.kvstore.calls[-1] == ("put", ["l2"], None), "daemon PUT unchanged")
try:
    service.dispatch("peer", rpc.GET, rpc.pack_blocks([1], ["a"], [0, 1], "r", "", [1], 0))
    mismatch_raised = False
except ValueError:
    mismatch_raised = True
check(mismatch_raised, "daemon rejects a priority count that differs from the layer count")
moved = rpc.unpack_json(service.dispatch("peer", rpc.SET_BUNDLES_PRIORITY,
                                         rpc.pack_json([[4, 5], 0])))
check(moved == 3 and service.kvstore.calls[-1] == ("raise", [4, 5], 0),
      "daemon SET_BUNDLES_PRIORITY calls KVStore.set_bundles_priority")
service.dispatch("peer", rpc.SORT_PRIORITY_LEVEL, rpc.pack_json([0]))
check(service.kvstore.calls[-1] == ("sort", 0),
      "daemon SORT_PRIORITY_LEVEL calls KVStore.sort_priority_level")


# Client side, without the transport: capture the payload it would send.
class FakeChannel:
    def __init__(self):
        self.sent = []

    def call(self, method, payload=b""):
        self.sent.append((method, payload))
        if method == rpc.SET_BUNDLES_PRIORITY:
            return rpc.pack_json(2)
        if method == rpc.SORT_PRIORITY_LEVEL:
            return rpc.pack_json(None)
        return (11).to_bytes(4, "little")


client = KVStoreRemote.__new__(KVStoreRemote)
client.has_only_mode = False
client.layer_names = ["l0", "l1", "l2"]
client.layer_idx = {"l0": 0, "l1": 1, "l2": 2}
client._pending = {}
client._early_done = {}
client.rpc = FakeChannel()
client.get([3], ["h"], ["l2", "l0"], "r", priorities={"l0": 1, "l1": 0, "l2": 2}, bundle_id=6)
method, payload = client.rpc.sent[-1]
_, _, layer_idx, _, _, priorities, bundle_id = rpc.unpack_blocks(payload)
check(method == rpc.GET and layer_idx == [2, 0] and priorities == [2, 1] and bundle_id == 6,
      "client GET sends one level per requested layer, in layer_idx order")
try:
    client.get([3], ["h"], ["l1"], "r", priorities={"l0": 1}, bundle_id=6)
    missing_raised = False
except ValueError:
    missing_raised = True
check(missing_raised, "client GET rejects priorities missing a requested layer")
check(client.set_bundles_priority([1, 2], 0) == 2
      and rpc.unpack_json(client.rpc.sent[-1][1]) == [[1, 2], 0],
      "client set_bundles_priority sends [bundle_ids, priority]")
client.sort_priority_level(0)
check(client.rpc.sent[-1][0] == rpc.SORT_PRIORITY_LEVEL
      and rpc.unpack_json(client.rpc.sent[-1][1]) == [0],
      "client sort_priority_level sends [priority]")

print("ALL PASS" if failures == 0 else f"FAILED ({failures})")
sys.exit(0 if failures == 0 else 1)
