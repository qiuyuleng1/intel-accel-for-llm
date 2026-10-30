"""KVShrinkConnector of the 1026 PR branch plus one log line per raise (dev helper, not
part of the PR). Load it with kv_connector_module_path=traced_kvshrink_connector and
this directory on PYTHONPATH.

Line format (worker ranks):
  raise_trace rank=<r> raised_reqs=<n> sync_batch=<0|1> bundles=<n> moved=<tasks>
    sync_layers=<n> sync_done=<n> sync_done_prefix=<n>
raised_reqs: promoted requests whose remaining layers this forward raised to P0;
sync_batch: 1 if this step also had a sync batch (its bundle is raised with them);
moved: tasks set_bundles_priority() moved, i.e. remaining-layer and sync-batch load tasks
that were still waiting in the queue;
sync_layers / sync_done / sync_done_prefix (known issue K4): layers of the sync batch, how
many of them had their decompression task finished right before set_bundles_priority(),
and how many consecutive ones from layer 0 had. Read with Context.unzip_is_complete(),
which changes nothing; a task still running counts as not finished.
"""

import logging

from kvshrink.kvshrink_connector import KVShrinkConnector as PRKVShrinkConnector

logger = logging.getLogger("kvshrink.kvshrink_connector")


class KVShrinkConnector(PRKVShrinkConnector):
    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        store = self.kvstore
        set_bundles_priority = store.set_bundles_priority
        self.trace_raises = []

        def traced_set_bundles_priority(bundle_ids, priority):
            sync_layers, sync_done, sync_done_prefix = self.sync_batch_progress()
            moved = set_bundles_priority(bundle_ids, priority)
            self.trace_raises.append(
                (len(bundle_ids), moved, sync_layers, sync_done, sync_done_prefix)
            )
            return moved

        store.set_bundles_priority = traced_set_bundles_priority

    def sync_batch_progress(self):
        if not self._current_get_tasks:
            return 0, 0, 0
        done = 0
        prefix = 0
        prefix_open = True
        for layer_name in self._layer_names:
            task = self._current_get_tasks.get(layer_name)
            if task is None:
                continue
            finished = task.ctx is None or task.ctx.unzip_is_complete()
            if finished:
                done += 1
            if finished and prefix_open:
                prefix += 1
            else:
                prefix_open = False
        return len(self._current_get_tasks), done, prefix

    def start_load_kv(self, forward_context, **kwargs):
        active_before = len(self._active_promoted_tasks)
        self.trace_raises = []
        super().start_load_kv(forward_context, **kwargs)
        for bundles, moved, sync_layers, sync_done, sync_done_prefix in self.trace_raises:
            raised_reqs = len(self._active_promoted_tasks) - active_before
            sync_batch = 1 if self._current_get_tasks else 0
            logger.info(
                "raise_trace rank=%d raised_reqs=%d sync_batch=%d bundles=%d moved=%d "
                "sync_layers=%d sync_done=%d sync_done_prefix=%d",
                self.tp_rank, raised_reqs, sync_batch, bundles, moved,
                sync_layers, sync_done, sync_done_prefix,
            )
