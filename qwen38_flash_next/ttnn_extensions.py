# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The names the model needs on top of upstream ttnn, defined here instead of in tt-metal.

Each entry fills a name only when ttnn lacks it, so a ttnn that grows the name keeps its own.  Importing this module
applies them; ``qwen38_flash_next.ttnn`` imports it first.  Host fakes that stand in for ttnn in the no-device tests
are left alone.
"""

from __future__ import annotations

import functools
import os
import tempfile
import types

import ttnn

_PROC_SELF_FD = "/proc/self/fd/"


def _is_real_ttnn() -> bool:
    return hasattr(ttnn, "_ttnn")


if _is_real_ttnn():
    from ttnn.tools.trace_allocation_tracker import (
        TRACE_ALLOC_TRACKING,
        TraceAllocationTracker,
        acknowledge_corruptible,
        corruptible_allocation_scope,
    )
else:  # a host fake
    TRACE_ALLOC_TRACKING = False
    TraceAllocationTracker = None
    acknowledge_corruptible = None
    corruptible_allocation_scope = None


class UnsafeAllocationTracker:
    """The per-mesh form of upstream's ``TraceAllocationTracker``, whose methods take the mesh per call."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

    def verify_before_replay(self, trace_id) -> None:
        TraceAllocationTracker.verify_before_replay(self.mesh_device, trace_id)


def mark_corruptible(tensor) -> None:
    """Take ``tensor``'s buffer out of the trace allocation accounting (a no-op while tracking is off)."""

    acknowledge_corruptible(tensor)


# moe_compute's ``prefill_rings`` (R replica rings that keep the expert weights resident) and its packed token lists
# (one call per slab) are not in upstream's op.  Without them the op runs its streaming ring and a slab runs as
# 128-row calls: the same pages, slower.
def moe_compute_has_prefill_rings() -> bool:
    return "prefill_rings" in (getattr(ttnn.experimental.moe_compute, "__doc__", None) or "")


def _moe_compute_without_prefill_rings(op):
    @functools.wraps(op)
    def moe_compute(*args, prefill_rings=None, **kwargs):
        return op(*args, **kwargs)

    return moe_compute


def _load_tensor_with_descriptors(load_tensor):
    """``ttnn.load_tensor`` that also reads a ``/proc/self/fd/N`` path: the BF4 cache passes the descriptor it holds
    open on a verified artifact, so the bytes read are the bytes it verified."""

    @functools.wraps(load_tensor)
    def load(file_name, *, device=None):
        name = str(file_name)
        if not name.startswith(_PROC_SELF_FD):
            return load_tensor(file_name, device=device)
        # The native reader wants a .tensorbin name: a link to the descriptor opens the file it holds, not whatever
        # sits at the artifact's path now.
        with tempfile.TemporaryDirectory() as directory:
            link = os.path.join(directory, "descriptor.tensorbin")
            os.symlink(name, link)
            return ttnn._ttnn.tensor.load_tensor_flatbuffer(link, device)

    return load


def get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate(mesh_device, noc, coordinate):
    """The DRAM-bank reader cores of the device at ``coordinate``.  Upstream binds only the mesh-wide assignment, and
    its DRAM-sharded programs place every device's readers by it, so that is the answer at each coordinate."""

    return ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, noc)


def _chunk_gdn_prims():
    transformer = ttnn._ttnn.operations.transformer
    horner = ttnn._ttnn.operations.transformer.ChunkGdnWyInverse.HORNER

    # Horner is the WY inverse the model's bitwise contracts were taken with; upstream's AUTO picks the
    # forward-substitution solve on Blackhole at chunk 32.
    def chunk_gdn_prep(*args, wy_inverse=horner, **kwargs):
        return transformer.chunk_gdn_prep(*args, wy_inverse=wy_inverse, **kwargs)

    def chunk_gdn_scan(v_beta, nkd, q_decay, intra, k_dec_t, dl, t_inv, initial_state, **kwargs):
        return transformer.chunk_gdn_scan(
            v_beta, nkd, q_decay, intra, k_dec_t, dl, t_inv, initial_state=initial_state, **kwargs
        )

    return chunk_gdn_prep, chunk_gdn_scan


def _chunk_gated_delta_rule_horner(op):
    horner = ttnn._ttnn.operations.transformer.ChunkGdnWyInverse.HORNER

    @functools.wraps(op)
    def chunk_gated_delta_rule(*args, wy_inverse=horner, **kwargs):
        return op(*args, wy_inverse=wy_inverse, **kwargs)

    return chunk_gated_delta_rule


def _register_prefill_adapter() -> None:
    try:
        from models.demos.common.prefill import adapter
    except ImportError:  # tt-metal's models are not on the path: no disaggregated prefill runner to serve
        return
    adapter.ADAPTER_PATHS.setdefault(
        "qwen38_flash_next", "qwen38_flash_next.prefill.adapter:Qwen38FlashNextPrefillAdapter"
    )


def apply() -> None:
    if not _is_real_ttnn() or getattr(ttnn, "_qwen38_extensions", False):
        return
    ttnn._qwen38_extensions = True
    if not hasattr(ttnn, "mark_corruptible"):
        ttnn.mark_corruptible = mark_corruptible
    if not hasattr(ttnn, "corruptible_allocation_scope"):
        ttnn.corruptible_allocation_scope = corruptible_allocation_scope
    if not hasattr(ttnn, "TRACE_ALLOC_TRACKING"):
        ttnn.TRACE_ALLOC_TRACKING = TRACE_ALLOC_TRACKING
    if not moe_compute_has_prefill_rings():
        ttnn.experimental.moe_compute = _moe_compute_without_prefill_rings(ttnn.experimental.moe_compute)
    ttnn.load_tensor = _load_tensor_with_descriptors(ttnn.load_tensor)
    if not hasattr(ttnn.device, "get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate"):
        ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate = (
            get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate
        )
    prep, scan = _chunk_gdn_prims()
    if not hasattr(ttnn, "prim"):
        ttnn.prim = types.SimpleNamespace()
    ttnn.prim.chunk_gdn_prep = prep
    ttnn.prim.chunk_gdn_scan = scan
    ttnn.transformer.chunk_gated_delta_rule = _chunk_gated_delta_rule_horner(ttnn.transformer.chunk_gated_delta_rule)
    _register_prefill_adapter()


apply()
