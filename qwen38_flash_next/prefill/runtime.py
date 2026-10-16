# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The prefill runtime the engine drives, and the slot cache it is handed.

The engine allocates one cache for ``num_users`` slots, owns its lifetime, and passes it into
every call.  This model keeps its prefill state per session rather than per slot --
``ttnn/model.py`` ``forward_prefill_chunk_generic(chunk_state, state)`` takes a generic state and
no slot index -- so ``Qwen38SlotCaches`` holds one pre-allocated state per slot and the runtime
indexes it.  The model itself is unchanged.

Three caches migrate, each numbered in its own space because the layer types interleave: the GDN
recurrent state over the 36 linear-attention layers, and the QSA and indexer KV over the 12 full-
attention layers.  See ``kv_chunk_table`` for why the GDN stage is one bucket.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import ttnn
from models.demos.blackhole.qwen38_flash_next.prefill import kv_chunk_table as kct
from models.demos.blackhole.qwen38_flash_next.prefill.adapter import GDN_LAYERS, QSA_LAYERS
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import KvCacheStage

# A layer is full attention when its global index is 3 mod 4 (ttnn/layer.py); the other 36 are GDN.
QSA_LAYER_STRIDE = 4
QSA_LAYER_RESIDUE = 3


@dataclass
class Qwen38RuntimeConfig:
    """What the engine reads off the runtime to drive the chunk schedule."""

    chunk_size: int
    max_seq_len: int
    first_layer_idx: int
    is_first_rank: bool
    is_last_rank: bool
    num_users: int
    mesh_shape: tuple


class Qwen38SlotCaches(KvCaches):
    """One generic state per user slot, each owning its own per-layer cache tensors.

    The engine treats this as opaque: it allocates it once through the adapter, passes it into
    every runtime call and frees it with the mesh.  ``states[slot]`` is the model's own session
    state; ``chunk_states[slot]`` is that state's 32-row chunk form.
    """

    def __init__(self, *, model, states, chunk_states, num_users: int):
        if len(states) != num_users or len(chunk_states) != num_users:
            raise ValueError(f"expected {num_users} slot states, got {len(states)} and {len(chunk_states)}")
        self.model = model
        self.states = list(states)
        self.chunk_states = list(chunk_states)
        self.num_users = num_users

    def _check(self, slot_id: int) -> int:
        if not isinstance(slot_id, int) or isinstance(slot_id, bool) or not 0 <= slot_id < self.num_users:
            raise ValueError(f"slot {slot_id!r} is outside [0,{self.num_users})")
        return slot_id

    def state_for(self, slot_id: int):
        return self.states[self._check(slot_id)]

    def chunk_state_for(self, slot_id: int):
        return self.chunk_states[self._check(slot_id)]

    def release(self) -> None:
        """Release in the order the chain uses: chunk states, then generic states."""

        for chunk_state in self.chunk_states:
            self.model.release_chunk_state(chunk_state)
        for state in self.states:
            self.model.release_generic_state(state)
        self.chunk_states.clear()
        self.states.clear()


def _checkpoint_root() -> str:
    return os.environ.get("PREFILL_HF_MODEL") or os.environ.get("MODEL_WEIGHTS_DIR") or ""


def _cache_root() -> str:
    return os.environ.get("PREFILL_TTNN_CACHE") or os.environ.get("QWEN38_CACHE_ROOT") or ""


def allocate_slot_caches(*, mesh_device, hf_config, params):
    """Allocate one generic state and its chunk state per user slot.

    The engine allocates the cache before it builds the runtime, so the model is built here and
    handed to the runtime through the cache object.  Nothing in the model forbids several live
    generic states -- validation checks a per-model owner token, not a per-state one -- but no
    other caller allocates more than one, so this is the first use of that.
    """

    built = build_target_model(mesh_device=mesh_device, params=params)
    model = built.model
    states, chunk_states = [], []
    for _ in range(params.num_users):
        state = model.allocate_generic_state()
        model.reset_generic_state_inplace(state)
        chunk_state = model.allocate_chunk_state(state)
        model.reset_chunk_state_inplace(chunk_state)
        states.append(state)
        chunk_states.append(chunk_state)
    caches = Qwen38SlotCaches(model=model, states=states, chunk_states=chunk_states, num_users=params.num_users)
    caches.built = built
    return caches


def build_target_model(*, mesh_device, params):
    """Run this model's own CPU preparation and builder against an already-open mesh.

    This is the chat server's construction path (``runtime_admission.prepare_cpu`` then
    ``construct_live_decode_diagnostic`` then ``build_target``), driven from the engine's
    ``PrefillRunParams`` and the environment instead of the server's argparse namespace.
    """

    import argparse

    import ttnn
    from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles, runtime_admission
    from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import construct_live_decode_diagnostic
    from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import resolve_route

    cache_root = Path(_cache_root())
    label = os.environ.get("PREFILL_CACHE_LABEL", "c32768-p150-line")
    args = argparse.Namespace(
        checkpoint=Path(_checkpoint_root()),
        component_cache_root=cache_root / "caches" / label / "components",
        routed_bf4_scratch_root=cache_root / "caches" / "bf4-experts",
        model_io_cache_root=cache_root / "caches" / label / "model-io",
        allocated_context=params.max_seq_len,
        bf4_producer_identity=None,
        source_head=None,
    )
    # No declared identity: runtime_admission derives it from the checkout, which is what a
    # serving run wants. Declaring one would also have to match the checkout head.

    profile = hardware_profiles.resolve_hardware_profile(os.environ.get("PREFILL_HARDWARE_PROFILE", "p150-line"))
    profile, _derivation = resolve_route(profile)

    def marker(_name: str) -> None:
        return None

    prepared, _oracle = runtime_admission.prepare_cpu(args, marker=marker, hardware_profile=profile)
    construction = construct_live_decode_diagnostic(
        prepared,
        mesh_device=mesh_device,
        collective_topology=ttnn.Topology.Linear,
        marker=marker,
    )
    return construction.builder.build_target()


def _tensor_bytes(tensor) -> int:
    return int(tensor.volume() * tensor.element_size())


class Qwen38PrefillRuntime:
    """Runs one prefill chunk into a slot and describes the caches for migration."""

    def __init__(self, *, mesh_device, hf_config, params):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = Qwen38RuntimeConfig(
            chunk_size=params.chunk_size,
            max_seq_len=params.max_seq_len,
            first_layer_idx=params.first_layer_idx,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            num_users=params.num_users,
            mesh_shape=tuple(params.mesh_shape),
        )
        if params.first_layer_idx != 0 or not (params.is_first_rank and params.is_last_rank):
            raise ValueError(
                "this model builds a fixed 48-layer target, so only a single rank is supported; "
                f"got first_layer_idx={params.first_layer_idx} first={params.is_first_rank} "
                f"last={params.is_last_rank}"
            )
        self._layer_ack = None
        self._built = None

    # ------------------------------------------------------------------
    # Migration description
    # ------------------------------------------------------------------
    def kv_migration_stages(self, kv_cache, first_layer_idx=None, num_my_layers=None) -> Sequence[KvCacheStage]:
        """One stage per cache, each in its own layer numbering rather than the global 0..47."""

        layers = kv_cache.states[0].layers
        return (
            KvCacheStage(int(layers[0].attention.recurrent.buffer_address()), 0, GDN_LAYERS),
            KvCacheStage(int(layers[QSA_LAYER_RESIDUE].attention.packed_kv_cache.buffer_address()), 0, QSA_LAYERS),
            KvCacheStage(
                int(layers[QSA_LAYER_RESIDUE].attention.compressed_index_cache.buffer_address()), 0, QSA_LAYERS
            ),
        )

    def _stage_addresses(self, kv_cache):
        """``{stage: {(cache_layer, slot): address}}``.  QSA lives on layers 3, 7, ... 47 and GDN
        on the rest, but a stage numbers its own layers from 0."""

        out = {kct.GDN_STATE: {}, kct.QSA_KV: {}, kct.INDEX_KV: {}}
        for slot, state in enumerate(kv_cache.states):
            gdn_layer = qsa_layer = 0
            for layer_index, layer in enumerate(state.layers):
                attention = layer.attention
                if layer_index % QSA_LAYER_STRIDE == QSA_LAYER_RESIDUE:
                    out[kct.QSA_KV][(qsa_layer, slot)] = int(attention.packed_kv_cache.buffer_address())
                    out[kct.INDEX_KV][(qsa_layer, slot)] = int(attention.compressed_index_cache.buffer_address())
                    qsa_layer += 1
                else:
                    out[kct.GDN_STATE][(gdn_layer, slot)] = int(attention.recurrent.buffer_address())
                    gdn_layer += 1
            if gdn_layer != GDN_LAYERS or qsa_layer != QSA_LAYERS:
                raise RuntimeError(f"slot {slot} has {gdn_layer} GDN and {qsa_layer} QSA layers")
        return out

    def build_kv_chunk_table(self, kv_cache, path: str) -> str:
        """Build and serialize the three-stage table.  Issues no comms; the engine publishes it."""

        addresses = self._stage_addresses(kv_cache)
        first = kv_cache.states[0].layers
        gdn_bytes = _tensor_bytes(first[0].attention.recurrent)
        qsa_row = first[QSA_LAYER_RESIDUE].attention.packed_kv_cache
        index_row = first[QSA_LAYER_RESIDUE].attention.compressed_index_cache
        chunk = self.config.chunk_size
        qsa_bytes = _tensor_bytes(qsa_row) * chunk // int(qsa_row.shape[-2])
        index_bytes = _tensor_bytes(index_row) * chunk // int(index_row.shape[-2])

        configs = kct.stage_configs(
            max_seq_len=self.config.max_seq_len,
            num_users=self.config.num_users,
            gdn_layers=GDN_LAYERS,
            qsa_layers=QSA_LAYERS,
            gdn_state_bytes=gdn_bytes,
            qsa_chunk_tokens=chunk,
            qsa_chunk_bytes=qsa_bytes,
            index_chunk_bytes=index_bytes,
        )
        table = kct.build_table(
            configs=configs,
            device_groups=[self._fabric_nodes()],
            host_name=socket.gethostname(),
            addresses=addresses,
            entry_bytes={kct.GDN_STATE: gdn_bytes, kct.QSA_KV: qsa_bytes, kct.INDEX_KV: index_bytes},
            qsa_chunk_tokens=chunk,
            max_seq_len=self.config.max_seq_len,
        )
        return kct.serialize(table, path)

    def kv_migration_base_address(self, kv_cache) -> int:
        return int(kv_cache.states[0].layers[0].attention.recurrent.buffer_address())

    def set_layer_ack_channel(self, channel) -> None:
        self._layer_ack = channel

    def _fabric_nodes(self):
        rows, cols = self.config.mesh_shape
        return [
            self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col))
            for row in range(rows)
            for col in range(cols)
        ]

    # ------------------------------------------------------------------
    # Chunk driving
    # ------------------------------------------------------------------
    def compile(self, kv_cache) -> None:
        """Run one throwaway chunk per slot so the first served chunk pays no first-run cost.

        A captured trace bakes the addresses of the state it was captured against, so with one
        state per slot the chunks run eagerly and there is nothing to capture here.
        """

        self._caches = kv_cache
        rows = kv_cache.model.CHUNK_ROWS if hasattr(kv_cache.model, "CHUNK_ROWS") else 32
        warm = [0] * rows
        for slot in range(kv_cache.num_users):
            state = kv_cache.state_for(slot)
            chunk_state = kv_cache.chunk_state_for(slot)
            kv_cache.model.write_chunk_inputs(chunk_state, warm, start_position=0)
            kv_cache.model.forward_prefill_chunk_generic(chunk_state, state)
            kv_cache.model.reset_generic_state_inplace(state)
            kv_cache.model.reset_chunk_state_inplace(chunk_state)

    def make_chunk_input(self, token_ids) -> list:
        """This rank is first and last, so a chunk's input is its token ids.

        The device tensor belongs to the slot's own chunk state and is written in
        ``prefill_chunk``, which is the only place the slot is known.
        """

        return list(token_ids)

    def prefill_chunk(self, input_tensor, kv_cache, *, slot_id, actual_start, actual_end, request_id=0):
        """Prefill one chunk into ``slot_id``'s state.  Returns None: this rank is the last one."""

        state = kv_cache.state_for(slot_id)
        chunk_state = kv_cache.chunk_state_for(slot_id)
        model = kv_cache.model
        accepted = int(actual_end) - int(actual_start)
        if accepted <= 0 or accepted > chunk_state.rows:
            raise ValueError(f"chunk holds {accepted} real rows, outside (0,{chunk_state.rows}]")

        model.write_chunk_inputs(chunk_state, list(input_tensor), start_position=int(actual_start))
        if accepted != chunk_state.rows:
            # The tail chunk is padded; only its real rows may be committed.
            model.write_chunk_accepted(chunk_state, accepted)
        model.forward_prefill_chunk_generic(chunk_state, state)
        if self._layer_ack is not None:
            self._layer_ack.bump(request_id)
        return None
