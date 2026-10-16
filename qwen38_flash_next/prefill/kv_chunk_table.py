# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The migration address table for this model's three caches.

The disaggregated prefill engine publishes one ``KvChunkAddressTable`` describing where every
cache chunk lives, and the decode side reads those addresses over UMD.  Two things about this
model shape the table.

The caches cover disjoint, interleaved layer subsets: of the 48 layers, 36 are linear attention
and 12 are full attention, in a repeating pattern of three then one.  ``KvCacheStage`` carries a
contiguous ``(first_layer, count)``, so each cache is numbered in its own 0..N-1 space rather
than the global one, which is the case ``kv_migration_stages`` exists for.

The GDN state is recurrent, not positional: a fixed ``[heads, dim, dim]`` fp32 per user per
layer that does not grow with the sequence.  Its config therefore sets ``chunk_n_tokens`` to the
whole ``max_sequence_length``, so the sequence is one bucket and the state is one entry.
``KvChunkAddressTable::lookup`` asserts ``position % chunk_n_tokens == 0``, so that stage is
addressable only at position 0; consumers already step their position loop by the chunk size, so
they read it once per (layer, slot).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import ttnn

GDN_STATE = "gdn_state"
QSA_KV = "qsa_kv"
INDEX_KV = "index_kv"
STAGE_NAMES = (GDN_STATE, QSA_KV, INDEX_KV)


def _disagg():
    return ttnn.experimental.disaggregation


def make_config(*, num_layers: int, max_seq_len: int, num_users: int, chunk_n_tokens: int, chunk_size_bytes: int):
    cfg = _disagg().KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = chunk_n_tokens
    cfg.chunk_size_bytes = chunk_size_bytes
    return cfg


def stage_configs(
    *,
    max_seq_len: int,
    num_users: int,
    gdn_layers: int,
    qsa_layers: int,
    gdn_state_bytes: int,
    qsa_chunk_tokens: int,
    qsa_chunk_bytes: int,
    index_chunk_bytes: int,
) -> dict:
    """The three named configs, in ``kv_migration_stages`` order."""

    return {
        # One bucket over the whole sequence: the recurrent state has no position.
        GDN_STATE: make_config(
            num_layers=gdn_layers,
            max_seq_len=max_seq_len,
            num_users=num_users,
            chunk_n_tokens=max_seq_len,
            chunk_size_bytes=gdn_state_bytes,
        ),
        QSA_KV: make_config(
            num_layers=qsa_layers,
            max_seq_len=max_seq_len,
            num_users=num_users,
            chunk_n_tokens=qsa_chunk_tokens,
            chunk_size_bytes=qsa_chunk_bytes,
        ),
        INDEX_KV: make_config(
            num_layers=qsa_layers,
            max_seq_len=max_seq_len,
            num_users=num_users,
            chunk_n_tokens=qsa_chunk_tokens,
            chunk_size_bytes=index_chunk_bytes,
        ),
    }


def _location(noc_addr: int, size_bytes: int, group_index):
    loc = _disagg().KvCacheLocation()
    loc.noc_addr = int(noc_addr)
    loc.size_bytes = int(size_bytes)
    loc.device_group_index = group_index
    return loc


def build_table(
    *,
    configs: Mapping[str, object],
    device_groups: Sequence[Iterable],
    host_name: str,
    addresses: Mapping[str, Mapping[tuple, int]],
    entry_bytes: Mapping[str, int],
    qsa_chunk_tokens: int,
    max_seq_len: int,
):
    """Populate the table from each cache tensor's own address.

    Each slot owns a separately allocated set of per-layer tensors rather than a slice of one
    shared buffer, so an entry's address cannot be derived from a base and a stride.
    ``addresses[stage][(cache_layer, slot)]`` is that tensor's ``buffer_address()``, where
    ``cache_layer`` counts within the stage (0..35 for the GDN state, 0..11 for the others)
    rather than the global 0..47 the model modules carry.

    ``device_groups`` is one iterable of ``ttnn.FabricNodeId`` per replica group.
    """

    table = _disagg().KvChunkAddressTable(dict(configs))
    groups = []
    for nodes in device_groups:
        nodes = list(nodes)
        groups.append(table.add_device_group(nodes))
        for node in nodes:
            table.set_fabric_node_host(node, host_name=host_name)
    if not groups:
        raise ValueError("the migration table needs at least one device group")
    group = groups[0]

    size = entry_bytes[GDN_STATE]
    for (layer, slot), addr in addresses[GDN_STATE].items():
        table.set(layer, 0, slot, _location(addr, size, group), config=GDN_STATE)

    for name in (QSA_KV, INDEX_KV):
        size = entry_bytes[name]
        for (layer, slot), addr in addresses[name].items():
            for chunk, position in enumerate(range(0, max_seq_len, qsa_chunk_tokens)):
                table.set(layer, position, slot, _location(addr + chunk * size, size, group), config=name)
    return table


def serialize(table, path: str) -> str:
    _disagg().export_to_protobuf_file(table, path)
    return path
