# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``Qwen38FlashNextPrefillAdapter`` — this model's plug-in for the disaggregated prefill engine.

The engine (``models/demos/common/prefill/runners/prefill_runner.py``) owns rank topology, the
sockets, the serving loop and the migration publish, and drives a model through this adapter.
The adapter is a factory and a descriptor; every operational behaviour lives on the runtime that
``build_runtime`` returns.

Two things here are specific to this model.

``layer_split_boundaries`` admits only 0, so the runner accepts a single rank and rejects any
pipeline split.  The builder constructs a fixed 48-layer target (``ttnn/builder.py``
``build_target``) and ``config.py`` asserts ``num_hidden_layers == 48``, so a rank holding a
layer slice cannot be built.  Returning ``{0}`` makes that a clean rejection rather than a model
built with the wrong layers.

``load_hf_config`` returns a mutable carrier rather than ``Qwen38Config`` itself, which is a
frozen dataclass: the runner assigns ``max_seq_len`` onto whatever this returns.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams

# Of the 48 layers, three of every four are linear attention and the fourth is full attention.
# Each cache is numbered in its own space, so these are the per-cache layer counts.
GDN_LAYERS = 36
QSA_LAYERS = 12

# The common runner reads PREFILL_NUM_LAYERS with DeepSeek's 61 as its default, and it reads it
# after importing this adapter, so a setdefault here gives this model its own count. An explicit
# value still wins.
os.environ.setdefault("PREFILL_NUM_LAYERS", "48")


class Qwen38FlashNextConfig:
    """Static dimensions the common runner reads.  It uses only ``FABRIC_PAYLOAD_SIZE``; the rest
    document the model for readers and feed the cache geometry."""

    EMB_SIZE = 2560  # hidden_size
    # The runner builds the fabric router config from this. Mirroring the embedding dim, as the
    # larger models do, puts the packet below this model's 4096-byte fp32 tile pages, and the
    # runtime recommends 8192 for its 2048-byte bf16 pages.
    FABRIC_PAYLOAD_SIZE = 8192
    NUM_LAYERS = 48
    GDN_LAYERS = GDN_LAYERS
    QSA_LAYERS = QSA_LAYERS
    GDN_VALUE_HEADS = 48
    GDN_VALUE_HEAD_DIM = 128
    QSA_KV_HEADS = 2
    QSA_HEAD_DIM = 256
    INDEX_KV_HEADS = 1
    INDEX_HEAD_DIM = 128
    VOCAB_SIZE = 248320


class HFConfigCarrier:
    """A mutable view of the frozen ``Qwen38Config`` for the runner, which assigns ``max_seq_len``."""

    def __init__(self, config):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_position_embeddings
        self.layer_types = config.layer_types


def _checkpoint_root() -> str:
    return (
        os.environ.get("PREFILL_HF_MODEL")
        or os.environ.get("MODEL_WEIGHTS_DIR")
        or Qwen38FlashNextPrefillAdapter.hf_model_default
    )


class Qwen38FlashNextPrefillAdapter(PrefillModelAdapter):
    """Qwen3.8-Flash-Next prefill adapter: GDN linear attention plus sparse full attention, single rank."""

    name = "qwen38_flash_next"
    model_config = Qwen38FlashNextConfig
    hf_model_default = "/home/ttuser/kyamaguchi/qwen38-checkpoint"
    ttnn_cache_default = "/data/kyamaguchi/qwen38-cache"
    prefill_trace_default = ""
    default_gate_mode = "DEVICE_FP32"  # unused here; kept for contract parity

    l1_small_size = 0

    def load_hf_config(self):
        from models.demos.blackhole.qwen38_flash_next.config import Qwen38Config

        return HFConfigCarrier(Qwen38Config.from_checkpoint(Path(_checkpoint_root())))

    def layer_split_boundaries(self, num_layers: int) -> Optional[set]:
        """Only rank 0 may start, so the runner rejects a pipeline split instead of mis-slicing."""

        return {0}

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        root = Path(os.environ.get("PREFILL_TTNN_CACHE") or self.ttnn_cache_default)
        path = root / "caches" / f"prefill-{mesh_shape[0]}x{mesh_shape[1]}" / "components"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.blackhole.qwen38_flash_next.prefill.runtime import allocate_slot_caches

        return allocate_slot_caches(mesh_device=mesh_device, hf_config=hf_config, params=params)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.blackhole.qwen38_flash_next.prefill.runtime import Qwen38PrefillRuntime

        return Qwen38PrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)
