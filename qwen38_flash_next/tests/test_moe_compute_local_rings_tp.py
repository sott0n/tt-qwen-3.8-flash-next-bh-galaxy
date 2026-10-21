# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The model's moe_compute (``qwen38_flash_next.ttnn.moe_compute``: the local output path with replica rings and
packed token lists) against upstream's op on the 1x4 mesh, layer 0's experts.

Every moe_compute call of the 1-, 32- and 128-row MoE runs on the model's program at each ring count and on upstream's
op (its streaming ring); the output rows and the per-expert counts must be bitwise equal.  A 2048-row slab in one call
must be bitwise its sixteen 128-row calls.

Run:  pytest qwen38_flash_next/tests/test_moe_compute_local_rings_tp.py -v -s
Env:  MODEL_WEIGHTS_DIR, QWEN38_CACHE_ROOT (skipped without it), QWEN38_FUSED_DEVICE_TEST=1
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, HIDDEN_SIZE
from qwen38_flash_next.ttnn import moe as moe_module
from qwen38_flash_next.ttnn import moe_compute as local_rings
from qwen38_flash_next.ttnn.contracts import MESH_SHAPE

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1",
    reason="a device test: set QWEN38_FUSED_DEVICE_TEST=1 on a held four-die line (the no-device sets are masked)",
)

LAYER_INDEX = 0
ROWS = (1, 32, 128)
RING_COUNTS = (0, 2, 3)
SLAB_ROWS = 2048
OUTPUT_SLOT = 5
COUNTS_SLOT = 0


def _download(tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(local).clone() for local in ttnn.get_device_tensors(tensor)]


def _bitwise(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    def bits(t):
        return t.view(torch.int16) if t.dtype == torch.bfloat16 else t

    return all(torch.equal(bits(x), bits(y)) for x, y in zip(a, b))


class _AgainstUpstream:
    """Stands in for ttnn.experimental.moe_compute: runs the model's program at every ring count, then upstream's op
    (whose outputs it returns), and records each call whose rows or counts differ."""

    def __init__(self, upstream):
        self.upstream = upstream
        self.mismatches = []

    def __call__(self, *args, **kwargs):
        mesh = args[0].device()
        tokens = math.prod(list(args[0].shape)[:-1])
        got = {}
        for rings in RING_COUNTS:
            outputs = local_rings.moe_compute(*args, **{**kwargs, "prefill_rings": rings})
            ttnn.synchronize_device(mesh)
            got[rings] = (_download(outputs[OUTPUT_SLOT]), _download(outputs[COUNTS_SLOT]))
            for tensor in outputs[:4]:
                ttnn.deallocate(tensor)
        kwargs = {key: value for key, value in kwargs.items() if key != "prefill_rings"}
        outputs = self.upstream(*args, **kwargs)
        ttnn.synchronize_device(mesh)
        expected = (_download(outputs[OUTPUT_SLOT]), _download(outputs[COUNTS_SLOT]))
        for rings, (rows, counts) in got.items():
            same = _bitwise(rows, expected[0]) and _bitwise(counts, expected[1])
            logger.info(f"{tokens} tokens, {rings} rings: bitwise {same}")
            if not same:
                self.mismatches.append((tokens, rings))
        return outputs


def _instance(mlp, rows, **kwargs):
    return moe_module.Qwen38TTNNMoE(
        mlp.mesh_device,
        mlp.mesh_contract,
        mlp.weights,
        tt_ccl=mlp.tt_ccl,
        collective_topology=mlp.collective_topology,
        rows=rows,
        **kwargs,
    )


@run_for_blackhole()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_moe_compute_local_rings_tp(tp_harness, monkeypatch):
    harness = tp_harness
    builder, layer = harness.build_layer(LAYER_INDEX)
    upstream = getattr(ttnn.experimental.moe_compute, "__wrapped__", ttnn.experimental.moe_compute)
    against = _AgainstUpstream(upstream)
    generator = torch.Generator().manual_seed(3000)

    with builder.expert_streamer.layer(LAYER_INDEX) as (packed_w0_w1, packed_w2):
        monkeypatch.setattr(ttnn.experimental, "moe_compute", against)
        for rows in ROWS:
            mlp = layer.mlp if rows == 1 else _instance(layer.mlp, rows)
            hidden = torch.randn((1, 1, rows, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)
            result = mlp.forward(harness.upload_sharded(hidden), packed_w0_w1, packed_w2)
            ttnn.deallocate(result.hidden_sharded)
        monkeypatch.undo()
        assert not against.mismatches, f"(tokens, rings) not bitwise upstream's op: {against.mismatches}"

        hidden = torch.randn((1, 1, SLAB_ROWS, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)
        slab_outputs = {}
        for one_call in ("0", "1"):
            monkeypatch.setenv(moe_module.MOE_SLAB_ONE_CALL_ENV, one_call)
            mlp = _instance(layer.mlp, SLAB_ROWS, admitted_rows=moe_module.SUPPORTED_ROWS + (SLAB_ROWS,))
            assert mlp.slab_one_call == (one_call == "1")
            result = mlp.forward(harness.upload_sharded(hidden), packed_w0_w1, packed_w2)
            slab_outputs[one_call] = harness.download_sharded(result.hidden_sharded).to(torch.bfloat16)
            mlp.release_owned_buffers()
        assert torch.equal(
            slab_outputs["0"].view(torch.int16), slab_outputs["1"].view(torch.int16)
        ), "the one-call slab is not bitwise its 128-row calls"
