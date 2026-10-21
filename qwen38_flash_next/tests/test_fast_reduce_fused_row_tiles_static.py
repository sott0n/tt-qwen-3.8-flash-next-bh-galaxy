# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source pins for the fused weighted reduce (``deepseek_moe_fast_reduce_nc_fused``) past one 32-token tile.

The op's reader used to build one score tile per expert (token rows 0..31) and the compute kernel scaled every output
tile with it, so an input taller than one row tile (tokens > 32) weighted rows 32.. with the scores of rows 0..31 (the
128-row MoE chunk's rows past the first tile came back wrong; measured on two 4x p150 hosts, 2026-09-06).  The fix builds one score tile
per (row tile, expert) and picks the group by the output tile's row tile.  These pins keep that shape (no device, no
ttnn import).
"""

from pathlib import Path
from qwen38_flash_next.paths import TT_METAL_ROOT

REPO_ROOT = TT_METAL_ROOT
OP = REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_program_factory_sizes_the_score_cb_and_passes_the_tile_geometry() -> None:
    factory = _source(OP / "device/deepseek_moe_fast_reduce_nc_fused_program_factory.cpp")

    required = (
        "const uint32_t num_row_tiles = num_tokens_x32 / tt::constants::TILE_HEIGHT;",
        ".total_size = num_row_tiles * reduction_dim_size * scores_tile_size,",
        "compute_desc.emplace_runtime_args(core, {start_tiles_read});",
    )
    for statement in required:
        assert statement in factory, statement
    # Both compute core groups carry the same three geometry compile-time args after the six the kernel had.
    assert (
        factory.count("        cb_out_id,\n        num_cores,\n        input_tensor_Wt,\n        num_row_tiles,\n") == 1
    )
    assert (
        factory.count(
            "            cb_out_id,\n            num_cores,\n            input_tensor_Wt,\n            num_row_tiles,\n"
        )
        == 1
    )


def test_device_operation_admits_any_number_of_row_tiles() -> None:
    operation = _source(OP / "device/deepseek_moe_fast_reduce_nc_fused_device_operation.cpp")

    assert "(scores_shape[0] > num_tokens - tt::constants::TILE_HEIGHT) && (scores_shape[0] <= num_tokens)" in operation
    assert "Any number of row tiles is supported" in operation
