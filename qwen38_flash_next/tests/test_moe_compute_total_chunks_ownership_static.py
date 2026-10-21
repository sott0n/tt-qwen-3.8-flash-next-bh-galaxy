# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from qwen38_flash_next.paths import TT_METAL_ROOT

REPO_ROOT = TT_METAL_ROOT
KERNELS = REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels"


def _source(name: str) -> str:
    return (KERNELS / name).read_text(encoding="utf-8")


def test_total_chunks_has_one_consumer_and_one_pop() -> None:
    reader = _source("tilize_reader.cpp")
    compute = _source("tilize_compute.cpp")
    writer = _source("tilize_writer.cpp")

    assert "cb_total_chunks.push_back(one_page);" in reader
    assert compute.count("cb_total_chunks.wait_front(one_page);") == 1
    assert compute.count("cb_total_chunks.pop_front(one_page);") == 1
    assert "total_chunks" not in writer


def test_writer_derives_work_from_per_expert_counts_only() -> None:
    writer = _source("tilize_writer.cpp")

    required = (
        "cb_per_expert_total_tokens.wait_front(1);",
        "num_tokens_per_expert[e] = per_expert_counts[e];",
        "uint32_t num_expert_tokens = num_tokens_per_expert[e];",
        "uint32_t num_expert_chunks = (num_expert_tokens + tokens_per_chunk - 1) / tokens_per_chunk;",
        "cb_per_expert_total_tokens.pop_front(one_page);",
    )
    for statement in required:
        assert statement in writer
