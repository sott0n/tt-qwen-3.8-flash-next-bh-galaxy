# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The two phase prims of ``chunk_gated_delta_rule`` are bound under the ``ttnn.prim.`` namespace, the runtime file
that binds them is an allowlisted runtime patch, and the prefill note says so.  No device."""

from __future__ import annotations

import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
NANOBIND = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_nanobind.cpp"
PRIMS = ("chunk_gdn_prep", "chunk_gdn_scan")


def test_the_patched_runtime_file_is_an_allowlisted_runtime_patch():
    manifest = json.loads((MODEL_DIR / "tools/release/manifest.json").read_text(encoding="utf-8"))
    assert NANOBIND in manifest["runtime_patches"]
    assert manifest["runtime_patches"] == sorted(manifest["runtime_patches"])


def test_the_prefill_note_names_the_two_prims():
    text = (MODEL_DIR / "docs/PREFILL.md").read_text(encoding="utf-8")
    for name in PRIMS:
        assert f"ttnn.prim.{name}" in text
