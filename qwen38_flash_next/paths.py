# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
# The tt-metal checkout whose build runs the model: TT_METAL_HOME when set (the JIT reads kernels and headers from
# there), else the repository's submodule.
TT_METAL_ROOT = Path(os.environ.get("TT_METAL_HOME") or MODEL_DIR.parent / "tt-metal").resolve()
