# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys

from qwen38_flash_next.paths import TT_METAL_ROOT

# The device fixtures (mesh_device, device_params, ...) live in tt-metal's root conftest.py.  pytest only loads the
# conftest of its own rootdir, so tt-metal's is registered as a plugin under a name that does not clash with this one.
_PLUGIN = "tt_metal_root_conftest"
_spec = importlib.util.spec_from_file_location(_PLUGIN, TT_METAL_ROOT / "conftest.py")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_PLUGIN] = _module
_spec.loader.exec_module(_module)

pytest_plugins = [_PLUGIN]
