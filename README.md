# Qwen3.8-Flash-Next on Blackhole Galaxy

The model package `qwen38_flash_next/` on top of an unmodified tt-metal, pinned as the `tt-metal/` submodule.
Everything the model needs beyond upstream ttnn lives in the package (`qwen38_flash_next/ttnn_extensions.py`).

## Set up

```bash
git clone --recurse-submodules https://github.com/sott0n/tt-qwen-3.8-flash-next-bh-galaxy.git
cd tt-qwen-3.8-flash-next-bh-galaxy/tt-metal
./build_metal.sh
./create_venv.sh
```

## Run

From the repository root, with the submodule's build:

```bash
export TT_METAL_HOME=$PWD/tt-metal
export PYTHONPATH=$PWD:$TT_METAL_HOME
$TT_METAL_HOME/python_env/bin/python -m qwen38_flash_next.tools.qwen38_chat_server --help
```

`TT_METAL_HOME` may name another tt-metal build instead of the submodule.  The launcher
`qwen38_flash_next/tools/run_qwen38_chat_server.sh` sets these itself.

## Test

```bash
tt-metal/python_env/bin/python -m pytest                     # the no-device tests
QWEN38_FUSED_DEVICE_TEST=1 MODEL_WEIGHTS_DIR=<checkpoint> \
    tt-metal/python_env/bin/python -m pytest qwen38_flash_next/tests/test_gdn_tp.py
```

`pytest.ini` puts the repository and `tt-metal/` on the path; `conftest.py` loads tt-metal's device fixtures.

## Differences from the in-tree model

- `moe_compute` runs upstream's streaming ring: no `prefill_rings`, and a prefill slab runs as 128-row calls.
- The DRAM-sharded decode matmuls place their second reader per bank by upstream's rule.

The model's own documentation is under `qwen38_flash_next/README.md` and `qwen38_flash_next/docs/`.
