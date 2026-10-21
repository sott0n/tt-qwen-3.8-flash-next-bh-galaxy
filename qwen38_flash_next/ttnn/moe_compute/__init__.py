# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``moe_compute``'s local output path as a ``generic_op`` program: the replica prefill rings (``prefill_rings`` 2 or
3, every ring core keeping one whole expert slice resident) and the packed (token, k slot) lists that let one call
cover a prefill slab.  The kernels under ``kernels/`` and the program below follow tt-metal's moe_compute
(``ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute``) for its LocalOutput path only: a cluster axis of extent
one, no combine kernels, no fabric, dm1 writing each expert's rows straight into the caller's ``[k, T, H]`` output.

:func:`moe_compute` takes the arguments of ``ttnn.experimental.moe_compute`` and returns its six slots; slots 3 and 4
are one tensor (the chunk staging buffer) and slot 1 a placeholder, as on the op's local output path."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNELS = Path(__file__).resolve().parent / "kernels"

TILE = 32
TOKENS_PER_CHUNK = 32
# Blackhole's L1 and DRAM alignment (hal::get_l1_alignment / get_dram_alignment).
L1_ALIGNMENT = 16
DRAM_ALIGNMENT = 64
NOC_MAX_BURST_BYTES = 16384
BF16_TILE_BYTES = 2048
BFP4_TILE_BYTES = 576
UINT32_BYTES = 4
# Pages of the non-tile CBs: at least one 16-B word, so the unpacker's fifo fields are non-zero.
NON_TILE_PAGE_BYTES = 16

# moe_ring_common.h
W0_W1_TXNS_PER_BLOCK = 2
DEFAULT_TILES_PER_TXN = 14
ALT_TILES_PER_TXN = 20
W0_W1_BLOCK_TILES_W = 4
W0_W1_HALF_BLOCK_TILES_W = 2
W2_TILES_PER_A2A_ITER_W = 4
W2_HALF_A2A_ITER_TILES_W = 2
TOKEN_LIST_SEGMENT_ALIGN_ENTRIES = 16
TOKEN_LIST_ENTRY_BYTES = 4
TOKEN_LIST_PAIR_BYTES = 8
MAX_CHUNK_HALVES = 5

# moe_core_placement.cpp
MAX_TILIZE_CORES = 4
COMBINE_STRIP_WIDTH = 2

SILU = 0


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


def _align(value: int, alignment: int) -> int:
    return _div_up(value, alignment) * alignment


def chunk_halves(prefill_rings: int) -> int:
    return 2 if prefill_rings < 2 else prefill_rings + 1


def _shard_tiles(n_tiles: int, core: int, n_cores: int) -> int:
    n_big = n_tiles % n_cores
    big = n_big > 0 and (core * n_big) % n_cores < n_big
    return n_tiles // n_cores + (1 if big else 0)


def _even_stride_at_least_a2a_width(tiles: int) -> int:
    even = (tiles + 1) & ~1
    return max(even, W2_TILES_PER_A2A_ITER_W)


def _w0_w1_compact(nt: int, n_cores: int) -> bool:
    max_cols = _div_up(nt, n_cores)
    return max_cols < _even_stride_at_least_a2a_width(max_cols)


def _w0_w1_stored_cols(nt: int, core: int, n_cores: int) -> int:
    if _w0_w1_compact(nt, n_cores):
        return _shard_tiles(nt, core, n_cores)
    return _even_stride_at_least_a2a_width(_div_up(nt, n_cores))


def _a2a_exchange_tiles(nt: int, n_cores: int) -> int:
    max_cols = _div_up(nt, n_cores)
    if not _w0_w1_compact(nt, n_cores):
        return _even_stride_at_least_a2a_width(max_cols)
    return max(max_cols, 2)


def _block_tiles_h(tiles_per_txn: int) -> int:
    return W0_W1_TXNS_PER_BLOCK * tiles_per_txn // W0_W1_BLOCK_TILES_W


def _half_block_tiles_h(tiles_per_txn: int) -> int:
    return W0_W1_TXNS_PER_BLOCK * tiles_per_txn // W0_W1_HALF_BLOCK_TILES_W


def _tiles_per_txn(ht: int, nt: int, has_bias: bool, n_cores: int) -> int:
    return ALT_TILES_PER_TXN if (ht, nt, has_bias, n_cores) == (80, 20, False, 8) else DEFAULT_TILES_PER_TXN


def _weight_cb_slots(tiles_per_txn: int) -> int:
    fit = (3 * W0_W1_TXNS_PER_BLOCK * DEFAULT_TILES_PER_TXN) // (W0_W1_TXNS_PER_BLOCK * tiles_per_txn)
    return max(fit, 3)


def _w0_w1_blocks_for_cols(cols: int, blocks_per_col: int, blocks_per_half_col: int) -> int:
    return (cols // 2) * blocks_per_col + (cols % 2) * blocks_per_half_col


def _w0_w1_bank_blocks_per_expert(k_tiles: int, nt: int, n_cores: int, num_banks: int, tiles_per_txn: int) -> int:
    per_col = _div_up(k_tiles, _block_tiles_h(tiles_per_txn))
    per_half_col = _div_up(k_tiles, _half_block_tiles_h(tiles_per_txn))
    expert_blocks = sum(
        _w0_w1_blocks_for_cols(_w0_w1_stored_cols(nt, c, n_cores), per_col, per_half_col) for c in range(n_cores)
    )
    return _div_up(expert_blocks, num_banks)


def _w2_core_blocks_per_expert(ht: int, k_tiles: int, n_cores: int, tiles_per_txn: int) -> int:
    iters = _div_up(_div_up(ht, n_cores), W2_TILES_PER_A2A_ITER_W)
    last = _div_up(ht, n_cores) % W2_TILES_PER_A2A_ITER_W
    half = 1 if tiles_per_txn != DEFAULT_TILES_PER_TXN and last != 0 and last <= W2_HALF_A2A_ITER_TILES_W else 0
    return (iters - half) * _div_up(k_tiles, _block_tiles_h(tiles_per_txn)) + half * _div_up(
        k_tiles, _half_block_tiles_h(tiles_per_txn)
    )


def _token_list_align(entries: int) -> int:
    return _align(entries, TOKEN_LIST_SEGMENT_ALIGN_ENTRIES)


def token_list_page_words(tokens: int, k: int, experts_per_device: int) -> int:
    header = _token_list_align(experts_per_device + 1)
    capacity = (
        _token_list_align(tokens * k + (TOKEN_LIST_SEGMENT_ALIGN_ENTRIES - 1) * experts_per_device) + TOKENS_PER_CHUNK
    )
    return header + capacity


def data_parallel_cores(hidden: int, ring_size: int) -> int:
    """The output width shard: the largest d <= 4 dividing both the hidden tiles and the ring (dm1 maps ring cores to
    output columns by ``ring_size / d``)."""

    hidden_tiles = hidden // TILE
    return next(d for d in (4, 3, 2, 1) if hidden_tiles % d == 0 and ring_size % d == 0)


# ---------------------------------------------------------------------------------------------------------------------
# Core placement (moe_core_placement.cpp), in logical coordinates.  Only the DRAM-bank-adjacent ring is admitted: the
# replica rings need its free cells, and the model's grid always places it.
# ---------------------------------------------------------------------------------------------------------------------

Core = tuple[int, int]


@dataclass(frozen=True)
class Placement:
    tilize: tuple[Core, ...]
    matmul: tuple[Core, ...]
    ring: tuple[Core, ...]  # ring-major: ring r's core of position p at r * len(matmul) + p; ring 0 = matmul
    combine: tuple[Core, ...]
    tilize_box: tuple[Core, Core]
    matmul_box: tuple[Core, Core]


def _box(cores) -> tuple[Core, Core]:
    xs = [x for x, _ in cores]
    ys = [y for _, y in cores]
    return (min(xs), min(ys)), (max(xs), max(ys))


def _box_cells(box) -> set[Core]:
    (x0, y0), (x1, y1) = box
    return {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}


def _intersects(a, b) -> bool:
    (ax0, ay0), (ax1, ay1) = a
    (bx0, by0), (bx1, by1) = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _box_size(box) -> int:
    (x0, y0), (x1, y1) = box
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def _rect_free(avoid, x0: int, y0: int, width: int, height: int) -> bool:
    return all((x0 + dx, y0 + dy) not in avoid for dy in range(height) for dx in range(width))


def _find_rect(avoid, grid: Core, width: int, height: int, max_y: int):
    y_limit = min(max_y + 1, grid[1])
    if width > grid[0] or height == 0:
        return None
    for sx in range(grid[0] - width, -1, -1):
        for sy in range(0, y_limit - height + 1):
            if _rect_free(avoid, sx, sy, width, height):
                return (sx, sy), (sx + width - 1, sy + height - 1)
    return None


def _rect_cores_row_major(rect, count: int) -> list[Core]:
    (x0, y0), (x1, y1) = rect
    cores = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    return cores[:count]


def _place_combine_and_tilize(grid: Core, matmul_box, num_combine: int, target_tilize: int, hidden_tiles: int):
    base_avoid = _box_cells(matmul_box)
    combine_max_y = grid[1] - 3 if grid[1] >= 3 else grid[1] - 1
    tilize_min_y = grid[1] - 2 if grid[1] >= 2 else grid[1] - 1

    strip = _find_rect(base_avoid, grid, COMBINE_STRIP_WIDTH, _div_up(num_combine, COMBINE_STRIP_WIDTH), combine_max_y)
    if strip is None:
        for width in sorted((w for w in range(1, num_combine + 1) if num_combine % w == 0), reverse=True):
            strip = _find_rect(base_avoid, grid, width, num_combine // width, combine_max_y)
            if strip is not None:
                break
    if strip is not None:
        combine = _rect_cores_row_major(strip, num_combine)
    else:
        y_limit = min(combine_max_y + 1, grid[1])
        combine = [(x, y) for y in range(y_limit) for x in range(grid[0]) if (x, y) not in base_avoid][:num_combine]
    if len(combine) != num_combine:
        return None
    combine_box = _box(combine)
    if _box_size(combine_box) != num_combine or _intersects(combine_box, matmul_box):
        return None

    tilize_avoid = base_avoid | _box_cells(combine_box)
    for count in range(target_tilize, 0, -1):
        if hidden_tiles % count:
            continue
        tilize = None
        sy = grid[1] - 2
        if grid[0] >= 2 and grid[1] >= 2:
            for sx in range(grid[0] - 2, -1, -1):
                if _rect_free(tilize_avoid, sx, sy, 2, 2):
                    tilize = [(sx + 1, sy + 1), (sx + 1, sy), (sx, sy + 1), (sx, sy)][:count]
                    break
        if tilize is None:
            tilize = [
                (x, y)
                for y in range(grid[1] - 1, tilize_min_y - 1, -1)
                for x in range(grid[0] - 1, -1, -1)
                if (x, y) not in tilize_avoid
            ][:count]
        if len(tilize) != count:
            continue
        tilize_box = _box(tilize)
        if _intersects(tilize_box, combine_box) or _intersects(tilize_box, matmul_box):
            continue
        return combine, tilize, tilize_box
    return None


@functools.lru_cache(maxsize=None)
def _placement(
    grid: Core, bank_cores: tuple[Core, ...], token_parallel: int, width_parallel: int, hidden: int, prefill_rings: int
) -> Placement:
    hidden_tiles = hidden // TILE
    target_tilize = min(MAX_TILIZE_CORES, hidden_tiles)
    while target_tilize > 1 and hidden_tiles % target_tilize:
        target_tilize -= 1
    matmul = list(bank_cores)
    matmul_box = _box(matmul)
    placed = _place_combine_and_tilize(grid, matmul_box, token_parallel * width_parallel, target_tilize, hidden_tiles)
    if placed is None:
        raise RuntimeError(
            f"moe_compute: no disjoint combine/tilize placement beside the DRAM-adjacent ring {matmul} on {grid}"
        )
    combine, tilize, tilize_box = placed

    ring = list(matmul)
    taken = set(matmul)
    (box_lo, _), (box_hi, _) = matmul_box
    for r in range(1, max(prefill_rings, 1)):
        for x0, y in matmul:
            x = x0 + r if 2 * x0 < box_lo + box_hi else x0 - r
            if not box_lo <= x <= box_hi or (x, y) in taken:
                raise RuntimeError(f"moe_compute: prefill_rings={prefill_rings}: no free cell for ring {r} at ({x}, {y})")
            taken.add((x, y))
            ring.append((x, y))
    combine.sort()
    return Placement(tuple(tilize), tuple(matmul), tuple(ring), tuple(combine), tilize_box, matmul_box)


def _coord(core: Core) -> ttnn.CoreCoord:
    return ttnn.CoreCoord(core[0], core[1])


def _cores(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(_coord(c), _coord(c)) for c in sorted(set(cores))])


def _rect(box) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(_coord(box[0]), _coord(box[1]))])


class _Semaphores:
    """Semaphore ids the way ``CreateSemaphore`` hands them out: the smallest id free on every core of the range."""

    def __init__(self):
        self.used: dict[Core, set[int]] = {}
        self.descriptors: list[ttnn.SemaphoreDescriptor] = []

    def create(self, cores, crs: ttnn.CoreRangeSet) -> int:
        cores = set(cores)
        sid = next(i for i in range(64) if all(i not in self.used.get(c, ()) for c in cores))
        for c in cores:
            self.used.setdefault(c, set()).add(sid)
        self.descriptors.append(ttnn.SemaphoreDescriptor(id=sid, core_ranges=crs, initial_value=0))
        return sid


def _cb(index: int, dtype, page_bytes: int, pages: int, crs: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _tensor_cb(index: int, tensor, dtype, page_bytes: int, pages: int, crs: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor, total_size=pages * page_bytes, core_ranges=crs)
    cb.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)]
    return cb


def _accessor(tensor) -> list[int]:
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def _dm(source: str, crs, ct_args, named, runtime_args, processor, noc, *, dynamic=False, defines=()):
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / source),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=list(ct_args),
        named_compile_time_args=sorted(named.items()),
        defines=list(defines),
        runtime_args=runtime_args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=processor,
            noc=noc,
            noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC if dynamic else ttnn.NOC_MODE.DM_DEDICATED_NOC,
        ),
    )


def _compute(source: str, crs, ct_args, named, runtime_args, *, fidelity, approx):
    return ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / source),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=list(ct_args),
        named_compile_time_args=sorted(named.items()),
        runtime_args=runtime_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fidelity,
            math_approx_mode=approx,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            bfp8_pack_precise=False,
        ),
    )


def _logical_shape(tensor) -> list[int]:
    return list(tensor.shape)


def _height_sharded_l1(grid: ttnn.CoreRangeSet, shard) -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, list(shard), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _allocate(shape, dtype, layout, mesh, memory_config):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, mesh, memory_config)


def moe_compute(
    tilize_input,
    indices,
    scores,
    mapping,
    w0_w1,
    w2,
    *,
    layer_id: int = 0,
    output_height_shard_dim: int,
    intermediate_size: int,
    has_bias: bool = False,
    cluster_axis: int | None = None,
    topology=None,
    num_links=None,
    mux_core_range_set=None,
    output_memory_config=None,
    optional_output_tensor=None,
    optional_cross_device_semaphore=None,
    activation_type=None,
    compute_only: bool = False,
    num_shared_experts_per_device: int | None = None,
    prefill_rings: int | None = None,
):
    mesh = tilize_input.device()
    if compute_only or cluster_axis is None or tuple(mesh.shape)[cluster_axis] != 1:
        raise ValueError("this moe_compute runs the local output path only: a cluster_axis of extent 1")
    if mux_core_range_set is not None or optional_cross_device_semaphore is not None:
        raise ValueError("the local output path takes no mux cores and no cross-device semaphore")
    if optional_output_tensor is None:
        raise ValueError("the local output path writes into the caller's optional_output_tensor")
    if (num_shared_experts_per_device or 0) != 0 or has_bias:
        raise ValueError("shared experts and expert biases are not admitted")
    if activation_type is not None and int(activation_type.value if hasattr(activation_type, "value") else 0) != SILU:
        raise ValueError(f"only the SiLU activation is admitted, got {activation_type}")
    rings = int(prefill_rings or 0)
    if rings not in (0, 2, 3, 4):
        raise ValueError(f"prefill_rings must be 0, 2, 3 or 4, got {rings}")

    input_shape = _logical_shape(tilize_input)
    hidden = input_shape[-1]
    hidden_tiles = hidden // TILE
    intermediate_tiles = intermediate_size // TILE
    tokens = 1
    for d in input_shape[:-1]:
        tokens *= d
    k = _logical_shape(indices)[-1]
    w0_w1_shape = _logical_shape(w0_w1)
    experts_per_device = w0_w1_shape[2]
    experts = _logical_shape(mapping)[-1]

    grid_size = mesh.compute_with_storage_grid_size()
    grid = (grid_size.x, grid_size.y)
    bank_cores = tuple(
        (c.x, c.y) for c in mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.RISCV_0_default)
    )
    ring_size = len(bank_cores)
    num_banks = mesh.dram_grid_size().x
    if ring_size != num_banks:
        raise RuntimeError(f"moe_compute: {ring_size} DRAM-adjacent workers for {num_banks} banks")
    width_parallel = data_parallel_cores(hidden, ring_size)
    place = _placement(grid, bank_cores, output_height_shard_dim, width_parallel, hidden, rings)
    halves = chunk_halves(rings)

    # Outputs (moe_compute_device_operation.cpp, compute_output_specs for the local output path).
    full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
    grid_cores = grid[0] * grid[1]
    counts_row_words = _align(experts_per_device * UINT32_BYTES, L1_ALIGNMENT) // UINT32_BYTES
    counts = _allocate(
        [grid_cores, counts_row_words],
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        _height_sharded_l1(full_grid, [1, counts_row_words]),
    )
    placeholder = _allocate(
        [1, L1_ALIGNMENT // UINT32_BYTES], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    page_words = token_list_page_words(tokens, k, experts_per_device)
    token_lists = _allocate([1, page_words], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    staging = _allocate(
        [grid_cores, halves, TOKENS_PER_CHUNK, hidden],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mesh,
        _height_sharded_l1(full_grid, [halves * TOKENS_PER_CHUNK, hidden]),
    )
    output = optional_output_tensor
    if _logical_shape(output) != [k, tokens, hidden]:
        raise ValueError(f"moe_compute: the output must be [{k}, {tokens}, {hidden}], got {_logical_shape(output)}")

    descriptor = _mesh_program(
        mesh,
        place,
        rings=rings,
        tokens=tokens,
        hidden=hidden,
        hidden_tiles=hidden_tiles,
        intermediate_tiles=intermediate_tiles,
        k=k,
        experts=experts,
        experts_per_device=experts_per_device,
        num_banks=num_banks,
        width_parallel=width_parallel,
        layer_id=layer_id,
        height_shard_dim=output_height_shard_dim,
        tensors=(tilize_input, indices, scores, mapping, w0_w1, w2, counts, placeholder, token_lists, staging, output),
    )
    ttnn.generic_op(
        [tilize_input, indices, scores, mapping, w0_w1, w2, counts, placeholder, token_lists, staging, output],
        descriptor,
    )
    # Every coordinate holds the partial of its own experts over the whole token set: no axis is sharded.
    for tensor in (counts, placeholder, token_lists, staging, output):
        tensor.update_tensor_topology(tilize_input.tensor_topology())
    return [counts, placeholder, token_lists, staging, staging, output]


def _mesh_program(
    mesh,
    place: Placement,
    *,
    rings,
    tokens,
    hidden,
    hidden_tiles,
    intermediate_tiles,
    k,
    experts,
    experts_per_device,
    num_banks,
    width_parallel,
    layer_id,
    height_shard_dim,
    tensors,
):
    (tilize_input, indices, scores, mapping, w0_w1, w2, counts, placeholder, token_lists, staging, output) = tensors
    halves = chunk_halves(rings)
    num_rings = max(rings, 1)
    phys = {}

    def physical(core: Core):
        if core not in phys:
            c = mesh.worker_core_from_logical_core(_coord(core))
            phys[core] = (c.x, c.y)
        return phys[core]

    tilize = list(place.tilize)
    matmul = list(place.matmul)
    ring = list(place.ring)
    n_tilize = len(tilize)
    n_matmul = len(matmul)
    tilize_crs = _cores(tilize)
    ring_crs = _cores(ring)
    tilize_matmul = set(tilize) | set(ring)
    tilize_matmul_crs = _cores(tilize_matmul)
    all_workers = tilize_matmul | set(place.combine)
    all_workers_crs = _cores(all_workers)
    all_box = _box(all_workers)

    # The drain's matmul-bound multicasts exclude the sender when it sits inside the ring's box.
    senders = [tilize[0]] + ([tilize[n_tilize // 2]] if n_tilize > 1 else [])
    inside = [all(place.matmul_box[0][i] <= s[i] <= place.matmul_box[1][i] for i in (0, 1)) for s in senders]
    if any(inside) and not all(inside):
        raise RuntimeError("moe_compute: the tilize multicast senders straddle the matmul box")
    matmul_dests = _box_size(place.matmul_box) - (1 if inside[0] else 0)

    sems = _Semaphores()
    partial_md_ready = sems.create(tilize, tilize_crs)
    tilize_chunk_ready = sems.create(tilize, tilize_crs)
    initial_gather = sems.create(tilize, tilize_crs)
    ring_sem = sems.create(ring, ring_crs)
    a2a_free = sems.create(ring, ring_crs)
    metadata_ready = sems.create(tilize_matmul, tilize_matmul_crs)
    matmul_chunk_ready = sems.create(tilize_matmul, tilize_matmul_crs)
    half_free = [0] * MAX_CHUNK_HALVES
    for h in range(halves):
        half_free[h] = sems.create(tilize_matmul, tilize_matmul_crs)
    feed_go = sems.create(tilize, tilize_crs)
    gather = [0] * MAX_CHUNK_HALVES
    for h in range(halves):
        gather[h] = sems.create(tilize, tilize_crs)
    combine_sync = sems.create(ring, ring_crs)
    matmul_combine_sync = sems.create(ring, ring_crs)

    # Tilize work split: each tilize core owns a contiguous slice of every token row.
    input_aligned_page = tilize_input.buffer_aligned_page_size()
    subtoken_bytes = _align(_div_up(input_aligned_page, n_tilize), L1_ALIGNMENT)
    units = _div_up(input_aligned_page, subtoken_bytes)
    if units % n_tilize:
        raise RuntimeError(f"moe_compute: {units} subtoken units do not split over {n_tilize} tilize cores")
    subtoken_sizes = []
    offset = 0
    for _ in range(n_tilize):
        size = min(units // n_tilize * subtoken_bytes, input_aligned_page - offset)
        subtoken_sizes.append(size)
        offset += size
    max_subtoken = units // n_tilize * subtoken_bytes

    staging_pages = halves * hidden_tiles
    counts_page = counts.buffer_page_size()
    e_t_page = token_lists.buffer_page_size()
    if e_t_page != token_list_page_words(tokens, k, experts_per_device) * UINT32_BYTES:
        raise RuntimeError("moe_compute: the packed token-list page does not match the kernels' layout")
    max_tokens_per_tilize = tokens - (n_tilize - 1) * (tokens // n_tilize)
    pairs_capacity = _align(_div_up(max_tokens_per_tilize, 2) * k * TOKEN_LIST_PAIR_BYTES, L1_ALIGNMENT)
    remote_counts_entry = _align((experts_per_device + 2) * UINT32_BYTES, L1_ALIGNMENT)

    cbs = [
        _tensor_cb(0, staging, ttnn.bfloat16, BF16_TILE_BYTES, staging_pages, _cores_full(staging)),
        _tensor_cb(1, staging, ttnn.bfloat16, BF16_TILE_BYTES, staging_pages, _cores_full(staging)),
        _tensor_cb(2, counts, ttnn.uint32, counts_page, 1, all_workers_crs),
        _cb(7, ttnn.uint32, e_t_page, 1, tilize_crs),
        _tensor_cb(
            4,
            indices,
            _uint_dtype(indices),
            indices.buffer_aligned_page_size(),
            indices.buffer_num_pages(),
            tilize_crs,
        ),
        _tensor_cb(
            6,
            scores,
            scores.dtype,
            scores.buffer_aligned_page_size(),
            scores.buffer_num_pages(),
            tilize_crs,
        ),
        _cb(5, mapping.dtype, mapping.buffer_aligned_page_size(), mapping.buffer_num_pages(), tilize_crs),
        _cb(8, tilize_input.dtype, max_subtoken, TOKENS_PER_CHUNK, tilize_crs),
        _cb(11, ttnn.uint32, UINT32_BYTES * experts_per_device, 1, tilize_crs),
        _cb(9, ttnn.uint32, pairs_capacity, 1, tilize_crs),
        _cb(10, ttnn.uint32, pairs_capacity, 1, tilize_crs),
    ]
    if n_tilize > 1:
        cbs.append(_cb(12, ttnn.uint32, 2 * pairs_capacity, n_tilize - 1, tilize_crs))
    cbs += [
        _cb(14, ttnn.uint32, remote_counts_entry, max(n_tilize - 1, 1), tilize_crs),
        _cb(3, ttnn.uint32, NON_TILE_PAGE_BYTES, 1, tilize_crs),
    ]

    # The ring cores' CBs.
    tiles_per_txn = _tiles_per_txn(hidden_tiles, intermediate_tiles, False, n_matmul)
    tiles_per_block = W0_W1_TXNS_PER_BLOCK * tiles_per_txn
    a2a_pages = _a2a_exchange_tiles(intermediate_tiles, n_matmul)
    order = sorted(range(n_matmul), key=lambda b: (-physical(matmul[b])[1], -physical(matmul[b])[0]))
    ring_pos = {bank: pos for pos, bank in enumerate(order)}
    replay_slice = rings >= 2
    if replay_slice:
        per_col = _div_up(hidden_tiles, _block_tiles_h(tiles_per_txn))
        per_half_col = _div_up(hidden_tiles, _half_block_tiles_h(tiles_per_txn))
        w2_blocks = _w2_core_blocks_per_expert(hidden_tiles, intermediate_tiles, n_matmul, tiles_per_txn)
        by_blocks: dict[int, list[Core]] = {}
        for bank in range(n_matmul):
            cols = _w0_w1_stored_cols(intermediate_tiles, ring_pos[bank], n_matmul)
            blocks = _w0_w1_blocks_for_cols(cols, per_col, per_half_col) + w2_blocks
            for r in range(num_rings):
                by_blocks.setdefault(blocks, []).append(ring[r * n_matmul + bank])
        for blocks, cores in sorted(by_blocks.items()):
            cbs.append(_cb(3, ttnn.bfloat4_b, BFP4_TILE_BYTES, blocks * tiles_per_block, _cores(cores)))
    else:
        cbs.append(_cb(3, ttnn.bfloat4_b, BFP4_TILE_BYTES, _weight_cb_slots(tiles_per_txn) * tiles_per_block, ring_crs))
    cbs += [
        _cb(4, ttnn.float32, NON_TILE_PAGE_BYTES, 1, ring_crs),
        _cb(5, ttnn.float32, NON_TILE_PAGE_BYTES, n_matmul, ring_crs),
        _cb(6, ttnn.bfloat16, BF16_TILE_BYTES, a2a_pages * n_matmul, ring_crs),
        _cb(7, ttnn.uint32, NON_TILE_PAGE_BYTES, 2, ring_crs),
    ]
    map_bytes = _align(TOKENS_PER_CHUNK * TOKEN_LIST_ENTRY_BYTES, DRAM_ALIGNMENT) + DRAM_ALIGNMENT
    cbs.append(_cb(9, ttnn.uint32, map_bytes, 1, ring_crs))

    tilize_phys = [physical(c) for c in tilize]
    primary = n_tilize // 2
    secondary = n_tilize - primary
    t_box_p = (physical(place.tilize_box[0]), physical(place.tilize_box[1]))
    m_box_p = (physical(place.matmul_box[0]), physical(place.matmul_box[1]))
    a_box_p = (physical(all_box[0]), physical(all_box[1]))
    combine_p = physical(place.combine[0])
    mcaster = tilize_phys[primary if n_tilize > 1 else 0]

    tilize_named = {
        "tilize_input_cb_id": 8,
        "tilize_output_cb_id": 0,
        "total_chunks_cb_id": 3,
        "indices_tensor_cb_id": 4,
        "scores_tensor_cb_id": 6,
        "mapping_tensor_cb_id": 5,
        "e_t_cb_id": 7,
        "expert_activation_cb_id": 9,
        "per_expert_total_tokens_cb_id": 2,
        "brisc_e_t_cb_id": 10,
        "brisc_expert_counts_cb_id": 11,
        "brisc_expert_activation_cb_id": 12,
        "brisc_activated_count_cb_id": 13,
        "remote_counts_cb_id": 14,
        "l1_alignment": L1_ALIGNMENT,
        "dram_alignment": DRAM_ALIGNMENT,
        "e_t_entry_size": L1_ALIGNMENT,
        "input_pages": tilize_input.buffer_num_pages(),
        "indices_pages": indices.buffer_num_pages(),
        "mapping_pages": mapping.buffer_num_pages(),
        "scores_pages": scores.buffer_num_pages(),
        "shared_cb_num_pages": staging_pages,
        "chunk_slot_pages": hidden_tiles,
        "input_page_size": tilize_input.buffer_page_size(),
        "indices_page_size": indices.buffer_page_size(),
        "mapping_page_size": mapping.buffer_page_size(),
        "per_expert_total_tokens_output_page_size": counts_page,
        "expert_activation_output_page_size": placeholder.buffer_page_size(),
        "e_t_output_page_size": e_t_page,
        "tilize_output_page_size": BF16_TILE_BYTES,
        "aligned_input_page_size": input_aligned_page,
        "aligned_indices_page_size": indices.buffer_aligned_page_size(),
        "aligned_mapping_page_size": mapping.buffer_aligned_page_size(),
        "aligned_scores_page_size": scores.buffer_aligned_page_size(),
        "tokens": tokens,
        "hidden_size": hidden,
        "remote_counts_entry_size": remote_counts_entry,
        "experts": experts,
        "experts_per_device": experts_per_device,
        "selected_experts_k": k,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "num_devices": mesh.get_num_devices(),
        "mesh_rows": tuple(mesh.shape)[0],
        "mesh_cols": tuple(mesh.shape)[1],
        "linearized_mesh_coord": 0,
        "cluster_axis": 0,
        "drain_core_noc_x": tilize_phys[0][0],
        "drain_core_noc_y": tilize_phys[0][1],
        "primary_mcast_gather_group_num_cores": primary,
        "secondary_mcast_gather_group_num_cores": secondary,
        "num_tilize_cores": n_tilize,
        "tilize_mcast_start_x": t_box_p[0][0],
        "tilize_mcast_start_y": t_box_p[0][1],
        "tilize_mcast_end_x": t_box_p[1][0],
        "tilize_mcast_end_y": t_box_p[1][1],
        "tilize_bounding_box_num_cores": _box_size(place.tilize_box),
        "num_matmul_cores": n_matmul,
        "matmul_mcast_start_x": m_box_p[0][0],
        "matmul_mcast_start_y": m_box_p[0][1],
        "matmul_mcast_end_x": m_box_p[1][0],
        "matmul_mcast_end_y": m_box_p[1][1],
        "matmul_bounding_box_num_cores": matmul_dests,
        "all_worker_cores_mcast_start_x": a_box_p[0][0],
        "all_worker_cores_mcast_start_y": a_box_p[0][1],
        "all_worker_cores_mcast_end_x": a_box_p[1][0],
        "all_worker_cores_mcast_end_y": a_box_p[1][1],
        "all_worker_cores_bounding_box_num_cores": _box_size(all_box),
        "partial_metadata_ready_semaphore_id": partial_md_ready,
        "metadata_ready_semaphore_id": metadata_ready,
        "chunk_halves": halves,
        **{f"half_free_semaphore_id_{h}": half_free[h] for h in range(MAX_CHUNK_HALVES)},
        "tilize_chunk_ready_semaphore_id": tilize_chunk_ready,
        "matmul_chunk_ready_semaphore_id": matmul_chunk_ready,
        "initial_gather_semaphore_id": initial_gather,
        "secondary_mcaster_noc_x": mcaster[0],
        "secondary_mcaster_noc_y": mcaster[1],
        "feed_go_semaphore_id": feed_go,
        **{f"gather_semaphore_id_{h}": gather[h] for h in range(MAX_CHUNK_HALVES)},
        "combine_sync_semaphore_id": combine_sync,
        "combine_sync_noc_x": combine_p[0],
        "combine_sync_noc_y": combine_p[1],
        "compute_only": 0,
        "local_output": 1,
        "pairs_capacity_bytes": pairs_capacity,
    }
    tilize_ct = []
    for t in (tilize_input, indices, scores, mapping, counts, placeholder, token_lists):
        tilize_ct += _accessor(t)

    tilize_rt = []
    tilize_compute_rt = []
    group_offset = 0
    global_offset = 0
    group_size = [sum(subtoken_sizes[:primary]), sum(subtoken_sizes[primary:])]
    per_core_tokens = tokens // n_tilize
    tile_width_bytes = TILE * tilize_input.element_size()
    for i, core in enumerate(tilize):
        gather_core = tilize_phys[0] if i < primary else tilize_phys[primary]
        args = [
            tilize_input.buffer_address(),
            indices.buffer_address(),
            scores.buffer_address(),
            mapping.buffer_address(),
            counts.buffer_address(),
            placeholder.buffer_address(),
            token_lists.buffer_address(),
            1 if i == 0 else 0,
            1 if n_tilize > 1 and i == primary else 0,
            gather_core[0],
            gather_core[1],
            global_offset,
            group_offset,
            group_size[0] if i < primary else group_size[1],
            subtoken_sizes[i],
            i * per_core_tokens,
            tokens if i == n_tilize - 1 else (i + 1) * per_core_tokens,
            i,
        ]
        for x, y in tilize_phys:
            args += [x, y]
        tilize_rt.append((_coord(core), args))
        tilize_compute_rt.append((_coord(core), [subtoken_sizes[i] // tile_width_bytes]))
        global_offset += subtoken_sizes[i]
        group_offset += subtoken_sizes[i]
        if i == primary - 1:
            group_offset = 0

    compute_tilize_named = {
        "tilize_input_cb_id": 8,
        "tilize_output_cb_id": 0,
        "total_chunks_cb_id": 3,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "max_tiles_per_local_chunk": max_subtoken // tile_width_bytes,
        "shared_cb_num_pages": staging_pages,
        "chunk_slot_pages": hidden_tiles,
    }

    matmul_ct = []
    for t in (w0_w1, w2, staging, token_lists, output):
        matmul_ct += _accessor(t)

    w0_w1_layers = _logical_shape(w0_w1)[1]
    w0_w1_expected = (
        num_banks
        * w0_w1_layers
        * experts_per_device
        * _w0_w1_bank_blocks_per_expert(hidden_tiles, intermediate_tiles, n_matmul, num_banks, tiles_per_txn)
        * tiles_per_block
    )
    w2_layers = _logical_shape(w2)[1]
    w2_expected = (
        n_matmul
        * w2_layers
        * experts_per_device
        * _w2_core_blocks_per_expert(hidden_tiles, intermediate_tiles, n_matmul, tiles_per_txn)
        * tiles_per_block
    )
    if w0_w1.buffer_num_pages() != w0_w1_expected or w2.buffer_num_pages() != w2_expected:
        raise RuntimeError(
            f"moe_compute: weight pages {w0_w1.buffer_num_pages()}/{w2.buffer_num_pages()}, the packed layout needs "
            f"{w0_w1_expected}/{w2_expected}"
        )

    staging_shard_rows = halves * TOKENS_PER_CHUNK
    segment_width = hidden // width_parallel
    token_expert_row_offset = (staging_shard_rows // 2) * (hidden // segment_width)
    matmul_named = {
        "num_experts": experts_per_device,
        "num_shared_experts": 0,
        "shared_expert_tp_factor": tuple(mesh.shape)[1],
        "layer_id": layer_id,
        "has_bias": 0,
        "num_cores": n_matmul,
        "num_banks": num_banks,
        "a2a_free_semaphore_id": a2a_free,
        "prefill_rings": rings,
        "w2_pages_per_ring_core_total": w2.buffer_num_pages() // n_matmul,
        "activation_function": SILU,
        "metadata_ready_semaphore_id": metadata_ready,
        "matmul_chunk_ready_semaphore_id": matmul_chunk_ready,
        "chunk_halves": halves,
        **{f"half_free_semaphore_id_{h}": half_free[h] for h in range(MAX_CHUNK_HALVES)},
        "per_expert_total_tokens_cb_id": 2,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "tilize_drain_core_noc_x": tilize_phys[0][0],
        "tilize_drain_core_noc_y": tilize_phys[0][1],
        "combine_shard_width_tiles": hidden // TILE // width_parallel,
        "tile_height": TILE,
        "tile_width": TILE,
        "tile_width_size_bytes": TILE * 2,
        "token_expert_row_offset": token_expert_row_offset,
        "height_shard_dim": height_shard_dim,
        "width_shard_dim": width_parallel,
        "hidden_tiles": hidden_tiles,
        "tiles_per_txn": tiles_per_txn,
        "intermediate_tiles": intermediate_tiles,
        "noc_max_burst_bytes": NOC_MAX_BURST_BYTES,
        "matmul_combine_sync_semaphore_id": matmul_combine_sync,
        "compute_only": 0,
        "local_output": 1,
        "total_tokens": tokens,
        "e_t_entry_size": L1_ALIGNMENT,
        "dram_alignment": DRAM_ALIGNMENT,
        "local_output_map_cb_id": 9,
        "replay_slice": 1 if replay_slice else 0,
    }

    shard_to_bank = [r.start.x for r in w0_w1.memory_config().shard_spec.grid.ranges()]
    if len(shard_to_bank) != num_banks:
        raise RuntimeError(f"moe_compute: the w0_w1 shard grid has {len(shard_to_bank)} banks, expected {num_banks}")
    vchannels = _ring_vchannels(ring, n_matmul)
    matmul_rt = []
    for idx, core in enumerate(ring):
        r, bank = divmod(idx, n_matmul)
        pos = ring_pos[bank]
        nxt = physical(ring[r * n_matmul + order[(pos + 1) % n_matmul]])
        prv = physical(ring[r * n_matmul + order[(pos + n_matmul - 1) % n_matmul]])
        args = [
            bank,
            vchannels[idx],
            w0_w1.buffer_address(),
            w2.buffer_address(),
            staging.buffer_address(),
            ring_sem,
            pos,
            nxt[0],
            nxt[1],
            r,
            prv[0],
            prv[1],
            *shard_to_bank,
            output.buffer_address(),
            token_lists.buffer_address(),
        ]
        matmul_rt.append((_coord(core), args))

    combine_map = ",".join(str(v) for c in place.combine for v in physical(c))
    dm1_defines = [("OUTPUT_SHARD_CORE_MAP", "{" + combine_map + "}")]

    mesh_program = ttnn.MeshProgramDescriptor()
    rows, cols = tuple(mesh.shape)
    for row in range(rows):
        for col in range(cols):
            named = dict(tilize_named, linearized_mesh_coord=row * cols + col)
            kernels = [
                _dm(
                    "tilize_reader.cpp",
                    tilize_crs,
                    tilize_ct,
                    named,
                    tilize_rt,
                    ttnn.DataMovementProcessor.RISCV_1,
                    ttnn.NOC.NOC_1,
                    dynamic=True,
                ),
                _dm(
                    "tilize_writer.cpp",
                    tilize_crs,
                    tilize_ct,
                    named,
                    tilize_rt,
                    ttnn.DataMovementProcessor.RISCV_0,
                    ttnn.NOC.NOC_1,
                    dynamic=True,
                ),
                _compute(
                    "tilize_compute.cpp",
                    tilize_crs,
                    [],
                    compute_tilize_named,
                    tilize_compute_rt,
                    fidelity=ttnn.MathFidelity.HiFi4,
                    approx=False,
                ),
                _dm(
                    "dm0.cpp",
                    ring_crs,
                    matmul_ct,
                    matmul_named,
                    matmul_rt,
                    ttnn.DataMovementProcessor.RISCV_1,
                    ttnn.NOC.NOC_0,
                ),
                _dm(
                    "dm1.cpp",
                    ring_crs,
                    matmul_ct,
                    matmul_named,
                    matmul_rt,
                    ttnn.DataMovementProcessor.RISCV_0,
                    ttnn.NOC.NOC_1,
                    defines=dm1_defines,
                ),
                _compute(
                    "compute.cpp",
                    ring_crs,
                    matmul_ct,
                    matmul_named,
                    matmul_rt,
                    fidelity=ttnn.MathFidelity.LoFi,
                    approx=True,
                ),
            ]
            coordinate = ttnn.MeshCoordinate(row, col)
            mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
                kernels=kernels, semaphores=list(sems.descriptors), cbs=list(cbs)
            )
    return mesh_program


def _cores_full(tensor) -> ttnn.CoreRangeSet:
    return tensor.memory_config().shard_spec.grid


def _uint_dtype(tensor):
    return tensor.dtype


def _ring_vchannels(ring: list[Core], ring_size: int) -> list[int]:
    """The NoC virtual channel of each ring core's writes: distinct within a row while four are free, else the least
    used one from the position's own channel on."""

    vchannels: list[int] = []
    for idx, (_, y) in enumerate(ring):
        used = 0
        for prev in range(idx):
            if ring[prev][1] == y:
                used |= 1 << vchannels[prev]
        vc = (idx % ring_size) & 3
        for _ in range(4):
            if not (used >> vc) & 1:
                break
            vc = (vc + 1) & 3
        if (used >> vc) & 1:
            count = [0] * 4
            for prev in range(idx):
                if ring[prev][1] == y:
                    count[vchannels[prev]] += 1
            vc = (idx % ring_size) & 3
            for tries in range(1, 4):
                candidate = ((idx % ring_size) + tries) & 3
                if count[candidate] < count[vc]:
                    vc = candidate
        vchannels.append(vc)
    return vchannels
