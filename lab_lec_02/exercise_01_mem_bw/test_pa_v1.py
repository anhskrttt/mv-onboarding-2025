# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import random
from typing import List, Optional, Tuple, Union
import itertools
import torch
import pytest
from aiter.test_common import checkAllclose, perftest, tensor_dump, tensor_load
from aiter import pertoken_quant
from aiter import dtypes
from enum import Enum
from einops import rearrange
import argparse

uniform_range = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": dtypes.bf16,
    "float": dtypes.fp32,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    x = 16 // torch_dtype.itemsize
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


FLOAT32_BYTES = torch.finfo(dtypes.fp32).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = 65536
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 32768  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {dtypes.fp16, dtypes.bf16}
DTYPES = [torch.half, dtypes.bf16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 120, 128, 192, 256]

BLOCK_SIZES = [16, 32]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8"]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

# 0: no quant. 1: (ignore this), FP8, 2: K/V per-token(prefer this)
PA_QUANT = 2


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    if 0 < logits_soft_cap:
        attn_weights = logits_soft_cap * torch.tanh(attn_weights / logits_soft_cap)
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def pertoken_quant_kvcache_symm(
    # [num_blocks, num_heads, head_size // x, block_size, x]
    key_cache: torch.Tensor,
    # [num_blocks, num_heads, head_size, block_size]
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,  # e.g. dtypes.fp8
    scale_dtype: torch.dtype = dtypes.fp32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = key_cache.shape[0]
    num_heads = key_cache.shape[1]
    head_dim = value_cache.shape[2]
    block_size = value_cache.shape[3]
    # x          = key_cache.shape[4]
    total_tokens = num_blocks * block_size

    # print(f"{key_cache.shape=}{key_cache.stride()=}")
    # print(f"{value_cache.shape=}{value_cache.stride()=}")

    key_cache_permute = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    value_cache_permute = (
        value_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    k_quant, k_scale = pertoken_quant(key_cache_permute, quant_dtype=quant_dtype)
    v_quant, v_scale = pertoken_quant(value_cache_permute, quant_dtype=quant_dtype)

    # NOTE: quant_x and original x could be different
    quant_x = 16 // quant_dtype.itemsize

    k_quant = (
        k_quant.view(num_blocks, num_heads, block_size, head_dim // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    k_scale = k_scale.permute(1, 0, 2, 3).view(num_heads, total_tokens).contiguous()
    v_quant = (
        v_quant.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    v_scale = v_scale.permute(1, 0, 2, 3).view(num_heads, total_tokens).contiguous()

    # print(f"{k_quant.shape=}{k_quant.stride()=}")
    # print(f"{k_scale.shape=}{k_scale.stride()=}")
    # print(f"{v_quant.shape=}{v_quant.stride()=}")
    # print(f"{v_scale.shape=}{v_scale.stride()=}")
    # print(f"key_cache_permute:{key_cache_permute[0, :, :, :]}, k_quant:{k_quant[0, :, :, :, :]}, k_scale:{k_scale[:, 0]}")

    return k_quant, k_scale, v_quant, v_scale


# @perftest()


def run_torch(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    logits_soft_cap,
    k_scale,
    v_scale,
    num_queries_per_kv,
):
    output = torch.zeros_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: List[torch.Tensor] = []
        values_lst: List[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, block_offset, :]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias, logits_soft_cap)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output, 1


@perftest()
def run_aiter(
    query,
    key_cache,
    value_cache,
    block_tables,
    cu_query_lens,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    kv_cache_layout,
    scale,
    alibi_slopes,
    logits_soft_cap,
    k_scale,
    v_scale,
    mtp=1,
):
    # copied from ops.PagedAttention.forward_decode()
    _PARTITION_SIZE_ROCM = 256
    fp8_out_scale = None

    num_seqs, num_heads, head_size = query.shape
    block_size = key_cache.shape[2 if kv_cache_layout == "HND" else 1]

    output = torch.empty_like(query)
    max_num_partitions = (
        max_seq_len + _PARTITION_SIZE_ROCM - 1
    ) // _PARTITION_SIZE_ROCM
    assert _PARTITION_SIZE_ROCM % block_size == 0

    # will use single workspace buffer to accommodate following 3 intermediate tensors:
    #   1. tmp_output (shape=(num_seqs, num_heads, max_num_partitions, head_size), dtype=output.dtype)
    #   2. exp_sums (shape=(num_seqs, num_heads, max_num_partitions), dtype=float32)
    #   3. max_logits (shape=(num_seqs, num_heads, max_num_partitions), dtype=float32)
    nbyes_per_qo_elem = torch.finfo(output.dtype).bits // 8
    workspace_buffer = torch.empty(
        (num_seqs * mtp * num_heads * max_num_partitions * head_size)
        * nbyes_per_qo_elem
        + 2 * (num_seqs * mtp * num_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=output.device,
    )

    cpa_fp8_out = False
    if fp8_out_scale is not None:
        output = torch.empty_like(output, dtype=dtypes.fp8)
        cpa_fp8_out = True
    torch.ops.aiter.paged_attention_v1(
        output,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        cu_query_lens,
        seq_lens,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale if cpa_fp8_out else None,
        _PARTITION_SIZE_ROCM,
    )
    if cpa_fp8_out:
        return output.view(num_seqs, num_heads * head_size)
    else:
        return output


def dump_input(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
):
    tensor_dump(query, "Q")
    # qbk = tensor_load('Q.bin')
    # checkAllclose(query, qbk)
    tensor_dump(key_cache, "K_cache")
    tensor_dump(value_cache, "V_cache")
    tensor_dump(block_tables, "block_tables")
    tensor_dump(seq_lens, "seq_lens")


def load_input():
    # return (tensor_load('Q.bin'),
    #         tensor_load('K_cache.bin'),
    #         tensor_load('V_cache.bin'),
    #         tensor_load('block_tables.bin'),
    #         tensor_load('seq_lens.bin'),
    #         tensor_load('out_aiter.bin'))
    # return (tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/Q_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/K_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/V_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/block_tables.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/seq_lens.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/OUT_16.bin'),
    #         )
    return (
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/Q_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/K_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/V_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/block_tables.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/seq_lens.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/OUT_BF16.bin"),
    )


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/X, head_size, X]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC


class InputSource(Enum):
    PreGen = 1
    Random = 2


class PAVariant(Enum):
    Shomy = 1
    Asm = 2
    Naive = 3


INPUT_SOURCE = InputSource.Random
DUMP_INPUTS = False  # whether to dump inputs
DUMP_OUTPUT = False  # whether to dump output


@pytest.mark.parametrize("ctx_lens", [1, 26, 128, 4097])
@pytest.mark.parametrize("num_seqs", [1, 3, 31, 128])
@pytest.mark.parametrize("num_heads", [(8, 1), (4, 2), (32, 4)])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("use_alibi", [False, True])
@pytest.mark.parametrize("block_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize("kv_cache_layout", ["NHD", "HND"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("pa_variant", [PAVariant.Shomy])
@pytest.mark.parametrize("quant_cache_dtype", [None, dtypes.fp8, dtypes.i8])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_paged_attention(
    ctx_lens: int,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    pa_variant: PAVariant,
    quant_cache_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    if pa_variant == PAVariant.Shomy:
        if quant_cache_dtype is not None:
            pytest.skip()
    elif pa_variant == PAVariant.Asm:
        if (
            use_alibi
            or head_size != 128
            or block_size != 16
            or dtype is not dtypes.bf16
            or quant_cache_dtype not in [None, dtypes.i8]
        ):
            pytest.skip()
    elif pa_variant == PAVariant.Naive:
        if use_alibi:
            pytest.skip()

    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_default_device(device)

    # Using default kv_scale
    k_scale = v_scale = torch.tensor([1.0], dtype=dtypes.fp32)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=dtypes.fp32)
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    max_seq_len = ctx_lens
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs
    print(f"{INPUT_SOURCE=}")

    # prepare inputs & golden output
    if INPUT_SOURCE == InputSource.PreGen:
        (query, key_cache, value_cache, block_tables, seq_lens, out_golden) = (
            load_input()
        )
    else:
        query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
        query.uniform_(*uniform_range)

        # Create the KV caches.
        key_caches, value_caches = kv_cache_factory(
            num_blocks,
            block_size,
            1,
            num_kv_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            seed,
            device,
        )

        key_cache, value_cache = key_caches[0], value_caches[0]

        # Create the block tables.
        block_tables = rearrange(
            torch.randperm(num_blocks, dtype=dtypes.i32, device=device),
            "(b nblocks) -> b nblocks",
            b=num_seqs,
        )

        # seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
        seq_lens = torch.full(size=(num_seqs,), fill_value=ctx_lens, dtype=torch.int)

        key_cache_new = rearrange(key_cache, "b h d1 s d2 -> b h s (d1 d2)")
        value_cache_new = rearrange(value_cache, "b h d s -> b h s d")
        out_golden, _ = run_torch(
            query,
            key_cache_new,
            value_cache_new,
            block_tables,
            seq_lens,
            max_seq_len,
            kv_cache_dtype,
            num_kv_heads,
            scale,
            alibi_slopes,
            logits_soft_cap,
            k_scale,
            v_scale,
            num_queries_per_kv,
        )
        cu_query_lens = torch.arange(0, num_seqs + 1, dtype=torch.int)

    if quant_cache_dtype is None:
        if kv_cache_layout == "NHD":
            key_cache_new = rearrange(key_cache_new, "b h s d -> b s h d")
            value_cache_new = rearrange(value_cache_new, "b h s d -> b s h d")

        out_aiter, time_aiter = run_aiter(
            query,
            key_cache_new.contiguous(),
            value_cache_new.contiguous(),
            block_tables,
            cu_query_lens,
            seq_lens,
            max_seq_len,
            kv_cache_dtype,
            kv_cache_layout,
            scale,
            alibi_slopes,
            logits_soft_cap,
            k_scale,
            v_scale,
        )
        assert (
            checkAllclose(out_golden, out_aiter, msg=f"golden vs aiter:{time_aiter}")
            < 0.01
        )

    if DUMP_INPUTS:
        dump_input(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            kv_cache_dtype,
            num_kv_heads,
            scale,
            alibi_slopes,
            k_scale,
            v_scale,
        )

    # atol, rtol = 1e-2, 1e-2
    # msg = f"[perf] dim: {str((num_seqs, num_heads, head_size)):<20}, dtype: {dtype}, {time_native=:<8.2f} us, {time_aiter=:<8.2f} us, uplift: {time_native/time_aiter-1:<5.1%}"
    # assert checkAllclose(out_native, out_aiter, atol=atol, rtol=rtol, msg=msg)
    # print(
    #     f"[test] dim: {str((ctx_lens, num_seqs, num_heads, head_size)):<20}, dtype: {dtype}, finished)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test Paged Attention V1",
    )
    parser.add_argument(
        "-c",
        "--ctx_len",
        type=int,
        default=[2048],
        nargs="*",
        help="""Context length.
    e.g. -c 2048""",
    )
    parser.add_argument(
        "-p",
        "--pa_variant",
        type=str,
        choices=[member.name for member in PAVariant],
        default=[PAVariant.Shomy],
        nargs="*",
        help=f"Paged Attention variant to test. {[member.name for member in PAVariant]}\n"
        + "    e.g. -p Shomy\n",
    )
    parser.add_argument(
        "-q",
        "--quant_cache_dtype",
        type=str,
        choices=["none"],
        default=["none"],
        nargs="*",
        help="""Quantization cache dtype.
    e.g. -q none""",
    )

    torch.set_printoptions(sci_mode=False)
    args = parser.parse_args()
    if not args.pa_variant == [PAVariant.Shomy]:
        args.pa_variant = [PAVariant[variant] for variant in args.pa_variant]
    args.quant_cache_dtype = [
        None if i == "none" else dtypes.d_dtypes[i] for i in args.quant_cache_dtype
    ]

    for ctx_len, pa_variant, quant_cache_dtype in itertools.product(
        args.ctx_len,
        args.pa_variant,
        args.quant_cache_dtype,
    ):

        if pa_variant == PAVariant.Shomy:
            if quant_cache_dtype is not None:
                continue

        test_paged_attention(
            ctx_len,
            8,
            (8, 1),
            128,
            False,
            16,
            dtypes.fp16,
            "auto",
            "NHD",
            0.0,
            pa_variant,
            quant_cache_dtype,
            0,
            "cuda:0",
        )