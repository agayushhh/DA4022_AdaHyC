import torch
from .compress_function_kernel import *

# ********************utils********************

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

def head_to_hidden_shape(x: torch.Tensor):
    bsz, _, seq_len, _ = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)

def update_dict(old_dict, new_dict, iteration):
    for name in old_dict.keys():
        if name == 'outlier_channel_index':
            if old_dict[name] is None:
                old_dict[name] = new_dict[name]
            else: # TODO merge logic
                old_dict[name] = new_dict[name]
        else:
            if old_dict[name] is None:
                old_dict[name] = new_dict[name]
            else:
                old_dict[name] = (iteration * old_dict[name] + new_dict[name]) / (iteration + 1)
    return old_dict


# DA4022 fixes (hyc_fixes): calibration statistics over every token of every sample instead of sample 0 only
def calib_rows(x, fixes):
    return x.reshape(-1, x.shape[-1]) if fixes else x[0]

# DA4022 fixes (hyc_fixes): released update_dict replaces the outlier index every calibration iteration but
# averages scales, so a channel that was an outlier (zeroed -> scale 1e-3) in an earlier iteration keeps a
# shrunk scale and clips for the rest of training. With fixes the index is frozen at the first calibration
# iteration and reused, so every averaged scale / zero point is computed over the same channel set.
def calib_channel(static_value, fixes):
    return static_value.get('outlier_channel_index') if fixes else None


def get_statistics_softmax(x: torch.Tensor, outlier_ratio: float):
    outlier = torch.kthvalue(x.float().flatten(), int(x.numel() * (1 - outlier_ratio))).values
    return outlier

def get_statistics_outlier(x: torch.Tensor, outlier_ratio: float):
    outlier = torch.kthvalue(x.float().abs().flatten(), int(x.numel() * (1 - outlier_ratio))).values
    return outlier


def get_statistics_structed_pruning(x: torch.Tensor, outlier_ratio: float):
    channel_norm = x.abs().norm(dim=-2)
    outlier_channel_index = torch.topk(channel_norm, int(x.shape[-1] * outlier_ratio), largest=True).indices
    return outlier_channel_index


def get_statistics_only_quant(x: torch.Tensor, q_bit: int = 8, q_method: str = 'per-tensor', fixes: bool = False):
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)

    x_sample = calib_rows(x, fixes)
    if q_method == 'per-tensor':
        scale = (x_sample.max() - x_sample.min()) / (2 ** q_bit - 1)
    elif q_method == 'per-channel':
        scale = (x_sample.max(dim=-2, keepdim=True).values - x_sample.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
    else:
        raise "Unsupport Quantize Method"
    
    del x
    return scale.to(torch.bfloat16)


def get_statistics_only_quant_zero_point(x: torch.Tensor, q_bit: int = 8, q_method: str = 'per-tensor'):
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)

    x_sample = x[0]
    if q_method == 'per-tensor':
        scale = (x_sample.max() - x_sample.min()) / (2 ** q_bit - 1)
    elif q_method == 'per-channel':
        scale = (x_sample.max(dim=-2, keepdim=True).values - x_sample.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
        zero_point = -torch.round(x_sample.min(dim=-2, keepdim=True).values / scale) - (2 ** (q_bit - 1))
    else:
        raise "Unsupport Quantize Method"
    
    del x
    return scale.to(torch.bfloat16), zero_point.to(torch.int8)


def get_statistics_channel_base(x: torch.Tensor, outlier_ratio: float, q_bit: int = 8, q_method: str = 'per-tensor', fixes: bool = False, channel=None):
    x_ = x.clone()
    if len(x_.shape) == 4:
        batch, num_head, seq_len, sep_dim = x_.shape
        x_ = x_.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)
    
    if channel is not None: # DA4022 fixes: outlier set frozen at the first calibration iteration
        outlier_channel_index = channel
    else:
        outlier_channel_index = get_statistics_structed_pruning(calib_rows(x_, fixes), outlier_ratio)

    x_outlier = x_[:, :, outlier_channel_index]
    x_outlier = x_outlier.to(torch.bfloat16)
    x_[:, :, outlier_channel_index] = 0

    x_sub_outlier = calib_rows(x_, fixes)
    if q_method == 'per-tensor':
        scale = (x_sub_outlier.max() - x_sub_outlier.min()) / (2 ** q_bit - 1)
    elif q_method == 'per-channel':
        scale = (x_sub_outlier.max(dim=-2, keepdim=True).values - x_sub_outlier.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
    else:
        raise "Unsupport Quantize Method"
    
    del x_
    
    # the scale factor should not be zero(but since the corresponding channel is removed, there's no impact on final result)
    scale += (scale == 0) * 1e-3
    return outlier_channel_index, scale.to(torch.bfloat16)


# ********************compress/decompress interface********************

# outlier subtraction + quantization
def outlier_subtraction_fuse_compression_quantization(x, s, channel, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)

    # decide dtype
    x, s = x.to(dtype), s.to(dtype)
    B, M, N = x.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    
    # remove outlier channels
    x_outlier = x[:, :, channel]
    x[:, :, channel] = 0
    
    # quantize the rest part
    compression_quantization_kernel[grid](
        x, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        quantize_bit, elem_per_position
    )

    del x
    return x_outlier, q


# outlier addition + dequantization
def outlier_addition_fuse_decompression_dequantization(q, s, x_outlier, channel, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    B, M, _ = q.shape
    N = s.shape[-1]

    # dtype
    s = s.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=q.device, dtype=torch.bfloat16)

    decompression_dequantization_kernel[grid](
        x, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        quantize_bit, elem_per_position,
    )
    x[:, :, channel] = x_outlier
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


# quantization (only with scale)
def compression_quantization(x, s, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)

    # decide dtype
    x, s = x.to(dtype), s.to(dtype)
    B, M, N = x.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    
    # ************************** triton version **************************
    # quantize the rest part
    compression_quantization_kernel[grid](
        x, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        quantize_bit, elem_per_position
    )
    
    # ************************** torch version **************************
    # q = torch.clamp(torch.round(x / s), -2 ** (quantize_bit - 1), 2 ** (quantize_bit - 1) - 1).to(torch.int8)
    
    del x
    return q


# dequantization (only with scale)
def decompression_dequantization(q, s, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    B, M, _ = q.shape
    N = s.shape[-1]

    # dtype
    s = s.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=q.device, dtype=torch.bfloat16)

    # ************************** triton version **************************
    decompression_dequantization_kernel[grid](
        x, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        quantize_bit, elem_per_position,
    )
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


# quantization (with scale and zero point)
def compression_quantization_with_zero_point(x, s, z, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)

    # decide dtype
    x, s, z = x.to(dtype), s.to(dtype), z.to(dtype)
    B, M, N = x.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    
    # quantize the rest part
    compression_quantization_with_zero_point_kernel[grid](
        x, q, s, z, 
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        z.stride(1),
        quantize_bit, elem_per_position
    )

    del x
    return q


# dequantization (with scale and zero point)
def decompression_dequantization_with_zero_point(q, s, z, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    B, M, _ = q.shape
    N = s.shape[-1]

    # dtype
    s = s.to(dtype)
    z = z.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=q.device, dtype=torch.bfloat16)

    decompression_dequantization_with_zero_point_kernel[grid](
        x, q, s, z,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        z.stride(1),
        quantize_bit, elem_per_position,
    )
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x

# compress softmax's attention map
def compression_softmax(x: torch.Tensor, outlier: float):
    mask = (x > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier.flatten().to_sparse()
    return x_outlier_sparse

# decompress softmax's attention map
def decompression_softmax(x_sparse: torch.Tensor):
    return x_sparse.to_dense()

# unstructed compression(use COO format)
def compression_unstructed_pruning_coo(x: torch.Tensor, outlier: float):
    mask = (x.abs() > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier.to_sparse()
    return x_outlier_sparse

# unstructed decompress(use COO format)
def decompression_unstructed_pruning_coo(x_sparse: torch.Tensor):
    return x_sparse.to_dense()

# unstructed compression(use bitmap format)
def compression_unstructed_pruning_bitmap(x: torch.Tensor, outlier: float):
    mask = (x.abs() > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier[mask]
    return x_outlier_sparse, mask

# unstructed decompress(use bitmap format)
def decompression_unstructed_pruning_bitmap(x_sparse: torch.Tensor, mask: torch.Tensor):
    x = torch.empty_like(mask, device=x_sparse.device, dtype=x_sparse.dtype)
    x[mask] = x_sparse
    return x


# ********************pack "get statistics" and "compress"********************
# type: outlier channel extraction + quantization
def compression_pack_channel_base(x, o_ratio, q_bit, q_method, it_num, it_num_thd, static_value, fixes=False):
    if it_num < it_num_thd:
        o_channel_idx, scale = get_statistics_channel_base(x, o_ratio, q_bit, q_method, fixes, calib_channel(static_value, fixes))
    else:
        o_channel_idx, scale = static_value['outlier_channel_index'], static_value['scale']
    o, q = outlier_subtraction_fuse_compression_quantization(x, scale, o_channel_idx, q_bit)
    return o, q, o_channel_idx, scale

# type: quantization
def compression_pack_quant_base(x, q_bit, q_method, it_num, it_num_thd, static_value, fixes=False):
    if it_num < it_num_thd:
        scale = get_statistics_only_quant(x, q_bit, q_method, fixes)
    else:
        scale = static_value['scale']
    q = compression_quantization(x, scale, q_bit)
    return q, scale

# type: quantization with zero point
def compression_pack_quant_zp_base(x, q_bit, q_method, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        scale, zero_point = get_statistics_only_quant_zero_point(x, q_bit, q_method)
    else:
        scale, zero_point = static_value['scale'], static_value['zero_point']
    q = compression_quantization_with_zero_point(x, scale, zero_point, q_bit)
    return q, scale, zero_point

# type: softmax attention map compression
def compression_pack_softmax_base(x, o_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        outlier = get_statistics_softmax(x, o_ratio)
    else:
        outlier = static_value['outlier']
    o = compression_softmax(x, outlier)
    return o, outlier

# type: unstructed pruning(using COO format)
def compression_pack_unstructed_pruning_coo(x, o_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        outlier = get_statistics_outlier(x, o_ratio) # a funny reuse
    else:
        outlier = static_value['outlier']
    x_outlier = compression_unstructed_pruning_coo(x, outlier)
    return x_outlier, outlier

# type: unstructed pruning(using bitmap format)
def compression_pack_unstructed_pruning_bitmap(x, o_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        outlier = get_statistics_outlier(x, o_ratio) # a funny reuse
    else:
        outlier = static_value['outlier']
    x_outlier, mask = compression_unstructed_pruning_bitmap(x, outlier)
    return x_outlier, outlier, mask
    

# ********************DA4022: optional zero-point dispatch********************
# Fresh helpers instead of reusing get_statistics_only_quant_zero_point: that one stores the zero point
# as int8, which overflows whenever a channel does not straddle 0 (z = -round(min/s) - 2^(b-1)).
# Zero-point kernels run with fp32 s/z/x, since |x/s + z| can exceed bf16's exact-integer range.
def get_statistics_zero_point(x: torch.Tensor, q_bit: int, fixes: bool = False):
    x_sample = calib_rows(x, fixes)
    x_min, x_max = x_sample.amin(dim=-2, keepdim=True).float(), x_sample.amax(dim=-2, keepdim=True).float()
    scale = (x_max - x_min) / (2 ** q_bit - 1)
    scale += (scale == 0) * 1e-3
    zero_point = -torch.round(x_min / scale) - 2 ** (q_bit - 1)
    return scale, zero_point

# returns (q, scale, zero_point); zero_point is None on the released symmetric path
def compression_pack_quant(x, q_bit, q_method, it_num, it_num_thd, static_value, zero_point=False, fixes=False):
    if not zero_point:
        return (*compression_pack_quant_base(x, q_bit, q_method, it_num, it_num_thd, static_value, fixes), None)
    if it_num < it_num_thd:
        scale, z = get_statistics_zero_point(x, q_bit, fixes)
    else:
        scale, z = static_value['scale'], static_value['zero_point']
    q = compression_quantization_with_zero_point(x, scale, z, q_bit, dtype=torch.float32)
    return q, scale, z

# returns (outlier, q, channel_idx, scale, zero_point); x is modified in place (pass a copy)
def compression_pack_channel(x, o_ratio, q_bit, q_method, it_num, it_num_thd, static_value, zero_point=False, fixes=False):
    if not zero_point:
        return (*compression_pack_channel_base(x, o_ratio, q_bit, q_method, it_num, it_num_thd, static_value, fixes), None)
    if it_num >= it_num_thd or calib_channel(static_value, fixes) is not None:
        o_channel_idx = static_value['outlier_channel_index']
    else:
        o_channel_idx = get_statistics_structed_pruning(calib_rows(x, fixes), o_ratio)
    x_outlier = x[:, :, o_channel_idx].to(torch.bfloat16)
    x[:, :, o_channel_idx] = 0
    q, scale, z = compression_pack_quant(x, q_bit, q_method, it_num, it_num_thd, static_value, zero_point=True, fixes=fixes)
    return x_outlier, q, o_channel_idx, scale, z

# inverse of the two packers above (pass x_outlier/channel only for the outlier-channel path)
def decompression_dequant(q, s, z, q_bit, x_outlier=None, channel=None):
    if z is None:
        x = decompression_dequantization(q, s, q_bit)
    else:
        x = decompression_dequantization_with_zero_point(q, s, z, q_bit, dtype=torch.float32)
    if channel is not None:
        x[:, :, channel] = x_outlier
    return x
