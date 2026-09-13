import torch
import triton
import triton.language as tl


@triton.jit
def tanh(x):
    # tanh(x) = 2 * sigmoid(2x) - 1
    return 2 * tl.sigmoid(2.0 * x) - 1.0

@triton.jit
def fused_spatial_softmax_kernel(
    x_ptr,           # 输入指针 (bfloat16)
    out_ptr,         # 输出指针 (float32 或 bfloat16)
    H, W,            # 动态高宽
    T,               # Temperature
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offset = pid * (H * W)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H * W)

    x_bf16 = tl.load(x_ptr + row_offset + offsets, mask=mask, other=-float('inf'))
    val = x_bf16.to(tl.float32)

    
    # Safe Softmax
    val = val / T
    max_val = tl.max(val, axis=0)
    numerator = tl.exp(val - max_val)

    # 坐标生成 (必须是 FP32)
    y_idx = offsets // W
    x_idx = offsets % W
    coord_x = -1.0 + (x_idx * 2.0) / (W - 1.0)
    coord_y = -1.0 + (y_idx * 2.0) / (H - 1.0)

    # 融合期望计算
    sum_numerator = tl.sum(numerator, axis=0) 
    sum_num_x = tl.sum(numerator * coord_x, axis=0)
    sum_num_y = tl.sum(numerator * coord_y, axis=0)

    expected_x = sum_num_x / sum_numerator
    expected_y = sum_num_y / sum_numerator

    tl.store(out_ptr + pid * 2, expected_x)
    tl.store(out_ptr + pid * 2 + 1, expected_y)


@triton.jit
def geglu_kernel(
    input_ptr,          # [M, 2*C]
    output_ptr,         # [M, C]
    stride_input_m, stride_input_c,
    stride_output_m, stride_output_c,
    C,                  # Hidden Dimension
    BLOCK_SIZE: tl.constexpr
):
    # 行索引
    row_idx = tl.program_id(0)
    
    # 基础偏移量
    input_row_start = input_ptr + row_idx * stride_input_m
    output_row_start = output_ptr + row_idx * stride_output_m
    
    # 循环加载处理 (Chunking)
    # 我们的目标是 Output 的维度 C
    # Input 分为两部分: x [0..C] 和 gate [C..2C]
    
    # 这里为了简单，假设 C 是 BLOCK_SIZE 的倍数或者一次处理完
    # 实际应用中需要循环处理 offset
    for off in range(0, C, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < C
        
        # 加载 x (前半部分)
        x_ptr_curr = input_row_start + cols * stride_input_c
        x_val = tl.load(x_ptr_curr, mask=mask, other=0.0)
        
        # 加载 gate (后半部分, 偏移 C)
        gate_ptr_curr = input_row_start + (C + cols) * stride_input_c
        gate_val = tl.load(gate_ptr_curr, mask=mask, other=0.0)
        
        # 优化: 避免多次计算，直接写公式
        tanh_val = tanh(0.7978845608 * (gate_val + 0.044715 * gate_val * gate_val * gate_val))
        gelu_gate = 0.5 * gate_val * (1.0 + tanh_val)
        
        # GEGLU = x * GELU(gate)
        out_val = x_val * gelu_gate
        
        # Store
        output_ptr_curr = output_row_start + cols * stride_output_c
        tl.store(output_ptr_curr, out_val, mask=mask)

def triton_geglu(x):
    """
    Input: x of shape [..., 2*C]
    Output: y of shape [..., C] = x_a * GELU(x_b)
    """
    # 1. Flatten to [M, 2*C]
    input_shape = x.shape
    dim_last = input_shape[-1]
    assert dim_last % 2 == 0, "Last dimension must be even for GEGLU"
    C = dim_last // 2
    
    x_flatten = x.view(-1, dim_last)
    M = x_flatten.shape[0]
    
    # 2. Allocate Output
    out_flatten = torch.empty((M, C), device=x.device, dtype=x.dtype)
    
    # 3. Kernel Launch
    grid = (M, )
    BLOCK_SIZE = 1024 # 可调 
    # 如果 C > 1024, 内核中有循环处理
    
    geglu_kernel[grid](
        x_flatten,
        out_flatten,
        x_flatten.stride(0), x_flatten.stride(1),
        out_flatten.stride(0), out_flatten.stride(1),
        C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 4. Reshape back
    output_shape = list(input_shape)
    output_shape[-1] = C
    return out_flatten.view(*output_shape)


def fast_spatial_softmax(x, temperature=1.0):
    """
    x: [B, C, H, W]
    """
    B, C, H, W = x.shape
    N = H * W
    
    # Flatten
    x = x.view(-1, N)
    if not x.is_contiguous():
        x = x.contiguous()
        
    out = torch.empty((B * C, 2), device=x.device, dtype=torch.float32)
    
    # 确定 Block Size (向上取整到 2 的幂)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # 硬件限制保护
    if BLOCK_SIZE > 8192:
        raise ValueError(f"Feature map too large (H*W={N}) for this optimized kernel.")

    grid = (B * C, )
    
    fused_spatial_softmax_kernel[grid](
        x, out,
        H, W, 
        temperature,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.view(B, C, 2)