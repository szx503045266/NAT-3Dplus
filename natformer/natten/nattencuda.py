"""
Neighborhood Attention PyTorch Module (CUDA only)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils.cpp_extension import load, is_ninja_available
import warnings
import os


if is_ninja_available():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    nattenav_cuda = load(
        'nattenav_cuda', [f'{this_dir}/src/nattenav_cuda.cpp', f'{this_dir}/src/nattenav_cuda_kernel.cu'], verbose=False)
    nattenqkrpb_cuda = load(
        'nattenqkrpb_cuda', [f'{this_dir}/src/nattenqkrpb_cuda.cpp', f'{this_dir}/src/nattenqkrpb_cuda_kernel.cu'], verbose=False)
else:
    warnings.warn("Ninja is not installed, looking up extensions manually.")
    try:
        import nattenav_cuda
        import nattenqkrpb_cuda
    except:
        raise RuntimeError("Could not load NATTEN CUDA extension. " +
                           "Please make sure your device has CUDA, the CUDA toolkit for PyTorch is installed, and that you've compiled NATTEN correctly.")


class NATTENAVFunction(Function):
    """
    AV autograd function
    Computes neighborhood attention outputs given attention weights, and values.
    This calls the `AV` kernel.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value):
        attn = attn.contiguous()
        value = value.contiguous()
        out = nattenav_cuda.forward(
                attn,
                value)
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenav_cuda.backward(
            grad_out.contiguous(), ctx.saved_variables[0], ctx.saved_variables[1])
        d_attn, d_value = outputs
        return d_attn, d_value, None


class NATTENQKRPBFunction(Function):
    """
    QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = nattenqkrpb_cuda.forward(
                query,
                key,
                rpb)
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = nattenqkrpb_cuda.backward(
            grad_out.contiguous(), ctx.saved_variables[0], ctx.saved_variables[1])
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None


class NeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention Module
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_tct_shift(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation with Shift, n_seg = {}".format(n_seg))

    def shift(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        if step < 0:
            # Shift Back
            out = torch.zeros_like(kv) + 1e-20
            out[:, :step,:,:,:,:] = kv[:, -1*step:,:,:,:,:]
        else:
            # Shift Forward
            out = torch.zeros_like(kv) + 1e-20
            out[:, step:,:,:,:,:] = kv[:, :-1*step,:,:,:,:]
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)

        #Back Correlation Attention
        k1 = self.shift(k, -1 * self.step)
        v1 = self.shift(v, -1 * self.step)
        attn = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = NATTENAVFunction.apply(attn, v1)

        #Front Correlation Attention
        k2 = self.shift(k, self.step)
        v2 = self.shift(v, self.step)
        attn = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x2 = NATTENAVFunction.apply(attn, v2)

        # Simple Addition
        x = (x + x1 + x2) / 3.0

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_tct_shift_pad(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation with Shift_pad, n_seg = {}".format(n_seg))
    
    def shift_pad(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        if step < 0:
            # Shift Back
            out = kv.clone()
            out[:, :step,:,:,:,:] = kv[:, -1*step:,:,:,:,:]
        else:
            # Shift Forward
            out = kv.clone()
            out[:, step:,:,:,:,:] = kv[:, :-1*step,:,:,:,:]
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)

        #Back Correlation Attention
        k1 = self.shift_pad(k, -1 * self.step)
        v1 = self.shift_pad(v, -1 * self.step)
        attn = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = NATTENAVFunction.apply(attn, v1)

        #Front Correlation Attention
        k2 = self.shift_pad(k, self.step)
        v2 = self.shift_pad(v, self.step)
        attn = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x2 = NATTENAVFunction.apply(attn, v2)

        # Simple Addition
        x = (x + x1 + x2) / 3.0

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_tct_roll(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out


    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)

        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = NATTENAVFunction.apply(attn, v1)

        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x2 = NATTENAVFunction.apply(attn, v2)

        # Simple Addition
        x = (x + x1 + x2) / 3.0

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_3D(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)

        #print("attn shape {}".format(attn0.shape)) # B,heads,H, W, dim
        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 3, _D)
        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)

        # Simple Addition
        x = x + x1 + x2

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_Dilated3D(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, dilated_step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.dilated_step = dilated_step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation Dilated3D with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def dilated_split(self, qkv):
        b, heads, h, w, dim = qkv.shape
        ds = self.dilated_step + 1
        assert h % ds==0 and w % ds==0, "dilated_step+1 {} has to be dividable by H {} or W {}".format(ds, h, w)
        qkv = qkv.reshape(b, heads, h//ds, ds, w//ds, ds, dim)
        qkv = qkv.permute(0, 3, 5, 1, 2, 4, 6) #-> b, ds, ds, heads, h/ds, w/ds, dim
        qkv = qkv.reshape(b*ds*ds, heads, h//ds, w//ds, dim)
        return qkv

    def dilated_restore(self, qkv):
        # restore qkv changed by dilated split
        ds = self.dilated_step + 1
        bds, heads, h, w, dim = qkv.shape
        b = bds // (ds * ds) 
        qkv = qkv.reshape(b, ds, ds, heads, h, w, dim)
        qkv = qkv.permute(0, 3, 4, 1, 5, 2, 6) #-> b, heads, h, ds, w, ds, dim
        qkv = qkv.reshape(b, heads, ds*h, ds*w, dim)
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        q_d = self.dilated_split(q)
        k1_d = self.dilated_split(k1)
        attn1 = NATTENQKRPBFunction.apply(q_d, k1_d, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        k2_d = self.dilated_split(k2)
        attn2 = NATTENQKRPBFunction.apply(q_d, k2_d, self.rpb)

        attn1 = self.dilated_restore(attn1)
        attn2 = self.dilated_restore(attn2)
        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 3, _D)
        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x = NATTENAVFunction.apply(attn, v)
        attn1 = self.dilated_split(attn1)
        attn2 = self.dilated_split(attn2)
        v1 = self.dilated_split(v1)
        v2 = self.dilated_split(v2)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        x1 = self.dilated_restore(x1)
        x2 = self.dilated_restore(x2)

        # Simple Addition
        x = x + x1 + x2

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_Dilated3D_hybrid(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, dilated_step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.dilated_step = dilated_step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation Dilated3D hybrid with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def dilated_split(self, qkv):
        b, heads, h, w, dim = qkv.shape
        ds = self.dilated_step + 1
        assert h % ds==0 and w % ds==0, "dilated_step+1 {} has to be dividable by H {} or W {}".format(ds, h, w)
        qkv = qkv.reshape(b, heads, h//ds, ds, w//ds, ds, dim)
        qkv = qkv.permute(0, 3, 5, 1, 2, 4, 6) #-> b, ds, ds, heads, h/ds, w/ds, dim
        qkv = qkv.reshape(b*ds*ds, heads, h//ds, w//ds, dim)
        return qkv

    def dilated_restore(self, qkv):
        # restore qkv changed by dilated split
        ds = self.dilated_step + 1
        bds, heads, h, w, dim = qkv.shape
        b = bds // (ds * ds) 
        qkv = qkv.reshape(b, ds, ds, heads, h, w, dim)
        qkv = qkv.permute(0, 3, 4, 1, 5, 2, 6) #-> b, heads, h, ds, w, ds, dim
        qkv = qkv.reshape(b, heads, ds*h, ds*w, dim)
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        num_heads_3d = self.num_heads // 2
        #num_heads_D3d = self.num_heads - num_heads_3d

        q_3d = q[:,:num_heads_3d,:,:,:]
        q_D3d = q[:,num_heads_3d:,:,:,:]
        k_3d = k[:,:num_heads_3d,:,:,:]
        k_D3d = k[:,num_heads_3d:,:,:,:]
        v_3d = v[:,:num_heads_3d,:,:,:]
        v_D3d = v[:,num_heads_3d:,:,:,:]
        rpb_3d = self.rpb[:num_heads_3d,:,:]
        rpb_D3d = self.rpb[num_heads_3d:,:,:]

        '''NAT-3D'''
        #Spatial Neighbour Attention
        attn_3d = NATTENQKRPBFunction.apply(q_3d, k_3d, rpb_3d)
        #Back Correlation Attention
        k1_3d = self.roll(k_3d, -1 * self.step)
        v1_3d = self.roll(v_3d, -1 * self.step)
        attn1_3d = NATTENQKRPBFunction.apply(q_3d, k1_3d, rpb_3d)
        #Front Correlation Attention
        k2_3d = self.roll(k_3d, self.step)
        v2_3d = self.roll(v_3d, self.step)
        attn2_3d = NATTENQKRPBFunction.apply(q_3d, k2_3d, rpb_3d)

        _b, _h, _H, _W, _D = attn_3d.shape
        attn_3d = torch.cat((attn_3d, attn1_3d, attn2_3d), dim=-1)
        attn_3d = attn_3d.softmax(dim=-1)
        attn_3d = self.attn_drop(attn_3d)
        attn_3d = attn_3d.reshape(_b, _h, _H, _W, 3, _D)
        attn1_3d = attn_3d[:,:,:,:,1,:].squeeze(4)
        attn2_3d = attn_3d[:,:,:,:,2,:].squeeze(4)
        attn_3d = attn_3d[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x_3d = NATTENAVFunction.apply(attn_3d, v_3d)
        x1_3d = NATTENAVFunction.apply(attn1_3d, v1_3d)
        x2_3d = NATTENAVFunction.apply(attn2_3d, v2_3d)

        # Simple Addition
        x_3d = x_3d + x1_3d + x2_3d

        '''NAT-Dilated3D'''
        #Spatial Neighbour Attention
        attn_D3d = NATTENQKRPBFunction.apply(q_D3d, k_D3d, rpb_D3d)
        #Back Correlation Attention
        k1_D3d = self.roll(k_D3d, -1 * self.step)
        v1_D3d = self.roll(v_D3d, -1 * self.step)
        q_d_D3d = self.dilated_split(q_D3d)
        k1_d_D3d = self.dilated_split(k1_D3d)
        attn1_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k1_d_D3d, rpb_D3d)
        #Front Correlation Attention
        k2_D3d = self.roll(k_D3d, self.step)
        v2_D3d = self.roll(v_D3d, self.step)
        k2_d_D3d = self.dilated_split(k2_D3d)
        attn2_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k2_d_D3d, rpb_D3d)

        attn1_D3d = self.dilated_restore(attn1_D3d)
        attn2_D3d = self.dilated_restore(attn2_D3d)
        _b, _h, _H, _W, _D = attn_D3d.shape
        attn_D3d = torch.cat((attn_D3d, attn1_D3d, attn2_D3d), dim=-1)
        attn_D3d = attn_D3d.softmax(dim=-1)
        attn_D3d = self.attn_drop(attn_D3d)
        attn_D3d = attn_D3d.reshape(_b, _h, _H, _W, 3, _D)
        attn1_D3d = attn_D3d[:,:,:,:,1,:].squeeze(4)
        attn2_D3d = attn_D3d[:,:,:,:,2,:].squeeze(4)
        attn_D3d = attn_D3d[:,:,:,:,0,:].squeeze(4)

        x_D3d = NATTENAVFunction.apply(attn_D3d, v_D3d)
        attn1_D3d = self.dilated_split(attn1_D3d)
        attn2_D3d = self.dilated_split(attn2_D3d)
        v1_D3d = self.dilated_split(v1_D3d)
        v2_D3d = self.dilated_split(v2_D3d)
        x1_D3d = NATTENAVFunction.apply(attn1_D3d, v1_D3d)
        x2_D3d = NATTENAVFunction.apply(attn2_D3d, v2_D3d)
        x1_D3d = self.dilated_restore(x1_D3d)
        x2_D3d = self.dilated_restore(x2_D3d)

        # Simple Addition
        x_D3d = x_D3d + x1_D3d + x2_D3d

        '''Combining 3D & NAT-Dilated3D'''
        x = torch.concat((x_3d, x_D3d), dim=1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        #attn_gt = NATTENQKRPBFunction.apply(q_gt, k_gt, self.rpb)
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_RmDup_SE(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.se_fc = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, dim),
            nn.Sigmoid()
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT RmDup SE with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def se(self, v, v_gt):
        t = self.n_seg
        bt, heads, h, w, dim = v.shape
        b = bt // t
        v_gap = v.reshape(b, t, heads, h, w, dim).permute(0, 2, 5, 1, 3, 4).reshape(b, heads*dim, t, h, w)
        v_gap = torch.nn.functional.adaptive_avg_pool3d(v_gap, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # b, heads*dim
        v_gap = self.se_fc(v_gap).reshape(b, 1, heads, 1, 1, dim).expand(b, t, heads, 1, 1, dim).reshape(bt, heads, 1, 1, dim)
        #print(v_gt.shape, v_gap.shape)
        return v_gt * v_gap

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        v_gt = self.se(v, v_gt)
        #attn_gt = NATTENQKRPBFunction.apply(q_gt, k_gt, self.rpb)
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_RmDup_SE2(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.se_fc = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, dim),
            nn.Sigmoid()
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT RmDup SE2 with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def se(self, x):
        t = self.n_seg
        bt, h, w, dim = x.shape
        b = bt // t
        x_gap = x.reshape(b, t, h, w, dim).permute(0, 4, 1, 2, 3)
        x_gap = torch.nn.functional.adaptive_avg_pool3d(x_gap, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # b, dim
        x_gap = self.se_fc(x_gap).reshape(b, 1, 1, 1, dim).expand(b, t, 1, 1, dim).reshape(bt, 1, 1, dim)
        #print(v_gt.shape, v_gap.shape)
        return x * x_gap

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        #attn_gt = NATTENQKRPBFunction.apply(q_gt, k_gt, self.rpb)
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = self.se(x)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_RmDup_SE3(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)

        # TODO by hzhang57
        self.se_v_convs = nn.Sequential(
            nn.Conv1d(dim, dim//2, kernel_size=1, groups=num_heads), # or nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim//2, dim, kernel_size=1, groups=num_heads), # or nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.Sigmoid()
        )
        self.se_k_convs = nn.Sequential(
            nn.Conv1d(dim, dim//2, kernel_size=1, groups=num_heads), # or nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim//2, dim, kernel_size=1, groups=num_heads), # or nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.Sigmoid()
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT RmDup SE3 with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def se_v(self, v_gt): # TODO by hzhang57
        t = self.n_seg
        bt, heads, h, w, dim = v_gt.shape
        b = bt // t
        v_gap = v_gt.reshape(b, t, heads, h, w, dim).permute(0, 2, 5, 1, 3, 4).reshape(b, heads*dim, t, h, w)
        v_gap = torch.nn.functional.adaptive_avg_pool3d(v_gap, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # Pool to 1 vector, b, heads*dim
        v_gap = self.se_v_convs(v_gap.unsqueeze(-1)).squeeze(-1) 
        v_gap = v_gap.reshape(b, 1, heads, 1, 1, dim).expand(b, t, heads, 1, 1, dim).reshape(bt, heads, 1, 1, dim)
        return v_gt * v_gap

    def se_k(self, k_gt): # TODO by hzhang57
        t = self.n_seg
        bt, heads, h, w, dim = k_gt.shape
        b = bt // t
        k_gap = k_gt.reshape(b, t, heads, h, w, dim).permute(0, 2, 5, 1, 3, 4).reshape(b, heads*dim, t, h, w)
        k_gap = torch.nn.functional.adaptive_avg_pool3d(k_gap, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # Pool to 1 vector, b, heads*dim
        k_gap = self.se_k_convs(k_gap.unsqueeze(-1)).squeeze(-1) 
        k_gap = k_gap.reshape(b, 1, heads, 1, 1, dim).expand(b, t, heads, 1, 1, dim).reshape(bt, heads, 1, 1, dim)
        return k_gt * k_gap

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        v_gt = self.se_v(v_gt) # Apply SE on value of condensed context, TODO by hzhang57
        k_gt = self.se_k(k_gt) # Apply SE on key of condensed context, TODO by hzhang57
        #attn_gt = NATTENQKRPBFunction.apply(q_gt, k_gt, self.rpb)
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_RmDup_shift(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT RmDup with Shift, n_seg = {}".format(n_seg))

    def shift(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        if step < 0:
            # Shift Back
            out = torch.zeros_like(kv) + 1e-20
            out[:, :step,:,:,:,:] = kv[:, -1*step:,:,:,:,:]
        else:
            # Shift Forward
            out = torch.zeros_like(kv) + 1e-20
            out[:, step:,:,:,:,:] = kv[:, :-1*step,:,:,:,:]
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.shift(k, -1 * self.step)
        v1 = self.shift(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.shift(k, self.step)
        v2 = self.shift(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        #attn_gt = NATTENQKRPBFunction.apply(q_gt, k_gt, self.rpb)
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_RmDup_shiftpad(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT RmDup with Shift, n_seg = {}".format(n_seg))

    def shift_pad(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        if step < 0:
            # Shift Back
            out = kv.clone()
            out[:, :step,:,:,:,:] = kv[:, -1*step:,:,:,:,:]
        else:
            # Shift Forward
            out = kv.clone()
            out[:, step:,:,:,:,:] = kv[:, :-1*step,:,:,:,:]
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.shift_pad(k, -1 * self.step)
        v1 = self.shift_pad(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.shift_pad(k, self.step)
        v2 = self.shift_pad(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        #attn_gt = NATTENQKRPBFunction.apply(q_gt, k_gt, self.rpb)
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT2_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT2 RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def shift(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        if step < 0:
            # Shift Back
            out = torch.zeros_like(kv)
            out[:, :step,:,:,:,:] = kv[:, -1*step:,:,:,:,:]
        else:
            # Shift Forward
            out = torch.zeros_like(kv)
            out[:, step:,:,:,:,:] = kv[:, :-1*step,:,:,:,:]
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)

        #Front Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_fgt = k.reshape(b, t, heads, h, w, dim).cumsum(dim=1)
        v_fgt = v.reshape(b, t, heads, h, w, dim).cumsum(dim=1)
        k_rep = k + self.shift(k, self.step)
        v_rep = v + self.shift(v, self.step)
        k_rep = k_rep.reshape(b, t, heads, h, w, dim)
        v_rep = v_rep.reshape(b, t, heads, h, w, dim)
        k_fgt = k_fgt - k_rep
        v_fgt = v_fgt - v_rep
        divide = torch.maximum(torch.Tensor([1e-20]), torch.arange(0, self.n_seg) - self.step).to(k_fgt.device)
        divide = divide.reshape(1, self.n_seg, 1, 1, 1, 1)
        k_fgt = torch.div(k_fgt, divide).reshape(bt, heads, h, w, dim)
        v_fgt = torch.div(v_fgt, divide).reshape(bt, heads, h, w, dim)

        attn_fgt = NATTENQKRPBFunction.apply(q, k_fgt, self.rpb)

        #Back Global Temporal
        k_bgt = k.reshape(b, t, heads, h, w, dim).flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        v_bgt = v.reshape(b, t, heads, h, w, dim).flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        k_rep = k + self.shift(k, -1 * self.step)
        v_rep = v + self.shift(v, -1 * self.step)
        k_rep = k_rep.reshape(b, t, heads, h, w, dim)
        v_rep = v_rep.reshape(b, t, heads, h, w, dim)
        k_bgt = k_bgt - k_rep
        v_bgt = v_bgt - v_rep
        divide = divide.flip(dims=[1])
        k_bgt = torch.div(k_bgt, divide).reshape(bt, heads, h, w, dim)
        v_bgt = torch.div(v_bgt, divide).reshape(bt, heads, h, w, dim)

        attn_bgt = NATTENQKRPBFunction.apply(q, k_bgt, self.rpb)

        _b, _h, _H, _W, _D = attn.shape
        _, _, _, _, _DD = attn_fgt.shape
        attn = torch.cat((attn, attn1, attn2, attn_fgt, attn_bgt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn1 = attn[:,:,:,:,_D:2*_D]
        attn2 = attn[:,:,:,:,2*_D:3*_D]
        attn_fgt = attn[:,:,:,:,-2*_DD:-_DD]
        attn_bgt = attn[:,:,:,:,-_DD:]
        attn = attn[:,:,:,:,:_D]

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        x_fgt = NATTENAVFunction.apply(attn_fgt, v_fgt)
        x_bgt = NATTENAVFunction.apply(attn_bgt, v_bgt)

        # Simple Addition
        x = x + x1 + x2 + x_fgt + x_bgt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_DownSpat_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT DownSpat RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def dilated_split_func(self, qkv, group=2):
        b, heads, h, w, dim = qkv.shape
        ds = group
        assert h % ds==0 and w % ds==0, "dilated_step+1 {} has to be dividable by H {} or W {}".format(ds, h, w)
        qkv = qkv.reshape(b, heads, h//ds, ds, w//ds, ds, dim)
        qkv = qkv.permute(0, 3, 5, 1, 2, 4, 6) #-> b, ds, ds, heads, h/ds, w/ds, dim
        qkv = qkv.reshape(b*ds*ds, heads, h//ds, w//ds, dim)
        return qkv

    def dilated_restore_fuc(self, qkv, group=2):
        # restore qkv changed by dilated split
        ds = group
        bds, heads, h, w, dim = qkv.shape
        b = bds // (ds * ds) 
        qkv = qkv.reshape(b, ds, ds, heads, h, w, dim)
        qkv = qkv.permute(0, 3, 4, 1, 5, 2, 6) #-> b, heads, h, ds, w, ds, dim
        qkv = qkv.reshape(b, heads, ds*h, ds*w, dim)
        return qkv

    def down_avgpool2d_and_expand(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        assert h % down_spat==0 and w % down_spat==0, "spatial downsample {} has to be dividable by H {} or W {}".format(down_spat, h, w)
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.avg_pool2d(qkv, down_spat, stride=down_spat) # b, heads* dim, h // down_spat, w //down_spat
        # Expand Back
        expand_size = (h, w)
        qkv = torch.nn.functional.interpolate(qkv, size=expand_size, mode='nearest') # b, heads*dim, h, w
        qkv = qkv.reshape(b, heads, dim, h, w)
        qkv = qkv.permute(0, 1, 3, 4, 2).contiguous() # b, heads, h, w, dim
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        # Downsamle Spatial by Pooling with step (default 2), and restore to original size
        k_gt = self.down_avgpool2d_and_expand(k_gt, self.down_spat) # b, heads, h//g, w//g, dim
        v_gt = self.down_avgpool2d_and_expand(v_gt, self.down_spat) # b, heads, h//g, w//g, dim

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_DownSpat_RmDup_V2(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT DownSpat RmDup V2 with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def down_avgpool2d_and_expand(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        assert h % down_spat==0 and w % down_spat==0, "spatial downsample {} has to be dividable by H {} or W {}".format(down_spat, h, w)
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.avg_pool2d(qkv, down_spat, stride=down_spat) # b, heads* dim, h // down_spat, w //down_spat
        # Expand Back
        expand_size = (h, w)
        qkv = torch.nn.functional.interpolate(qkv, size=expand_size, mode='nearest') # b, heads*dim, h, w
        qkv = qkv.reshape(b, heads, dim, h, w)
        qkv = qkv.permute(0, 1, 3, 4, 2).contiguous() # b, heads, h, w, dim
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        # Downsamle Spatial by Pooling with step (default 2), and restore to original size
        k_gt = self.down_avgpool2d_and_expand(k_gt, self.down_spat) # b, heads, h//g, w//g, dim
        v_gt = self.down_avgpool2d_and_expand(v_gt, self.down_spat) # b, heads, h//g, w//g, dim
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_ConcentrateContx_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=3):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT ConcentrateContx RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def concentrate_context(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        assert down_spat % 2==1, "down_spat {} has to be odd number".format(down_spat)
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average context concentration
        qkv = torch.nn.functional.avg_pool2d(qkv, down_spat, stride=1, padding=(down_spat-1)//2, count_include_pad=False) # b, heads* dim, h, w
        qkv = qkv.reshape(b, heads, dim, h, w)
        qkv = qkv.permute(0, 1, 3, 4, 2).contiguous() # b, heads, h, w, dim
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        # Concentrate context by Pooling
        k_gt = self.concentrate_context(k_gt, self.down_spat) # b, heads, h, w, dim
        v_gt = self.concentrate_context(v_gt, self.down_spat) # b, heads, h, w, dim
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_CondenseContext_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT CondenseContext RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def down_adaptive_avgpool2d(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.adaptive_avg_pool2d(qkv, (int(h//down_spat), int(w//down_spat))) # b, heads* dim, h//down_spat, w//down_spat
        # Expand Back
        qkv = qkv.reshape(b, heads, dim, int(h//down_spat), int(w//down_spat))
        qkv = qkv.reshape(b, heads, dim, (int(h//down_spat))*(int(w//down_spat))).permute(0, 1, 3, 2).contiguous() # b, heads, (h//down_spat)*(w//down_spat), dim
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        # TODO Downsamle Spatial by Pooling to fixed shape [h//down_spat, w//down_spat]
        k_gt = self.down_adaptive_avgpool2d(k_gt, self.down_spat) # bt, heads, N, dim
        v_gt = self.down_adaptive_avgpool2d(v_gt, self.down_spat) # bt, heads, N, dim
        q_gt = q.reshape(bt, heads, h*w, dim)
	
        #down_scale = ((h/self.down_spat)*(w/self.down_spat))/(self.kernel_size*self.kernel_size)
        attn_gt = (q_gt) @ (k_gt.transpose(-2,-1))
        #attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        _, _, _, _DD = attn_gt.shape
        attn_gt = attn_gt.reshape(_b, _h, _H, _W, _DD)
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        #attn1 = attn[:,:,:,:,1,:].squeeze(4)
        #attn2 = attn[:,:,:,:,2,:].squeeze(4)
        #attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        #attn = attn[:,:,:,:,0,:].squeeze(4)
        attn1 = attn[:,:,:,:,_D:2*_D]
        attn2 = attn[:,:,:,:,2*_D:3*_D]
        attn_gt = attn[:,:,:,:,-_DD:]
        attn = attn[:,:,:,:,:_D]

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        attn_gt = attn_gt.reshape(_b, _h, _H*_W, _DD)
        x_gt = attn_gt @ v_gt ###### TODO Standard Attention
        x_gt = x_gt.reshape(bt, heads, h, w, dim)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_CondenseContext_RmDup_SE(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.se_fc = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, dim),
            nn.Sigmoid()
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT CondenseContext RmDup SE with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def down_adaptive_avgpool2d(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.adaptive_avg_pool2d(qkv, (h//down_spat, w//down_spat)) # b, heads* dim, h//down_spat, w//down_spat
        qkv = qkv.reshape(b, heads, dim, h//down_spat, w//down_spat)
        qkv = qkv.reshape(b, heads, dim, (h//down_spat)*(w//down_spat)).permute(0, 1, 3, 2).contiguous() # b, heads, (h//down_spat)*(w//down_spat), dim
        return qkv

    def se(self, v, v_gt):
        t = self.n_seg
        bt, heads, h, w, dim = v.shape
        b = bt // t
        v_gap = v.reshape(b, t, heads, h, w, dim).permute(0, 2, 5, 1, 3, 4).reshape(b, heads*dim, t, h, w)
        v_gap = torch.nn.functional.adaptive_avg_pool3d(v_gap, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # b, heads*dim
        v_gap = self.se_fc(v_gap).reshape(b, 1, heads, 1, dim).expand(b, t, heads, 1, dim).reshape(bt, heads, 1, dim)
        return v_gt * v_gap

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        # TODO Downsamle Spatial by Pooling to fixed shape [h//down_spat, w//down_spat]
        k_gt = self.down_adaptive_avgpool2d(k_gt, self.down_spat) # bt, heads, N, dim
        v_gt = self.down_adaptive_avgpool2d(v_gt, self.down_spat) # bt, heads, N, dim
        
        v_gt = self.se(v, v_gt)
        q_gt = q.reshape(bt, heads, h*w, dim)
	
        #down_scale = ((h/self.down_spat)*(w/self.down_spat))/(self.kernel_size*self.kernel_size)
        attn_gt = (q_gt) @ (k_gt.transpose(-2,-1))
        #attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        _, _, _, _DD = attn_gt.shape
        attn_gt = attn_gt.reshape(_b, _h, _H, _W, _DD)
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        #attn1 = attn[:,:,:,:,1,:].squeeze(4)
        #attn2 = attn[:,:,:,:,2,:].squeeze(4)
        #attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        #attn = attn[:,:,:,:,0,:].squeeze(4)
        attn1 = attn[:,:,:,:,_D:2*_D]
        attn2 = attn[:,:,:,:,2*_D:3*_D]
        attn_gt = attn[:,:,:,:,-_DD:]
        attn = attn[:,:,:,:,:_D]

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        attn_gt = attn_gt.reshape(_b, _h, _H*_W, _DD)
        x_gt = attn_gt @ v_gt ###### TODO Standard Attention
        x_gt = x_gt.reshape(bt, heads, h, w, dim)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_CondenseContext_RmDup_SE2(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.se_fc = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, dim),
            nn.Sigmoid()
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT CondenseContext RmDup SE2 with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def down_adaptive_avgpool2d(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.adaptive_avg_pool2d(qkv, (h//down_spat, w//down_spat)) # b, heads* dim, h//down_spat, w//down_spat
        qkv = qkv.reshape(b, heads, dim, h//down_spat, w//down_spat)
        qkv = qkv.reshape(b, heads, dim, (h//down_spat)*(w//down_spat)).permute(0, 1, 3, 2).contiguous() # b, heads, (h//down_spat)*(w//down_spat), dim
        return qkv

    def se(self, x):
        t = self.n_seg
        bt, h, w, dim = x.shape
        b = bt // t
        x_gap = x.reshape(b, t, h, w, dim).permute(0, 4, 1, 2, 3)
        x_gap = torch.nn.functional.adaptive_avg_pool3d(x_gap, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # b, dim
        x_gap = self.se_fc(x_gap).reshape(b, 1, 1, 1, dim).expand(b, t, 1, 1, dim).reshape(bt, 1, 1, dim)
        #print(v_gt.shape, v_gap.shape)
        return x * x_gap

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        # TODO Downsamle Spatial by Pooling to fixed shape [h//down_spat, w//down_spat]
        k_gt = self.down_adaptive_avgpool2d(k_gt, self.down_spat) # bt, heads, N, dim
        v_gt = self.down_adaptive_avgpool2d(v_gt, self.down_spat) # bt, heads, N, dim
        
        q_gt = q.reshape(bt, heads, h*w, dim)
	
        #down_scale = ((h/self.down_spat)*(w/self.down_spat))/(self.kernel_size*self.kernel_size)
        attn_gt = (q_gt) @ (k_gt.transpose(-2,-1))
        #attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        _, _, _, _DD = attn_gt.shape
        attn_gt = attn_gt.reshape(_b, _h, _H, _W, _DD)
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        #attn1 = attn[:,:,:,:,1,:].squeeze(4)
        #attn2 = attn[:,:,:,:,2,:].squeeze(4)
        #attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        #attn = attn[:,:,:,:,0,:].squeeze(4)
        attn1 = attn[:,:,:,:,_D:2*_D]
        attn2 = attn[:,:,:,:,2*_D:3*_D]
        attn_gt = attn[:,:,:,:,-_DD:]
        attn = attn[:,:,:,:,:_D]

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        attn_gt = attn_gt.reshape(_b, _h, _H*_W, _DD)
        x_gt = attn_gt @ v_gt ###### TODO Standard Attention
        x_gt = x_gt.reshape(bt, heads, h, w, dim)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = self.se(x)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_CondenseContext_RmDup_SE3(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)

        # TODO by hzhang57
        self.se_v_convs = nn.Sequential(
            nn.Conv1d(dim, dim//2, kernel_size=1, groups=num_heads), # or nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim//2, dim, kernel_size=1, groups=num_heads), # or nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.Sigmoid()
        )
        self.se_k_convs = nn.Sequential(
            nn.Conv1d(dim, dim//2, kernel_size=1, groups=num_heads), # or nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim//2, dim, kernel_size=1, groups=num_heads), # or nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.Sigmoid()
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT CondenseContext RmDup SE3 with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def down_adaptive_avgpool2d(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.adaptive_avg_pool2d(qkv, (h//down_spat, w//down_spat)) # b, heads* dim, h//down_spat, w//down_spat
        qkv = qkv.reshape(b, heads, dim, h//down_spat, w//down_spat)
        qkv = qkv.reshape(b, heads, dim, (h//down_spat)*(w//down_spat)).permute(0, 1, 3, 2).contiguous() # b, heads, (h//down_spat)*(w//down_spat), dim
        return qkv

    def se_v(self, v_gt): # TODO by hzhang57
        t = self.n_seg
        bt, heads, N, dim = v_gt.shape
        b = bt // t
        v_gap = v_gt.reshape(b, t, heads, N, dim).permute(0, 2, 4, 1, 3).reshape(b, heads*dim, t, N)
        v_gap = torch.nn.functional.adaptive_avg_pool2d(v_gap, 1).squeeze(-1).squeeze(-1)  # Pool to 1 vector, b, heads*dim
        v_gap = self.se_v_convs(v_gap.unsqueeze(-1)).squeeze(-1) 
        v_gap = v_gap.reshape(b, 1, heads, 1, dim).expand(b, t, heads, 1, dim).reshape(bt, heads, 1, dim)
        return v_gt * v_gap

    def se_k(self, k_gt): # TODO by hzhang57
        t = self.n_seg
        bt, heads, N, dim = k_gt.shape
        b = bt // t
        k_gap = k_gt.reshape(b, t, heads, N, dim).permute(0, 2, 4, 1, 3).reshape(b, heads*dim, t, N)
        k_gap = torch.nn.functional.adaptive_avg_pool2d(k_gap, 1).squeeze(-1).squeeze(-1)  # Pool to 1 vector, b, heads*dim
        k_gap = self.se_k_convs(k_gap.unsqueeze(-1)).squeeze(-1) 
        k_gap = k_gap.reshape(b, 1, heads, 1, dim).expand(b, t, heads, 1, dim).reshape(bt, heads, 1, dim)
        return k_gt * k_gap

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        # TODO Downsamle Spatial by Pooling to fixed shape [h//down_spat, w//down_spat]
        k_gt = self.down_adaptive_avgpool2d(k_gt, self.down_spat) # bt, heads, N, dim
        v_gt = self.down_adaptive_avgpool2d(v_gt, self.down_spat) # bt, heads, N, dim
        
        v_gt = self.se_v(v_gt) # Apply SE on value of condensed context, TODO by hzhang57
        k_gt = self.se_k(k_gt) # Apply SE on key of condensed context, TODO by hzhang57
        q_gt = q.reshape(bt, heads, h*w, dim)
	
        #down_scale = ((h/self.down_spat)*(w/self.down_spat))/(self.kernel_size*self.kernel_size)
        attn_gt = (q_gt) @ (k_gt.transpose(-2,-1))
        #attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        _, _, _, _DD = attn_gt.shape
        attn_gt = attn_gt.reshape(_b, _h, _H, _W, _DD)
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        #attn1 = attn[:,:,:,:,1,:].squeeze(4)
        #attn2 = attn[:,:,:,:,2,:].squeeze(4)
        #attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        #attn = attn[:,:,:,:,0,:].squeeze(4)
        attn1 = attn[:,:,:,:,_D:2*_D]
        attn2 = attn[:,:,:,:,2*_D:3*_D]
        attn_gt = attn[:,:,:,:,-_DD:]
        attn = attn[:,:,:,:,:_D]

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        attn_gt = attn_gt.reshape(_b, _h, _H*_W, _DD)
        x_gt = attn_gt @ v_gt ###### TODO Standard Attention
        x_gt = x_gt.reshape(bt, heads, h, w, dim)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_CondenseContext_RmDup_shift(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT CondenseContext RmDup with Shift, n_seg = {}".format(n_seg))

    def shift(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        if step < 0:
            # Shift Back
            out = torch.zeros_like(kv) + 1e-20
            out[:, :step,:,:,:,:] = kv[:, -1*step:,:,:,:,:]
        else:
            # Shift Forward
            out = torch.zeros_like(kv) + 1e-20
            out[:, step:,:,:,:,:] = kv[:, :-1*step,:,:,:,:]
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def down_adaptive_avgpool2d(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.adaptive_avg_pool2d(qkv, (h//down_spat, w//down_spat)) # b, heads* dim, h//down_spat, w//down_spat
        # Expand Back
        qkv = qkv.reshape(b, heads, dim, h//down_spat, w//down_spat)
        qkv = qkv.reshape(b, heads, dim, (h//down_spat)*(w//down_spat)).permute(0, 1, 3, 2).contiguous() # b, heads, (h//down_spat)*(w//down_spat), dim
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.shift(k, -1 * self.step)
        v1 = self.shift(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.shift(k, self.step)
        v2 = self.shift(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        # TODO Downsamle Spatial by Pooling to fixed shape [h//down_spat, w//down_spat]
        k_gt = self.down_adaptive_avgpool2d(k_gt, self.down_spat) # bt, heads, N, dim
        v_gt = self.down_adaptive_avgpool2d(v_gt, self.down_spat) # bt, heads, N, dim
        q_gt = q.reshape(bt, heads, h*w, dim)
	
        #down_scale = ((h/self.down_spat)*(w/self.down_spat))/(self.kernel_size*self.kernel_size)
        attn_gt = (q_gt) @ (k_gt.transpose(-2,-1))
        #attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        _, _, _, _DD = attn_gt.shape
        attn_gt = attn_gt.reshape(_b, _h, _H, _W, _DD)
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        #attn1 = attn[:,:,:,:,1,:].squeeze(4)
        #attn2 = attn[:,:,:,:,2,:].squeeze(4)
        #attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        #attn = attn[:,:,:,:,0,:].squeeze(4)
        attn1 = attn[:,:,:,:,_D:2*_D]
        attn2 = attn[:,:,:,:,2*_D:3*_D]
        attn_gt = attn[:,:,:,:,-_DD:]
        attn = attn[:,:,:,:,:_D]

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        attn_gt = attn_gt.reshape(_b, _h, _H*_W, _DD)
        x_gt = attn_gt @ v_gt ###### TODO Standard Attention
        x_gt = x_gt.reshape(bt, heads, h, w, dim)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT_CondenseContext_RmDup_shiftpad(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT CondenseContext RmDup with Shift pad, n_seg = {}".format(n_seg))

    def shift_pad(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        if step < 0:
            # Shift Back
            out = kv.clone()
            out[:, :step,:,:,:,:] = kv[:, -1*step:,:,:,:,:]
        else:
            # Shift Forward
            out = kv.clone()
            out[:, step:,:,:,:,:] = kv[:, :-1*step,:,:,:,:]
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def down_adaptive_avgpool2d(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.adaptive_avg_pool2d(qkv, (h//down_spat, w//down_spat)) # b, heads* dim, h//down_spat, w//down_spat
        # Expand Back
        qkv = qkv.reshape(b, heads, dim, h//down_spat, w//down_spat)
        qkv = qkv.reshape(b, heads, dim, (h//down_spat)*(w//down_spat)).permute(0, 1, 3, 2).contiguous() # b, heads, (h//down_spat)*(w//down_spat), dim
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.shift_pad(k, -1 * self.step)
        v1 = self.shift_pad(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.shift_pad(k, self.step)
        v2 = self.shift_pad(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_gt = k.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).sum(1)

        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep = k + k1 + k2
        k_gt = k_gt - k_rep
        k_gt = k_gt / (t-3)
        v_rep = v + v1 + v2
        v_gt = v_gt - v_rep
        v_gt = v_gt / (t-3)

        # TODO Downsamle Spatial by Pooling to fixed shape [h//down_spat, w//down_spat]
        k_gt = self.down_adaptive_avgpool2d(k_gt, self.down_spat) # bt, heads, N, dim
        v_gt = self.down_adaptive_avgpool2d(v_gt, self.down_spat) # bt, heads, N, dim
        q_gt = q.reshape(bt, heads, h*w, dim)
	
        #down_scale = ((h/self.down_spat)*(w/self.down_spat))/(self.kernel_size*self.kernel_size)
        attn_gt = (q_gt) @ (k_gt.transpose(-2,-1))
        #attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        _, _, _, _DD = attn_gt.shape
        attn_gt = attn_gt.reshape(_b, _h, _H, _W, _DD)
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        #attn1 = attn[:,:,:,:,1,:].squeeze(4)
        #attn2 = attn[:,:,:,:,2,:].squeeze(4)
        #attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        #attn = attn[:,:,:,:,0,:].squeeze(4)
        attn1 = attn[:,:,:,:,_D:2*_D]
        attn2 = attn[:,:,:,:,2*_D:3*_D]
        attn_gt = attn[:,:,:,:,-_DD:]
        attn = attn[:,:,:,:,:_D]

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        attn_gt = attn_gt.reshape(_b, _h, _H*_W, _DD)
        x_gt = attn_gt @ v_gt ###### TODO Standard Attention
        x_gt = x_gt.reshape(bt, heads, h, w, dim)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT2_CondenseContext_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, down_spat=2):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.down_spat = down_spat
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT2 CondenseContext RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def shift(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        if step < 0:
            # Shift Back
            out = torch.zeros_like(kv)
            out[:, :step,:,:,:,:] = kv[:, -1*step:,:,:,:,:]
        else:
            # Shift Forward
            out = torch.zeros_like(kv)
            out[:, step:,:,:,:,:] = kv[:, :-1*step,:,:,:,:]
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def down_adaptive_avgpool2d(self, qkv, down_spat):
        b, heads, h, w, dim = qkv.shape
        qkv = qkv.permute(0, 1, 4, 2, 3) # b, heads, dim, h, w
        qkv = qkv.reshape(b, heads * dim, h, w) # b, heads * dim, h, w
        # Average Spatial DownSample
        qkv = torch.nn.functional.adaptive_avg_pool2d(qkv, (h//down_spat, w//down_spat)) # b, heads* dim, h//down_spat, w//down_spat
        qkv = qkv.reshape(b, heads, dim, h//down_spat, w//down_spat)
        qkv = qkv.reshape(b, heads, dim, (h//down_spat)*(w//down_spat)).permute(0, 1, 3, 2).contiguous() # b, heads, (h//down_spat)*(w//down_spat), dim
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)

        #Front Global Temporal, DownSample Spatial
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        k_fgt = k.reshape(b, t, heads, h, w, dim).cumsum(dim=1)
        v_fgt = v.reshape(b, t, heads, h, w, dim).cumsum(dim=1)
        k_rep = k + self.shift(k, self.step)
        v_rep = v + self.shift(v, self.step)
        k_rep = k_rep.reshape(b, t, heads, h, w, dim)
        v_rep = v_rep.reshape(b, t, heads, h, w, dim)
        k_fgt = k_fgt - k_rep
        v_fgt = v_fgt - v_rep
        divide = torch.maximum(torch.Tensor([1e-20]), torch.arange(0, self.n_seg) - self.step).to(k_fgt.device)
        divide = divide.reshape(1, self.n_seg, 1, 1, 1, 1)
        k_fgt = torch.div(k_fgt, divide).reshape(bt, heads, h, w, dim)
        v_fgt = torch.div(v_fgt, divide).reshape(bt, heads, h, w, dim)

        k_fgt = self.down_adaptive_avgpool2d(k_fgt, self.down_spat) # bt, heads, N, dim
        v_fgt = self.down_adaptive_avgpool2d(v_fgt, self.down_spat) # bt, heads, N, dim
        q_gt = q.reshape(bt, heads, h*w, dim)
        attn_fgt = (q_gt) @ (k_fgt.transpose(-2,-1))

        #Back Global Temporal, DownSample Spatial
        k_bgt = k.reshape(b, t, heads, h, w, dim).flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        v_bgt = v.reshape(b, t, heads, h, w, dim).flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        k_rep = k + self.shift(k, -1 * self.step)
        v_rep = v + self.shift(v, -1 * self.step)
        k_rep = k_rep.reshape(b, t, heads, h, w, dim)
        v_rep = v_rep.reshape(b, t, heads, h, w, dim)
        k_bgt = k_bgt - k_rep
        v_bgt = v_bgt - v_rep
        divide = divide.flip(dims=[1])
        k_bgt = torch.div(k_bgt, divide).reshape(bt, heads, h, w, dim)
        v_bgt = torch.div(v_bgt, divide).reshape(bt, heads, h, w, dim)

        k_bgt = self.down_adaptive_avgpool2d(k_bgt, self.down_spat) # bt, heads, N, dim
        v_bgt = self.down_adaptive_avgpool2d(v_bgt, self.down_spat) # bt, heads, N, dim
        attn_bgt = (q_gt) @ (k_bgt.transpose(-2,-1))

        _b, _h, _H, _W, _D = attn.shape
        _, _, _, _DD = attn_fgt.shape
        attn_fgt = attn_fgt.reshape(_b, _h, _H, _W, _DD)
        attn_bgt = attn_fgt.reshape(_b, _h, _H, _W, _DD)
        attn = torch.cat((attn, attn1, attn2, attn_fgt, attn_bgt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn1 = attn[:,:,:,:,_D:2*_D]
        attn2 = attn[:,:,:,:,2*_D:3*_D]
        attn_fgt = attn[:,:,:,:,-2*_DD:-_DD]
        attn_bgt = attn[:,:,:,:,-_DD:]
        attn = attn[:,:,:,:,:_D]

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        attn_fgt = attn_fgt.reshape(_b, _h, _H*_W, _DD)
        attn_bgt = attn_bgt.reshape(_b, _h, _H*_W, _DD)
        x_fgt = attn_fgt @ v_fgt
        x_fgt = x_fgt.reshape(bt, heads, h, w, dim)
        x_bgt = attn_bgt @ v_bgt
        x_bgt = x_bgt.reshape(bt, heads, h, w, dim)

        # Simple Addition
        x = x + x1 + x2 + x_fgt + x_bgt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_Dilated3D_hybrid_GlobalT_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, dilated_step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.dilated_step = dilated_step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation Dilated3D hybrid GlobalT RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def dilated_split(self, qkv):
        b, heads, h, w, dim = qkv.shape
        ds = self.dilated_step + 1
        assert h % ds==0 and w % ds==0, "dilated_step+1 {} has to be dividable by H {} or W {}".format(ds, h, w)
        qkv = qkv.reshape(b, heads, h//ds, ds, w//ds, ds, dim)
        qkv = qkv.permute(0, 3, 5, 1, 2, 4, 6) #-> b, ds, ds, heads, h/ds, w/ds, dim
        qkv = qkv.reshape(b*ds*ds, heads, h//ds, w//ds, dim)
        return qkv

    def dilated_restore(self, qkv):
        # restore qkv changed by dilated split
        ds = self.dilated_step + 1
        bds, heads, h, w, dim = qkv.shape
        b = bds // (ds * ds) 
        qkv = qkv.reshape(b, ds, ds, heads, h, w, dim)
        qkv = qkv.permute(0, 3, 4, 1, 5, 2, 6) #-> b, heads, h, ds, w, ds, dim
        qkv = qkv.reshape(b, heads, ds*h, ds*w, dim)
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        num_heads_3d = self.num_heads // 2
        #num_heads_D3d = self.num_heads - num_heads_3d

        q_3d = q[:,:num_heads_3d,:,:,:]
        q_D3d = q[:,num_heads_3d:,:,:,:]
        k_3d = k[:,:num_heads_3d,:,:,:]
        k_D3d = k[:,num_heads_3d:,:,:,:]
        v_3d = v[:,:num_heads_3d,:,:,:]
        v_D3d = v[:,num_heads_3d:,:,:,:]
        rpb_3d = self.rpb[:num_heads_3d,:,:]
        rpb_D3d = self.rpb[num_heads_3d:,:,:]

        '''NAT-3D'''
        #Spatial Neighbour Attention
        attn_3d = NATTENQKRPBFunction.apply(q_3d, k_3d, rpb_3d)
        #Back Correlation Attention
        k1_3d = self.roll(k_3d, -1 * self.step)
        v1_3d = self.roll(v_3d, -1 * self.step)
        attn1_3d = NATTENQKRPBFunction.apply(q_3d, k1_3d, rpb_3d)
        #Front Correlation Attention
        k2_3d = self.roll(k_3d, self.step)
        v2_3d = self.roll(v_3d, self.step)
        attn2_3d = NATTENQKRPBFunction.apply(q_3d, k2_3d, rpb_3d)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q_3d.shape
        b = bt // t
        k_gt_3d = k_3d.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt_3d = v_3d.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt_3d = k_gt_3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt_3d = v_gt_3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep_3d = k_3d + k1_3d + k2_3d
        k_gt_3d = k_gt_3d - k_rep_3d
        k_gt_3d = k_gt_3d / (t-3)
        v_rep_3d = v_3d + v1_3d + v2_3d
        v_gt_3d = v_gt_3d - v_rep_3d
        v_gt_3d = v_gt_3d / (t-3)
        attn_gt_3d = NATTENQKRPBFunction.apply(q_3d, k_gt_3d, rpb_3d)

        _b, _h, _H, _W, _D = attn_3d.shape
        attn_3d = torch.cat((attn_3d, attn1_3d, attn2_3d, attn_gt_3d), dim=-1)
        attn_3d = attn_3d.softmax(dim=-1)
        attn_3d = self.attn_drop(attn_3d)
        attn_3d = attn_3d.reshape(_b, _h, _H, _W, 4, _D)
        attn1_3d = attn_3d[:,:,:,:,1,:].squeeze(4)
        attn2_3d = attn_3d[:,:,:,:,2,:].squeeze(4)
        attn_gt_3d = attn_3d[:,:,:,:,3,:].squeeze(4)
        attn_3d = attn_3d[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x_3d = NATTENAVFunction.apply(attn_3d, v_3d)
        x1_3d = NATTENAVFunction.apply(attn1_3d, v1_3d)
        x2_3d = NATTENAVFunction.apply(attn2_3d, v2_3d)
        x_gt_3d = NATTENAVFunction.apply(attn_gt_3d, v_gt_3d)

        # Simple Addition
        x_3d = x_3d + x1_3d + x2_3d + x_gt_3d

        '''NAT-Dilated3D'''
        #Spatial Neighbour Attention
        q_d_D3d = self.dilated_split(q_D3d)
        k_d_D3d = self.dilated_split(k_D3d)
        attn_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k_d_D3d, rpb_D3d)
        #Back Correlation Attention
        k1_D3d = self.roll(k_D3d, -1 * self.step)
        v1_D3d = self.roll(v_D3d, -1 * self.step)
        k1_d_D3d = self.dilated_split(k1_D3d)
        attn1_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k1_d_D3d, rpb_D3d)
        #Front Correlation Attention
        k2_D3d = self.roll(k_D3d, self.step)
        v2_D3d = self.roll(v_D3d, self.step)
        k2_d_D3d = self.dilated_split(k2_D3d)
        attn2_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k2_d_D3d, rpb_D3d)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q_D3d.shape
        b = bt // t
        k_gt_D3d = k_D3d.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt_D3d = v_D3d.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt_D3d = k_gt_D3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt_D3d = v_gt_D3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep_D3d = k_D3d + k1_D3d + k2_D3d
        k_gt_D3d = k_gt_D3d - k_rep_D3d
        k_gt_D3d = k_gt_D3d / (t-3)
        v_rep_D3d = v_D3d + v1_D3d + v2_D3d
        v_gt_D3d = v_gt_D3d - v_rep_D3d
        v_gt_D3d = v_gt_D3d / (t-3)
        k_gt_d_D3d = self.dilated_split(k_gt_D3d)
        v_gt_d_D3d = self.dilated_split(v_gt_D3d)
        attn_gt_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k_gt_d_D3d, rpb_D3d)

        _b, _h, _H, _W, _D = attn_D3d.shape
        attn_D3d = torch.cat((attn_D3d, attn1_D3d, attn2_D3d, attn_gt_D3d), dim=-1)
        attn_D3d = attn_D3d.softmax(dim=-1)
        attn_D3d = self.attn_drop(attn_D3d)
        attn_D3d = attn_D3d.reshape(_b, _h, _H, _W, 4, _D)
        attn1_D3d = attn_D3d[:,:,:,:,1,:].squeeze(4)
        attn2_D3d = attn_D3d[:,:,:,:,2,:].squeeze(4)
        attn_gt_D3d = attn_D3d[:,:,:,:,3,:].squeeze(4)
        attn_D3d = attn_D3d[:,:,:,:,0,:].squeeze(4)

        v_d_D3d = self.dilated_split(v_D3d)
        v1_d_D3d = self.dilated_split(v1_D3d)
        v2_d_D3d = self.dilated_split(v2_D3d)
        v_gt_d_D3d = self.dilated_split(v_gt_D3d)
        x_D3d = NATTENAVFunction.apply(attn_D3d, v_d_D3d)
        x1_D3d = NATTENAVFunction.apply(attn1_D3d, v1_d_D3d)
        x2_D3d = NATTENAVFunction.apply(attn2_D3d, v2_d_D3d)
        x_gt_D3d = NATTENAVFunction.apply(attn_gt_D3d, v_gt_d_D3d)
        x_D3d = self.dilated_restore(x_D3d)
        x1_D3d = self.dilated_restore(x1_D3d)
        x2_D3d = self.dilated_restore(x2_D3d)
        x_gt_D3d = self.dilated_restore(x_gt_D3d)

        # Simple Addition
        x_D3d = x_D3d + x1_D3d + x2_D3d + x_gt_D3d

        '''Combining NAT-3D & NAT-Dilated3D'''
        x = torch.concat((x_3d, x_D3d), dim=1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_Dilated3D_hybrid_GlobalT_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, dilated_step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.dilated_step = dilated_step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation Dilated3D hybrid GlobalT RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def dilated_split(self, qkv):
        b, heads, h, w, dim = qkv.shape
        ds = self.dilated_step + 1
        assert h % ds==0 and w % ds==0, "dilated_step+1 {} has to be dividable by H {} or W {}".format(ds, h, w)
        qkv = qkv.reshape(b, heads, h//ds, ds, w//ds, ds, dim)
        qkv = qkv.permute(0, 3, 5, 1, 2, 4, 6) #-> b, ds, ds, heads, h/ds, w/ds, dim
        qkv = qkv.reshape(b*ds*ds, heads, h//ds, w//ds, dim)
        return qkv

    def dilated_restore(self, qkv):
        # restore qkv changed by dilated split
        ds = self.dilated_step + 1
        bds, heads, h, w, dim = qkv.shape
        b = bds // (ds * ds) 
        qkv = qkv.reshape(b, ds, ds, heads, h, w, dim)
        qkv = qkv.permute(0, 3, 4, 1, 5, 2, 6) #-> b, heads, h, ds, w, ds, dim
        qkv = qkv.reshape(b, heads, ds*h, ds*w, dim)
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        num_heads_3d = self.num_heads // 2
        #num_heads_D3d = self.num_heads - num_heads_3d

        q_3d = q[:,:num_heads_3d,:,:,:]
        q_D3d = q[:,num_heads_3d:,:,:,:]
        k_3d = k[:,:num_heads_3d,:,:,:]
        k_D3d = k[:,num_heads_3d:,:,:,:]
        v_3d = v[:,:num_heads_3d,:,:,:]
        v_D3d = v[:,num_heads_3d:,:,:,:]
        rpb_3d = self.rpb[:num_heads_3d,:,:]
        rpb_D3d = self.rpb[num_heads_3d:,:,:]

        '''NAT-3D'''
        #Spatial Neighbour Attention
        attn_3d = NATTENQKRPBFunction.apply(q_3d, k_3d, rpb_3d)
        #Back Correlation Attention
        k1_3d = self.roll(k_3d, -1 * self.step)
        v1_3d = self.roll(v_3d, -1 * self.step)
        attn1_3d = NATTENQKRPBFunction.apply(q_3d, k1_3d, rpb_3d)
        #Front Correlation Attention
        k2_3d = self.roll(k_3d, self.step)
        v2_3d = self.roll(v_3d, self.step)
        attn2_3d = NATTENQKRPBFunction.apply(q_3d, k2_3d, rpb_3d)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q_3d.shape
        b = bt // t
        k_gt_3d = k_3d.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt_3d = v_3d.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt_3d = k_gt_3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt_3d = v_gt_3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep_3d = k_3d + k1_3d + k2_3d
        k_gt_3d = k_gt_3d - k_rep_3d
        k_gt_3d = k_gt_3d / (t-3)
        v_rep_3d = v_3d + v1_3d + v2_3d
        v_gt_3d = v_gt_3d - v_rep_3d
        v_gt_3d = v_gt_3d / (t-3)
        attn_gt_3d = NATTENQKRPBFunction.apply(q_3d, k_gt_3d, rpb_3d)

        _b, _h, _H, _W, _D = attn_3d.shape
        attn_3d = torch.cat((attn_3d, attn1_3d, attn2_3d, attn_gt_3d), dim=-1)
        attn_3d = attn_3d.softmax(dim=-1)
        attn_3d = self.attn_drop(attn_3d)
        attn_3d = attn_3d.reshape(_b, _h, _H, _W, 4, _D)
        attn1_3d = attn_3d[:,:,:,:,1,:].squeeze(4)
        attn2_3d = attn_3d[:,:,:,:,2,:].squeeze(4)
        attn_gt_3d = attn_3d[:,:,:,:,3,:].squeeze(4)
        attn_3d = attn_3d[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x_3d = NATTENAVFunction.apply(attn_3d, v_3d)
        x1_3d = NATTENAVFunction.apply(attn1_3d, v1_3d)
        x2_3d = NATTENAVFunction.apply(attn2_3d, v2_3d)
        x_gt_3d = NATTENAVFunction.apply(attn_gt_3d, v_gt_3d)

        # Simple Addition
        x_3d = x_3d + x1_3d + x2_3d + x_gt_3d

        '''NAT-Dilated3D'''
        #Spatial Neighbour Attention
        q_d_D3d = self.dilated_split(q_D3d)
        k_d_D3d = self.dilated_split(k_D3d)
        attn_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k_d_D3d, rpb_D3d)
        #Back Correlation Attention
        k1_D3d = self.roll(k_D3d, -1 * self.step)
        v1_D3d = self.roll(v_D3d, -1 * self.step)
        k1_d_D3d = self.dilated_split(k1_D3d)
        attn1_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k1_d_D3d, rpb_D3d)
        #Front Correlation Attention
        k2_D3d = self.roll(k_D3d, self.step)
        v2_D3d = self.roll(v_D3d, self.step)
        k2_d_D3d = self.dilated_split(k2_D3d)
        attn2_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k2_d_D3d, rpb_D3d)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q_D3d.shape
        b = bt // t
        k_gt_D3d = k_D3d.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt_D3d = v_D3d.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt_D3d = k_gt_D3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt_D3d = v_gt_D3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep_D3d = k_D3d + k1_D3d + k2_D3d
        k_gt_D3d = k_gt_D3d - k_rep_D3d
        k_gt_D3d = k_gt_D3d / (t-3)
        v_rep_D3d = v_D3d + v1_D3d + v2_D3d
        v_gt_D3d = v_gt_D3d - v_rep_D3d
        v_gt_D3d = v_gt_D3d / (t-3)
        k_gt_d_D3d = self.dilated_split(k_gt_D3d)
        v_gt_d_D3d = self.dilated_split(v_gt_D3d)
        attn_gt_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k_gt_d_D3d, rpb_D3d)

        _b, _h, _H, _W, _D = attn_D3d.shape
        attn_D3d = torch.cat((attn_D3d, attn1_D3d, attn2_D3d, attn_gt_D3d), dim=-1)
        attn_D3d = attn_D3d.softmax(dim=-1)
        attn_D3d = self.attn_drop(attn_D3d)
        attn_D3d = attn_D3d.reshape(_b, _h, _H, _W, 4, _D)
        attn1_D3d = attn_D3d[:,:,:,:,1,:].squeeze(4)
        attn2_D3d = attn_D3d[:,:,:,:,2,:].squeeze(4)
        attn_gt_D3d = attn_D3d[:,:,:,:,3,:].squeeze(4)
        attn_D3d = attn_D3d[:,:,:,:,0,:].squeeze(4)

        v_d_D3d = self.dilated_split(v_D3d)
        v1_d_D3d = self.dilated_split(v1_D3d)
        v2_d_D3d = self.dilated_split(v2_D3d)
        v_gt_d_D3d = self.dilated_split(v_gt_D3d)
        x_D3d = NATTENAVFunction.apply(attn_D3d, v_d_D3d)
        x1_D3d = NATTENAVFunction.apply(attn1_D3d, v1_d_D3d)
        x2_D3d = NATTENAVFunction.apply(attn2_D3d, v2_d_D3d)
        x_gt_D3d = NATTENAVFunction.apply(attn_gt_D3d, v_gt_d_D3d)
        x_D3d = self.dilated_restore(x_D3d)
        x1_D3d = self.dilated_restore(x1_D3d)
        x2_D3d = self.dilated_restore(x2_D3d)
        x_gt_D3d = self.dilated_restore(x_gt_D3d)

        # Simple Addition
        x_D3d = x_D3d + x1_D3d + x2_D3d + x_gt_D3d

        '''Combining NAT-3D & NAT-Dilated3D'''
        x = torch.concat((x_3d, x_D3d), dim=1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_Shrink3D_GlobalT_RmDup(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, shrink_size=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.shrink_size = shrink_size
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        #self.pool = nn.MaxPool2d(shrink_size)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation Shrink3D GlobalT RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def shrink_old(self, qkv):
        b, heads, h, w, dim = qkv.shape
        qkv_pool = self.pool(qkv.permute(0, 1, 4, 2, 3).reshape(b, heads*dim, h, w))
        _, _, _h, _w = qkv_pool.shape
        qkv_pool = qkv_pool.reshape(b, heads, dim, _h, _w).permute(0, 1, 3, 4, 2)
        return qkv_pool

    def restore_old(self, qkv):
        b, heads, h, w, dim = qkv.shape
        qkv_origin = qkv.unsqueeze(3).unsqueeze(5)
        qkv_origin = qkv_origin.expand(b, heads, h, self.shrink_size, w, self.shrink_size, dim)
        qkv_origin = qkv_origin.reshape(b, heads, h*self.shrink_size, w*self.shrink_size, dim)
        return qkv_origin

    def shrink(self, qkv):
        b, heads, h, w, dim = qkv.shape
        qkv_pool = qkv.permute(0, 1, 4, 2, 3).reshape(b, heads*dim, h, w)
        qkv_pool = F.unfold(qkv_pool, kernel_size=self.shrink_size, stride=self.shrink_size)
        new_dim = dim * self.shrink_size * self.shrink_size
        new_h, new_w = h // self.shrink_size, w // self.shrink_size
        qkv_pool = qkv_pool.reshape(b, heads, new_dim, new_h, new_w).permute(0, 1, 3, 4, 2)
        return qkv_pool

    def restore(self, qkv):
        b, heads, h, w, dim = qkv.shape
        qkv_origin = qkv.permute(0, 1, 4, 2, 3).reshape(b, heads*dim, h*w)
        old_h, old_w = h * self.shrink_size, w * self.shrink_size
        old_dim = (dim // self.shrink_size) // self.shrink_size
        qkv_origin = F.fold(qkv_origin, output_size=(old_h,old_w), kernel_size=self.shrink_size, stride=self.shrink_size)
        qkv_origin = qkv_origin.reshape(b, heads, old_dim, old_h, old_w).permute(0, 1, 3, 4, 2)
        return qkv_origin

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        num_heads_3d = self.num_heads // 2
        #num_heads_S3d = self.num_heads - num_heads_3d

        q_3d = q[:,:num_heads_3d,:,:,:]
        q_S3d = q[:,num_heads_3d:,:,:,:]
        k_3d = k[:,:num_heads_3d,:,:,:]
        k_S3d = k[:,num_heads_3d:,:,:,:]
        v_3d = v[:,:num_heads_3d,:,:,:]
        v_S3d = v[:,num_heads_3d:,:,:,:]
        rpb_3d = self.rpb[:num_heads_3d,:,:]
        rpb_S3d = self.rpb[num_heads_3d:,:,:]

        '''NAT-3D'''
        #Spatial Neighbour Attention
        attn_3d = NATTENQKRPBFunction.apply(q_3d, k_3d, rpb_3d)
        #Back Correlation Attention
        k1_3d = self.roll(k_3d, -1 * self.step)
        v1_3d = self.roll(v_3d, -1 * self.step)
        attn1_3d = NATTENQKRPBFunction.apply(q_3d, k1_3d, rpb_3d)
        #Front Correlation Attention
        k2_3d = self.roll(k_3d, self.step)
        v2_3d = self.roll(v_3d, self.step)
        attn2_3d = NATTENQKRPBFunction.apply(q_3d, k2_3d, rpb_3d)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q_3d.shape
        b = bt // t
        k_gt_3d = k_3d.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt_3d = v_3d.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt_3d = k_gt_3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt_3d = v_gt_3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep_3d = k_3d + k1_3d + k2_3d
        k_gt_3d = k_gt_3d - k_rep_3d
        k_gt_3d = k_gt_3d / (t-3)
        v_rep_3d = v_3d + v1_3d + v2_3d
        v_gt_3d = v_gt_3d - v_rep_3d
        v_gt_3d = v_gt_3d / (t-3)
        attn_gt_3d = NATTENQKRPBFunction.apply(q_3d, k_gt_3d, rpb_3d)

        _b, _h, _H, _W, _D = attn_3d.shape
        attn_3d = torch.cat((attn_3d, attn1_3d, attn2_3d, attn_gt_3d), dim=-1)
        attn_3d = attn_3d.softmax(dim=-1)
        attn_3d = self.attn_drop(attn_3d)
        attn_3d = attn_3d.reshape(_b, _h, _H, _W, 4, _D)
        attn1_3d = attn_3d[:,:,:,:,1,:].squeeze(4)
        attn2_3d = attn_3d[:,:,:,:,2,:].squeeze(4)
        attn_gt_3d = attn_3d[:,:,:,:,3,:].squeeze(4)
        attn_3d = attn_3d[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x_3d = NATTENAVFunction.apply(attn_3d, v_3d)
        x1_3d = NATTENAVFunction.apply(attn1_3d, v1_3d)
        x2_3d = NATTENAVFunction.apply(attn2_3d, v2_3d)
        x_gt_3d = NATTENAVFunction.apply(attn_gt_3d, v_gt_3d)

        # Simple Addition
        x_3d = x_3d + x1_3d + x2_3d + x_gt_3d

        '''NAT-Shrink3D'''
        #Spatial Neighbour Attention
        q_s_S3d = self.shrink(q_S3d)
        k_s_S3d = self.shrink(k_S3d)
        v_s_S3d = self.shrink(v_S3d)
        attn_S3d = NATTENQKRPBFunction.apply(q_s_S3d, k_s_S3d, rpb_S3d)
        #Back Correlation Attention
        k1_S3d = self.roll(k_S3d, -1 * self.step)
        v1_S3d = self.roll(v_S3d, -1 * self.step)
        k1_s_S3d = self.shrink(k1_S3d)
        v1_s_S3d = self.shrink(v1_S3d)
        attn1_S3d = NATTENQKRPBFunction.apply(q_s_S3d, k1_s_S3d, rpb_S3d)
        #Front Correlation Attention
        k2_S3d = self.roll(k_S3d, self.step)
        v2_S3d = self.roll(v_S3d, self.step)
        k2_s_S3d = self.shrink(k2_S3d)
        v2_s_S3d = self.shrink(v2_S3d)
        attn2_S3d = NATTENQKRPBFunction.apply(q_s_S3d, k2_s_S3d, rpb_S3d)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q_S3d.shape
        b = bt // t
        k_gt_S3d = k_S3d.reshape(b, t, heads, h, w, dim).sum(1)
        v_gt_S3d = v_S3d.reshape(b, t, heads, h, w, dim).sum(1)
        k_gt_S3d = k_gt_S3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt_S3d = v_gt_S3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_rep_S3d = k_S3d + k1_S3d + k2_S3d
        k_gt_S3d = k_gt_S3d - k_rep_S3d
        k_gt_S3d = k_gt_S3d / (t-3)
        v_rep_S3d = v_S3d + v1_S3d + v2_S3d
        v_gt_S3d = v_gt_S3d - v_rep_S3d
        v_gt_S3d = v_gt_S3d / (t-3)
        k_gt_s_S3d = self.shrink(k_gt_S3d)
        v_gt_s_S3d = self.shrink(v_gt_S3d)
        attn_gt_S3d = NATTENQKRPBFunction.apply(q_s_S3d, k_gt_s_S3d, rpb_S3d)

        _b, _h, _H, _W, _D = attn_S3d.shape
        attn_S3d = torch.cat((attn_S3d, attn1_S3d, attn2_S3d, attn_gt_S3d), dim=-1)
        attn_S3d = attn_S3d.softmax(dim=-1)
        attn_S3d = self.attn_drop(attn_S3d)
        attn_S3d = attn_S3d.reshape(_b, _h, _H, _W, 4, _D)
        attn1_S3d = attn_S3d[:,:,:,:,1,:].squeeze(4)
        attn2_S3d = attn_S3d[:,:,:,:,2,:].squeeze(4)
        attn_gt_S3d = attn_S3d[:,:,:,:,3,:].squeeze(4)
        attn_S3d = attn_S3d[:,:,:,:,0,:].squeeze(4)

        x_S3d = NATTENAVFunction.apply(attn_S3d, v_s_S3d)
        x1_S3d = NATTENAVFunction.apply(attn1_S3d, v1_s_S3d)
        x2_S3d = NATTENAVFunction.apply(attn2_S3d, v2_s_S3d)
        x_gt_S3d = NATTENAVFunction.apply(attn_gt_S3d, v_gt_s_S3d)

        # Simple Addition
        x_S3d = x_S3d + x1_S3d + x2_S3d + x_gt_S3d
        x_S3d = self.restore(x_S3d)

        '''Combining NAT-3D & NAT-Shrink3D'''
        x = torch.concat((x_3d, x_S3d), dim=1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_GlobalT(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D GlobalT with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        #q_gt = q.reshape(b, t, heads, h, w, dim).mean(1)
        k_gt = k.reshape(b, t, heads, h, w, dim).mean(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).mean(1)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_gt = k_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        #attn_gt = NATTENQKRPBFunction.apply(q_gt, k_gt, self.rpb)
        attn_gt = NATTENQKRPBFunction.apply(q, k_gt, self.rpb)
        #_, _, _, _, _dim = attn_gt.shape

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        #x_gt = NATTENAVFunction.apply(attn_gt, v_gt)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_Dilated3D_hybrid_GlobalT(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, dilated_step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.dilated_step = dilated_step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation Dilated3D hybrid GlobalT RmDup with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def dilated_split(self, qkv):
        b, heads, h, w, dim = qkv.shape
        ds = self.dilated_step + 1
        assert h % ds==0 and w % ds==0, "dilated_step+1 {} has to be dividable by H {} or W {}".format(ds, h, w)
        qkv = qkv.reshape(b, heads, h//ds, ds, w//ds, ds, dim)
        qkv = qkv.permute(0, 3, 5, 1, 2, 4, 6) #-> b, ds, ds, heads, h/ds, w/ds, dim
        qkv = qkv.reshape(b*ds*ds, heads, h//ds, w//ds, dim)
        return qkv

    def dilated_restore(self, qkv):
        # restore qkv changed by dilated split
        ds = self.dilated_step + 1
        bds, heads, h, w, dim = qkv.shape
        b = bds // (ds * ds) 
        qkv = qkv.reshape(b, ds, ds, heads, h, w, dim)
        qkv = qkv.permute(0, 3, 4, 1, 5, 2, 6) #-> b, heads, h, ds, w, ds, dim
        qkv = qkv.reshape(b, heads, ds*h, ds*w, dim)
        return qkv

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        num_heads_3d = self.num_heads // 2
        #num_heads_D3d = self.num_heads - num_heads_3d

        q_3d = q[:,:num_heads_3d,:,:,:]
        q_D3d = q[:,num_heads_3d:,:,:,:]
        k_3d = k[:,:num_heads_3d,:,:,:]
        k_D3d = k[:,num_heads_3d:,:,:,:]
        v_3d = v[:,:num_heads_3d,:,:,:]
        v_D3d = v[:,num_heads_3d:,:,:,:]
        rpb_3d = self.rpb[:num_heads_3d,:,:]
        rpb_D3d = self.rpb[num_heads_3d:,:,:]

        '''NAT-3D'''
        #Spatial Neighbour Attention
        attn_3d = NATTENQKRPBFunction.apply(q_3d, k_3d, rpb_3d)
        #Back Correlation Attention
        k1_3d = self.roll(k_3d, -1 * self.step)
        v1_3d = self.roll(v_3d, -1 * self.step)
        attn1_3d = NATTENQKRPBFunction.apply(q_3d, k1_3d, rpb_3d)
        #Front Correlation Attention
        k2_3d = self.roll(k_3d, self.step)
        v2_3d = self.roll(v_3d, self.step)
        attn2_3d = NATTENQKRPBFunction.apply(q_3d, k2_3d, rpb_3d)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q_3d.shape
        b = bt // t
        k_gt_3d = k_3d.reshape(b, t, heads, h, w, dim).mean(1)
        v_gt_3d = v_3d.reshape(b, t, heads, h, w, dim).mean(1)
        k_gt_3d = k_gt_3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt_3d = v_gt_3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        attn_gt_3d = NATTENQKRPBFunction.apply(q_3d, k_gt_3d, rpb_3d)

        _b, _h, _H, _W, _D = attn_3d.shape
        attn_3d = torch.cat((attn_3d, attn1_3d, attn2_3d, attn_gt_3d), dim=-1)
        attn_3d = attn_3d.softmax(dim=-1)
        attn_3d = self.attn_drop(attn_3d)
        attn_3d = attn_3d.reshape(_b, _h, _H, _W, 4, _D)
        attn1_3d = attn_3d[:,:,:,:,1,:].squeeze(4)
        attn2_3d = attn_3d[:,:,:,:,2,:].squeeze(4)
        attn_gt_3d = attn_3d[:,:,:,:,3,:].squeeze(4)
        attn_3d = attn_3d[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x_3d = NATTENAVFunction.apply(attn_3d, v_3d)
        x1_3d = NATTENAVFunction.apply(attn1_3d, v1_3d)
        x2_3d = NATTENAVFunction.apply(attn2_3d, v2_3d)
        x_gt_3d = NATTENAVFunction.apply(attn_gt_3d, v_gt_3d)

        # Simple Addition
        x_3d = x_3d + x1_3d + x2_3d + x_gt_3d

        '''NAT-Dilated3D'''
        #Spatial Neighbour Attention
        q_d_D3d = self.dilated_split(q_D3d)
        k_d_D3d = self.dilated_split(k_D3d)
        attn_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k_d_D3d, rpb_D3d)
        #Back Correlation Attention
        k1_D3d = self.roll(k_D3d, -1 * self.step)
        v1_D3d = self.roll(v_D3d, -1 * self.step)
        k1_d_D3d = self.dilated_split(k1_D3d)
        attn1_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k1_d_D3d, rpb_D3d)
        #Front Correlation Attention
        k2_D3d = self.roll(k_D3d, self.step)
        v2_D3d = self.roll(v_D3d, self.step)
        k2_d_D3d = self.dilated_split(k2_D3d)
        attn2_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k2_d_D3d, rpb_D3d)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q_D3d.shape
        b = bt // t
        k_gt_D3d = k_D3d.reshape(b, t, heads, h, w, dim).mean(1)
        v_gt_D3d = v_D3d.reshape(b, t, heads, h, w, dim).mean(1)
        k_gt_D3d = k_gt_D3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        v_gt_D3d = v_gt_D3d.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        k_gt_d_D3d = self.dilated_split(k_gt_D3d)
        v_gt_d_D3d = self.dilated_split(v_gt_D3d)
        attn_gt_D3d = NATTENQKRPBFunction.apply(q_d_D3d, k_gt_d_D3d, rpb_D3d)

        _b, _h, _H, _W, _D = attn_D3d.shape
        attn_D3d = torch.cat((attn_D3d, attn1_D3d, attn2_D3d, attn_gt_D3d), dim=-1)
        attn_D3d = attn_D3d.softmax(dim=-1)
        attn_D3d = self.attn_drop(attn_D3d)
        attn_D3d = attn_D3d.reshape(_b, _h, _H, _W, 4, _D)
        attn1_D3d = attn_D3d[:,:,:,:,1,:].squeeze(4)
        attn2_D3d = attn_D3d[:,:,:,:,2,:].squeeze(4)
        attn_gt_D3d = attn_D3d[:,:,:,:,3,:].squeeze(4)
        attn_D3d = attn_D3d[:,:,:,:,0,:].squeeze(4)

        v_d_D3d = self.dilated_split(v_D3d)
        v1_d_D3d = self.dilated_split(v1_D3d)
        v2_d_D3d = self.dilated_split(v2_D3d)
        v_gt_d_D3d = self.dilated_split(v_gt_D3d)
        x_D3d = NATTENAVFunction.apply(attn_D3d, v_d_D3d)
        x1_D3d = NATTENAVFunction.apply(attn1_D3d, v1_d_D3d)
        x2_D3d = NATTENAVFunction.apply(attn2_D3d, v2_d_D3d)
        x_gt_D3d = NATTENAVFunction.apply(attn_gt_D3d, v_gt_d_D3d)
        x_D3d = self.dilated_restore(x_D3d)
        x1_D3d = self.dilated_restore(x1_D3d)
        x2_D3d = self.dilated_restore(x2_D3d)
        x_gt_D3d = self.dilated_restore(x_gt_D3d)

        # Simple Addition
        x_D3d = x_D3d + x1_D3d + x2_D3d + x_gt_D3d

        '''Combining NAT-3D & NAT-Dilated3D'''
        x = torch.concat((x_3d, x_D3d), dim=1)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_gt(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D gt with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        #Global Temporal
        t = self.n_seg
        bt, heads, h, w, dim = q.shape
        b = bt // t
        q_gt = q.reshape(b, t, heads, h, w, dim).mean(1)
        k_gt = k.reshape(b, t, heads, h, w, dim).mean(1)
        v_gt = v.reshape(b, t, heads, h, w, dim).mean(1)
        v_gt = v_gt.unsqueeze(1).expand(b, t, heads, h, w, dim).reshape(bt, heads, h, w, dim)
        attn_gt = NATTENQKRPBFunction.apply(q_gt, k_gt, self.rpb)
        _, _, _, _, _dim = attn_gt.shape
        attn_gt = attn_gt.unsqueeze(1).expand(b, t, heads, h, w, _dim).reshape(bt, heads, h, w, _dim)

        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2, attn_gt), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 4, _D)

        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn_gt = attn[:,:,:,:,3,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)
        x_gt = NATTENAVFunction.apply(attn_gt, v_gt)

        # Simple Addition
        x = x + x1 + x2 + x_gt

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_2FrontD(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 2FrontD with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out


    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)

        #print("attn shape {}".format(attn0.shape)) # B,heads,H, W, dim
        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn2), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 2, _D)
        attn2 = attn[:,:,:,:,1,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x = NATTENAVFunction.apply(attn, v)
        x2 = NATTENAVFunction.apply(attn2, v2)

        # Simple Addition
        x = x  + x2

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_2BackD(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 2BackD with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        #Back Correlation Attention
        k1 = self.roll(k, -1*self.step)
        v1 = self.roll(v, -1*self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)

        #print("attn shape {}".format(attn0.shape)) # B,heads,H, W, dim
        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 2, _D)
        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)

        # Simple Addition
        x = x  + x1

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_3D_pool(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, pool_size=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.pool = nn.MaxPool2d(pool_size)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D pool with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)

        k_pool = self.pool(k.permute(0, 1, 4, 2, 3).reshape(B, C, H, W))
        _, _, _H, _W = k_pool.shape
        k_pool = k_pool.reshape(B, self.num_heads, self.head_dim, _H, _W).permute(0, 1, 3, 4, 2)
        #v_pool = self.pool(v.permute(0, 1, 4, 2, 3).reshape(B, C, H, W))
        #v_pool = v_pool.reshape(B, self.num_heads, self.head_dim, _H, _W).permute(0, 1, 3, 4, 2)

        #Back Correlation Attention
        k1 = self.roll(k_pool, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k_pool, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)

        #print("attn shape {}".format(attn0.shape)) # B,heads,H, W, dim
        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 3, _D)
        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)

        # print('attn:',attn.shape, 'attn1:', attn1.shape, 'attn2:', attn2.shape)
        # print('v:',v.shape, 'v1:', v1.shape, 'v2:', v2.shape)
        # print('x:',x.shape, 'x1:', x1.shape, 'x2:', x2.shape)

        # Simple Addition
        x = x + x1 + x2

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

class NeighborhoodAttention_3D_convpool(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1, pool_size=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.pool_k = nn.Conv2d(dim, dim, 3, stride=pool_size, padding=1, groups=dim, bias=False)
        self.pool_v = nn.Conv2d(dim, dim, 3, stride=pool_size, padding=1, groups=dim, bias=False)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation 3D convpool with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out


    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)

        k = self.pool_k(k.permute(0, 1, 4, 2, 3).reshape(B, C, H, W))
        _, _, _H, _W = k.shape
        k = k.reshape(B, self.num_heads, self.head_dim, _H, _W).permute(0, 1, 3, 4, 2)
        v = self.pool_v(v.permute(0, 1, 4, 2, 3).reshape(B, C, H, W))
        v = v.reshape(B, self.num_heads, self.head_dim, _H, _W).permute(0, 1, 3, 4, 2)

        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)

        #print("attn shape {}".format(attn0.shape)) # B,heads,H, W, dim
        _b, _h, _H, _W, _D = attn.shape
        attn = torch.cat((attn, attn1, attn2), dim=-1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn.reshape(_b, _h, _H, _W, 3, _D)
        attn1 = attn[:,:,:,:,1,:].squeeze(4)
        attn2 = attn[:,:,:,:,2,:].squeeze(4)
        attn = attn[:,:,:,:,0,:].squeeze(4)
        #print("attn0 shape {}".format(attn0.shape)) # B,heads,H, W, dim

        x = NATTENAVFunction.apply(attn, v)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x2 = NATTENAVFunction.apply(attn2, v2)

        # Simple Addition
        x = x + x1 + x2

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_tct_roll_split(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Temporal Correlation with Roll_split, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        num_heads_f = num_heads_b = self.num_heads // 3
        num_heads_s = self.num_heads - num_heads_f - num_heads_b

        #Spatial Neighbour Attention
        q_s = q[:,num_heads_f:-num_heads_b,:,:,:]
        k_s = k[:,num_heads_f:-num_heads_b,:,:,:]
        v_s = v[:,num_heads_f:-num_heads_b,:,:,:]
        rpb_s = self.rpb[num_heads_f:-num_heads_b,:,:]
        attn = NATTENQKRPBFunction.apply(q_s, k_s, rpb_s)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = NATTENAVFunction.apply(attn, v_s)

        #Back Correlation Attention
        q_b = q[:,-num_heads_b:,:,:,:]
        k_b = k[:,-num_heads_b:,:,:,:]
        v_b = v[:,-num_heads_b:,:,:,:]
        rpb_b = self.rpb[-num_heads_b:,:,:]
        k_b = self.roll(k_b, -1 * self.step)
        v_b = self.roll(v_b, -1 * self.step)
        attn = NATTENQKRPBFunction.apply(q_b, k_b, rpb_b)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_b = NATTENAVFunction.apply(attn, v_b)

        #Front Correlation Attention
        q_f = q[:,:num_heads_f,:,:,:]
        k_f = k[:,:num_heads_f,:,:,:]
        v_f = v[:,:num_heads_f,:,:,:]
        rpb_f = self.rpb[:num_heads_f,:,:]
        k_f = self.roll(k_f, self.step)
        v_f = self.roll(v_f, self.step)
        attn = NATTENQKRPBFunction.apply(q_f, k_f, rpb_f)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_f = NATTENAVFunction.apply(attn, v_f)

        # Simple Addition
        x = torch.concat((x_f, x_s, x_b), 1)

        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_seq_tct_roll(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv_f = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_s = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_b = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb_f = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        self.rpb_s = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        self.rpb_b = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb_f, std=.02, mean=0., a=-2., b=2.)
        trunc_normal_(self.rpb_s, std=.02, mean=0., a=-2., b=2.)
        trunc_normal_(self.rpb_b, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop_f = nn.Dropout(attn_drop)
        self.attn_drop_s = nn.Dropout(attn_drop)
        self.attn_drop_b = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("seq Temporal Correlation with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        
        qkv = self.qkv_f(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        #Front Correlation Attention
        k = self.roll(k, self.step)
        v = self.roll(v, self.step)
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb_f)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop_f(attn)
        x1 = NATTENAVFunction.apply(attn, v)

        x = x1.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        qkv = self.qkv_s(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb_s)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop_s(attn)
        x2 = NATTENAVFunction.apply(attn, v)

        x = x2.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        qkv = self.qkv_b(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        #Back Correlation Attention
        k = self.roll(k, -1 * self.step)
        v = self.roll(v, -1 * self.step)
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb_b)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop_b(attn)
        x3 = NATTENAVFunction.apply(attn, v)

        # Simple Addition
        #x = (x1 + x2 + x3) / 3.0

        x = x3.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class Fusion_sk(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inter_dim = max(dim//2, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim, inter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inter_dim, 3 * dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2, x3):
        input_groups = torch.concat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), dim=1)  #(B, 3, H, W, C)
        input_groups = input_groups.permute(0, 1, 4, 2, 3)   #(B, 3, C, H, W)
        initial = torch.sum(input_groups, dim=1)   #(B, C, H, W)
        B, C, H, W = initial.shape
        gap = self.pool(initial)   #(B, C, 1, 1)
        atten = self.fc2(self.act(self.fc1(gap.squeeze())))   #(B, 3*C)
        atten = self.softmax(atten.view(B, 3, C, 1, 1))   #(B, 3, C, 1, 1)
        feats = torch.sum(input_groups * atten, dim=1)   #(B, C, H, W)
        output = feats.permute(0, 2, 3, 1)
        return output

class Fusion_sk_3d(nn.Module):
    def __init__(self, dim, n_seg):
        super().__init__()
        self.dim = dim
        self.n_seg = n_seg
        inter_dim = max(dim//2, 32)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(dim, inter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inter_dim, 3 * dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2, x3):
        input_groups = torch.concat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), dim=1)  #(B, 3, H, W, C)
        t = self.n_seg
        bt, _, H, W, C = input_groups.shape
        b = bt // t
        input_groups = input_groups.reshape(b, t, 3, H, W, C).permute(0, 2, 5, 1, 3, 4)   #(b, 3, C, t, H, W)
        initial = torch.sum(input_groups, dim=1)   #(b, C, t, H, W)
        gap = self.pool(initial).squeeze()   #(b, C)
        atten = self.fc2(self.act(self.fc1(gap)))   #(b, 3*C)
        atten = self.softmax(atten.view(b, 3, C, 1, 1, 1))   #(b, 3, C, 1, 1, 1)
        feats = torch.sum(input_groups * atten, dim=1)   #(b, C, t, H, W)
        output = feats.permute(0, 2, 3, 4, 1).reshape(bt, H, W, C)
        return output

class Fusion_motion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        inter_dim = max(dim//2, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_heads * 49 * 3, inter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inter_dim, 3 * dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, motion, x1, x2, x3):  #motion:(B, 49*3*h, H, W)
        input_groups = torch.concat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), dim=1)  #(B, 3, H, W, C)
        input_groups = input_groups.permute(0, 1, 4, 2, 3)   #(B, 3, C, H, W)
        B, _, C, H, W = input_groups.shape
        gap = self.pool(motion).squeeze()
        atten = self.fc2(self.act(self.fc1(gap)))   #(B, 3*C)
        atten = self.softmax(atten.view(B, 3, C, 1, 1))   #(B, 3, C, 1, 1)
        feats = torch.sum(input_groups * atten, dim=1)   #(B, C, H, W)
        output = feats.permute(0, 2, 3, 1)
        return output

class Fusion_motion_concat(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        inter_dim = max(dim//2, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(dim + num_heads * 49 * 3, inter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inter_dim, 3 * dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, motion, x1, x2, x3):  #motion:(B, 49*3*h, H, W)
        input_groups = torch.concat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), dim=1)  #(B, 3, H, W, C)
        input_groups = input_groups.permute(0, 1, 4, 2, 3)   #(B, 3, C, H, W)
        initial = torch.sum(input_groups, dim=1)   #(B, C, H, W)
        B, C, H, W = initial.shape
        initial = torch.concat((initial, motion), dim=1)
        gap = self.pool(initial).squeeze()
        atten = self.fc2(self.act(self.fc1(gap)))   #(B, 3*C)
        atten = self.softmax(atten.view(B, 3, C, 1, 1))   #(B, 3, C, 1, 1)
        feats = torch.sum(input_groups * atten, dim=1)   #(B, C, H, W)
        output = feats.permute(0, 2, 3, 1)
        return output


class NeighborhoodAttention_fuse_tct_roll(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.fuse = Fusion_sk_3d(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Fuse Temporal Correlation with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = NATTENAVFunction.apply(attn, v1)
        x1 = x1.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x2 = NATTENAVFunction.apply(attn, v2)
        x2 = x2.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        x = self.fuse(x, x1, x2)

        #x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_fuse_motion_tct_roll(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.fuse = Fusion_motion_concat(dim, num_heads)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Fuse motion Temporal Correlation with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)   #(B, h, H, W, 49)
        x = NATTENAVFunction.apply(attn, v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn1 = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        attn1 = attn.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        x1 = NATTENAVFunction.apply(attn1, v1)
        x1 = x1.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn2 = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        attn2 = attn.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        x2 = NATTENAVFunction.apply(attn2, v2)
        x2 = x2.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        motion = torch.concat((attn2, attn, attn1), dim=-1)
        motion = motion.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)  #(B, 49*3*h, H, W)
        x = self.fuse(motion, x2, x, x1)

        #x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class Bridge_channel(nn.Module):
    def __init__(self, dim, n_seg):
        super().__init__()
        self.dim = dim
        self.n_seg = n_seg
        inter_dim = max(dim//2, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, inter_dim, 1)
        self.bn = nn.BatchNorm3d(inter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(inter_dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, v):   #x:(B, H, W, C) v:(B, h, H, W, d)
        t = self.n_seg
        bt, h, _, _, d = v.shape
        b = bt // t
        gap = self.pool(x.permute(0, 3, 1, 2))   #(B, C, 1, 1)
        feats = self.fc1(gap).view(b, t, -1, 1, 1).permute(0, 2, 1, 3, 4)   #(b, C/2, t, 1, 1)
        feats = self.act(self.bn(feats)).permute(0, 2, 1, 3, 4).view(bt, -1, 1, 1)  #(B, C/2, 1, 1)
        atten = self.sigmoid(self.fc2(feats))   #(B, C, 1, 1)
        output = v * (atten.view(bt, h, 1, 1, d))  #(B, h, H, W, d)
        return output

class Bridge_channel2(nn.Module):
    def __init__(self, dim, n_seg):
        super().__init__()
        self.dim = dim
        self.n_seg = n_seg
        inter_dim = max(dim//2, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, inter_dim, 1)
        self.bn = nn.BatchNorm3d(inter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(inter_dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, v):   #x:(B, H, W, C) v:(B, h, H, W, d)
        t = self.n_seg
        bt, h, _, _, d = v.shape
        b = bt // t
        gap = self.pool(x.permute(0, 3, 1, 2))   #(B, C, 1, 1)
        feats = self.fc1(gap).view(b, t, -1, 1, 1).permute(0, 2, 1, 3, 4)   #(b, C/2, t, 1, 1)
        feats = self.act(self.bn(feats)).permute(0, 2, 1, 3, 4).view(bt, -1, 1, 1)  #(B, C/2, 1, 1)
        atten = 2 * self.sigmoid(self.fc2(feats))   #(B, C, 1, 1)
        output = v * (atten.view(bt, h, 1, 1, d))  #(B, h, H, W, d)
        return output

class Bridge_spatial(nn.Module):
    def __init__(self, dim, n_seg):
        super().__init__()
        self.dim = dim
        self.n_seg = n_seg
        inter_dim = dim//4
        self.fc1 = nn.Conv2d(dim, inter_dim, 1)
        self.bn = nn.BatchNorm3d(inter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(inter_dim, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, q):   #x:(B, H, W, C) q:(B, h, H, W, d)
        t = self.n_seg
        bt, h, H, W, d = q.shape
        b = bt // t
        feats = x.permute(0, 3, 1, 2)   #(B, C, H, W)
        feats = self.fc1(feats).view(b, t, -1, H, W).permute(0, 2, 1, 3, 4)   #(b, C/4, t, H, W)
        feats = self.act(self.bn(feats)).permute(0, 2, 1, 3, 4).view(bt, -1, H, W)  #(B, C/4, H, W)
        atten = self.sigmoid(self.fc2(feats))   #(B, 1, H, W)
        output = q * (atten.unsqueeze(-1))  #(B, h, H, W, d)
        return output

class Bridge_spatial2(nn.Module):
    def __init__(self, dim, n_seg):
        super().__init__()
        self.dim = dim
        self.n_seg = n_seg
        inter_dim = dim//4
        self.fc1 = nn.Conv2d(dim, inter_dim, 1)
        self.bn = nn.BatchNorm3d(inter_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(inter_dim, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, q):   #x:(B, H, W, C) q:(B, h, H, W, d)
        t = self.n_seg
        bt, h, H, W, d = q.shape
        b = bt // t
        feats = x.permute(0, 3, 1, 2)   #(B, C, H, W)
        feats = self.fc1(feats).view(b, t, -1, H, W).permute(0, 2, 1, 3, 4)   #(b, C/4, t, H, W)
        feats = self.act(self.bn(feats)).permute(0, 2, 1, 3, 4).view(bt, -1, H, W)  #(B, C/4, H, W)
        atten = 2 * self.sigmoid(self.fc2(feats))   #(B, 1, H, W)
        output = q * (atten.unsqueeze(-1))  #(B, h, H, W, d)
        return output

class NeighborhoodAttention_bridge_fuse_tct_roll(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.bridge_fs = Bridge_channel2(dim, n_seg)
        # self.bridge_sb = Bridge_channel2(dim, n_seg)
        self.bridge_fs = Bridge_spatial2(dim, n_seg)
        self.bridge_sb = Bridge_spatial2(dim, n_seg)
        self.fuse = Fusion_sk(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Bridge Fuse Temporal Correlation with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x2 = NATTENAVFunction.apply(attn, v2)
        x2 = x2.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        v = self.bridge_fs(x2, v)

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        v = self.bridge_sb(x, v)

        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = NATTENAVFunction.apply(attn, v1)
        x1 = x1.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        x = self.fuse(x, x1, x2)

        #x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))


class NeighborhoodAttention_bridge_q_fuse_tct_roll(nn.Module):
    """
    Neighborhood Attention Module, temporal correlation extention
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_seg=0, step=1):
        super().__init__()
        self.num_heads = num_heads
        self.n_seg = n_seg
        self.step = step
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.bridge_fs = Bridge_spatial(dim, n_seg)
        self.bridge_sb = Bridge_spatial(dim, n_seg)
        self.fuse = Fusion_sk(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        print("Bridge q Fuse Temporal Correlation with Roll, n_seg = {}".format(n_seg))

    def roll(self, kv, step):
        t = self.n_seg
        bt, heads, h, w, dim = kv.shape
        b = bt // t
        kv = kv.reshape(b, t, heads, h, w, dim).contiguous()
        out = torch.roll(kv, shifts=(step,), dims=(1,))
        out = out.reshape(bt, heads, h, w, dim).contiguous()
        return out

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.kernel_size or W < self.kernel_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.kernel_size - W)
            pad_b = max(0, self.kernel_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        #Front Correlation Attention
        k2 = self.roll(k, self.step)
        v2 = self.roll(v, self.step)
        attn = NATTENQKRPBFunction.apply(q, k2, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x2 = NATTENAVFunction.apply(attn, v2)
        x2 = x2.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        q = self.bridge_fs(x2, q)

        #Spatial Neighbour Attention
        attn = NATTENQKRPBFunction.apply(q, k, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTENAVFunction.apply(attn, v)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        q = self.bridge_sb(x, q)

        #Back Correlation Attention
        k1 = self.roll(k, -1 * self.step)
        v1 = self.roll(v, -1 * self.step)
        attn = NATTENQKRPBFunction.apply(q, k1, self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = NATTENAVFunction.apply(attn, v1)
        x1 = x1.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        x = self.fuse(x, x1, x2)

        #x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

