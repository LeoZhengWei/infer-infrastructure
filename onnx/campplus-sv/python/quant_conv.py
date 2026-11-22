"""W4A4 / W4A8 convolutions for CAM++.

Weights are per-output-channel signed int4, stored in int8.
Activations are per-tensor signed int4 (W4A4) or int8 (W4A8).
The product is an int8 matrix multiply (`torch._int_mm`) after im2col.
Batch-norm stays in fp32 on the dequantized convolution output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quant_act(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 7 if bits == 4 else 127
    amax = x.detach().abs().amax()
    scale = torch.clamp(amax / qmax, min=1e-8)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
    return q, scale


class QuantConv(nn.Module):
    def __init__(self, conv: nn.Module, act_bits: int):
        super().__init__()
        if not isinstance(conv, (nn.Conv1d, nn.Conv2d)):
            raise TypeError(type(conv))
        if conv.groups != 1:
            raise ValueError("grouped conv is not used by CAM++")
        weight = conv.weight.detach().float()
        flat = weight.reshape(weight.shape[0], -1)
        amax = flat.abs().amax(dim=1).clamp(min=1e-8)
        scale = amax / 7.0
        q = torch.clamp(torch.round(flat / scale[:, None]), -8, 7).to(torch.int8)
        self.register_buffer("wq", q.contiguous())
        self.register_buffer("wscale", scale.contiguous())
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.detach().float())
        else:
            self.bias = None
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.act_bits = act_bits
        self.is_1d = isinstance(conv, nn.Conv1d)
        self.kshape = tuple(weight.shape[2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq, sx = _quant_act(x, self.act_bits)
        y = self._conv_int(xq)
        scale = sx.to(torch.float32) * self.wscale
        if self.is_1d:
            yf = y.float() * scale.view(1, -1, 1)
        else:
            yf = y.float() * scale.view(1, -1, 1, 1)
        if self.bias is not None:
            view = (1, -1, 1) if self.is_1d else (1, -1, 1, 1)
            yf = yf + self.bias.view(*view)
        return yf

    def _conv_int(self, xq: torch.Tensor) -> torch.Tensor:
        if self.is_1d:
            return self._conv1d(xq)
        return self._conv2d(xq)

    def _conv1d(self, xq: torch.Tensor) -> torch.Tensor:
        n, channels, _ = xq.shape
        kernel = self.kshape[0]
        stride, pad, dil = self.stride[0], self.padding[0], self.dilation[0]
        if kernel == 1 and stride == 1 and pad == 0 and dil == 1:
            col = xq.permute(1, 0, 2).reshape(channels, n * xq.shape[-1])
            out_len = xq.shape[-1]
        else:
            col, out_len = _im2col_1d(xq, kernel, stride, pad, dil)
        y = torch._int_mm(self.wq, col.contiguous())
        return y.view(self.wq.shape[0], n, out_len).permute(1, 0, 2).contiguous()

    def _conv2d(self, xq: torch.Tensor) -> torch.Tensor:
        n, _, height, width = xq.shape
        # Small integers are exact in fp32, so unfold does not change the codes.
        col = F.unfold(
            xq.float(),
            self.kshape,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).to(torch.int8)
        k_flat, length = col.shape[1], col.shape[2]
        col = col.permute(1, 0, 2).reshape(k_flat, n * length)
        y = torch._int_mm(self.wq, col.contiguous())
        out_h = _conv_out(height, self.kshape[0], self.stride[0], self.padding[0], self.dilation[0])
        out_w = _conv_out(width, self.kshape[1], self.stride[1], self.padding[1], self.dilation[1])
        return y.view(self.wq.shape[0], n, out_h, out_w).permute(1, 0, 2, 3).contiguous()


def _conv_out(size: int, kernel: int, stride: int, pad: int, dil: int) -> int:
    return (size + 2 * pad - dil * (kernel - 1) - 1) // stride + 1


def _im2col_1d(x: torch.Tensor, kernel: int, stride: int, pad: int, dil: int):
    if pad:
        x = F.pad(x, (pad, pad))
    n, channels, length = x.shape
    out_len = (length - dil * (kernel - 1) - 1) // stride + 1
    base = torch.arange(out_len, device=x.device) * stride
    offs = torch.arange(kernel, device=x.device) * dil
    index = base[None, :] + offs[:, None]
    cols = x[:, :, index]
    col = cols.permute(1, 2, 0, 3).reshape(channels * kernel, n * out_len)
    return col, out_len


def quantize_model(model: nn.Module, act_bits: int) -> nn.Module:
    if act_bits not in (4, 8):
        raise ValueError(act_bits)

    def swap(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Conv1d, nn.Conv2d)):
                setattr(module, name, QuantConv(child, act_bits))
            else:
                swap(child)

    swap(model)
    model.eval()
    return model
