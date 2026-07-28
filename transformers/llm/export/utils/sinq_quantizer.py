"""Pre-SINQ weight reparameterization for Qwen-style decoder blocks.

This follows SINQ's deployable Pre-SINQ variant: Sinkhorn column scales are
absorbed into RMSNorm and adjacent linear weights, so the final model can use
the regular MNN grouped-weight format without a dual-scale runtime kernel.
"""

import torch


class SinqQuantizer:
    def __init__(self, model, group_size=64, iterations=4, repeats=1):
        self.model = model
        self.group_size = group_size
        self.iterations = iterations
        self.repeats = repeats
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _block_size(width, requested):
        block = width if requested == 0 else requested
        while width % block != 0 and block > 1:
            block //= 2
        if block < 1 or width % block != 0:
            raise ValueError(f"SINQ group size {requested} cannot divide input width {width}")
        return block

    def _sinkhorn_column_scale(self, weights):
        """Return SINQ's median-normalized column scale for matching linears."""
        width = weights[0].shape[1]
        if any(weight.shape[1] != width for weight in weights):
            raise ValueError("SINQ linears must have the same input width")
        block = self._block_size(width, self.group_size)

        matrix = torch.cat([weight.detach().to(self.device, torch.float32) for weight in weights], dim=0)
        rows = matrix.shape[0]
        groups = width // block
        matrix = matrix.reshape(rows, groups, block).permute(1, 0, 2).contiguous()

        # Batched form of SINQ's sinkhorn_log(). Each input-channel block has
        # independent row and column balancing factors.
        row_std0 = matrix.std(dim=2).clamp(1e-3, 1e3)
        col_std0 = matrix.std(dim=1).clamp(1e-3, 1e3)
        target = torch.minimum(row_std0.amin(dim=1), col_std0.amin(dim=1)).add_(1e-6)
        log_mu1 = torch.zeros((groups, block), device=self.device)
        log_mu2 = torch.zeros((groups, rows, 1), device=self.device)
        best_imbalance = torch.full((groups,), float("inf"), device=self.device)
        best_mu1 = log_mu1.exp().clone()
        best_mu2 = log_mu2.exp().clone()
        stopped = torch.zeros((groups,), dtype=torch.bool, device=self.device)

        for _ in range(self.iterations):
            current = matrix / log_mu1.exp().unsqueeze(1) / log_mu2.exp()
            row_std = current.std(dim=2).clamp(1e-12)
            col_std = current.std(dim=1).clamp(1e-12)
            imbalance = torch.maximum(row_std.amax(dim=1), col_std.amax(dim=1)) / torch.minimum(
                row_std.amin(dim=1), col_std.amin(dim=1)
            )
            better = imbalance <= best_imbalance
            best_imbalance = torch.minimum(best_imbalance, imbalance)
            best_mu1 = torch.where(better[:, None], log_mu1.exp(), best_mu1)
            best_mu2 = torch.where(better[:, None, None], log_mu2.exp(), best_mu2)
            stopped |= imbalance > best_imbalance
            active = (~stopped).to(torch.float32)
            log_mu1 = (log_mu1 + (col_std / target[:, None]).clamp(0.7, 2.0).log() * active[:, None]).clamp(-0.3, 10.0)
            log_mu2 = (log_mu2 + (row_std / target[:, None]).clamp(0.7, 2.0).log().unsqueeze(-1) * active[:, None, None]).clamp(-0.3, 10.0)

        del matrix, log_mu1, log_mu2, best_mu2
        scale = best_mu1.reshape(-1)
        scale.div_(scale.median().clamp_min(1e-6))
        return scale.cpu()

    @staticmethod
    def _scale_norm_and_inputs(norm, linears, scale):
        dtype = norm.weight.dtype
        norm_scale = scale.to(norm.weight.device, dtype)
        norm.weight.data.mul_(norm_scale)
        for linear in linears:
            linear.weight.data.div_(norm_scale.view(1, -1).to(linear.weight.device, linear.weight.dtype))

    @staticmethod
    def _scale_mlp_hidden(up_proj, down_proj, scale):
        up_scale = scale.to(up_proj.weight.device, up_proj.weight.dtype)
        down_scale = scale.to(down_proj.weight.device, down_proj.weight.dtype)
        up_proj.weight.data.mul_(up_scale.view(-1, 1))
        down_proj.weight.data.div_(down_scale.view(1, -1))

    @torch.no_grad()
    def quantize(self):
        blocks = self.model.blocks
        for repeat in range(self.repeats):
            print(f"Pre-SINQ: pass {repeat + 1}/{self.repeats}")
            for index, block in enumerate(blocks):
                attn = block.self_attn
                mlp = block.mlp
                required = ("q_proj", "k_proj", "v_proj")
                if not all(hasattr(attn, name) for name in required):
                    raise NotImplementedError(f"Pre-SINQ requires QKV projections (block {index})")
                if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
                    raise NotImplementedError(f"Pre-SINQ requires gated MLP projections (block {index})")

                qkv = [attn.q_proj, attn.k_proj, attn.v_proj]
                qkv_scale = self._sinkhorn_column_scale([linear.weight for linear in qkv])
                self._scale_norm_and_inputs(block.input_layernorm, qkv, qkv_scale)

                gate_up = [mlp.gate_proj, mlp.up_proj]
                gate_up_scale = self._sinkhorn_column_scale([linear.weight for linear in gate_up])
                self._scale_norm_and_inputs(block.post_attention_layernorm, gate_up, gate_up_scale)

                down_scale = self._sinkhorn_column_scale([mlp.down_proj.weight])
                self._scale_mlp_hidden(mlp.up_proj, mlp.down_proj, down_scale)

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
