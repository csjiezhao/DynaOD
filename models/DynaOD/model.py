from models.WeDAN.model import Diffusion
from models.WeDAN.utils.tool import generate_Gen_mask

import torch
import torch.nn as nn
import numpy as np

from torch.cuda.amp import autocast
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F


class DynaOD(nn.Module):
    def __init__(self, diff_config, pretrained_ckpt=None, freeze_diffusion=True,
                 poi_dim=34, demo_dim=97):
        super().__init__()
        self.diff_config = diff_config
        self.diffusion_net = Diffusion(diff_config)
        self.shape_net = ShapeNet()
        self.poi_dim = poi_dim
        self.demo_dim = demo_dim

        if pretrained_ckpt is not None:
            self.diffusion_net.load_state_dict(torch.load(pretrained_ckpt, map_location="cpu"))
            print(f"[DynaOD] Loaded pretrained diffusion model from {pretrained_ckpt}")

        if freeze_diffusion:
            for param in self.diffusion_net.parameters():
                param.requires_grad = False
            print("[DynaOD] diffusion_net parameters frozen.")

    def _ddim_step(self, cur_net, t, t_pre, c,
                   use_amp: bool,
                   use_ckpt: bool):
        """
        单步 DDIM：复制 WeDAN Diffusion.DDIM_sample 的核心逻辑，
        但允许我们控制 AMP + checkpoint。
        """
        cur_n, cur_e = cur_net
        real_net, masks, dis, batchlization = c
        real_n, real_e = real_net
        mask_n, mask_e = masks

        # build inpainting inputs (same as WeDAN.DDIM_sample)
        input_n = torch.cat([(real_n * mask_n).unsqueeze(-1),
                             (cur_n * (1 - mask_n)).unsqueeze(-1)], dim=-1)
        input_e = torch.cat([(real_e * mask_e * batchlization).unsqueeze(-1),
                             (cur_e * (1 - mask_e) * batchlization).unsqueeze(-1)], dim=-1)
        inputs = (input_n, input_e)

        amp_ctx = autocast(dtype=torch.bfloat16) if (use_amp and cur_n.is_cuda) else nullcontext()

        def _denoise_ckpt(input_n, input_e, mask_n, mask_e, dis, t, batchlization):
            return self.diffusion_net.model((input_n, input_e), (mask_n, mask_e), dis, t, batchlization)

        with amp_ctx:
            if use_ckpt and self.training:
                noise_n_hat, noise_e_hat = checkpoint(
                    _denoise_ckpt,
                    input_n, input_e, mask_n, mask_e, dis, t, batchlization,
                    use_reentrant=False
                )
            else:
                noise_n_hat, noise_e_hat = self.diffusion_net.model(inputs, masks, dis, t, batchlization)

        noise_e_hat = noise_e_hat * batchlization

        # DDIM update (same as WeDAN.DDIM_sample)
        sqrt_ab_t = self.diffusion_net.sqrt_alphas_cumprod[t]  # (1,)
        sqrt_om_t = self.diffusion_net.sqrt_one_minus_alphas_cumprod[t]  # (1,)
        sqrt_ab_pre = self.diffusion_net.sqrt_alphas_cumprod[t_pre]  # (1,)

        n0_hat = (cur_n - noise_n_hat * sqrt_om_t) / (sqrt_ab_t + 1e-12)
        e0_hat = (cur_e - noise_e_hat * sqrt_om_t) / (sqrt_ab_t + 1e-12)

        # DDIM sigma / coef
        # NOTE: follow your WeDAN code exactly
        sigma_t = self.diff_config["DDIM_eta"] * (
                (1 - self.diffusion_net.alphas_cumprod[t_pre]) / (1 - self.diffusion_net.alphas_cumprod[t]) *
                (1 - self.diffusion_net.alphas_cumprod[t] / self.diffusion_net.alphas_cumprod[t_pre])
        ).sqrt()
        coef_noise = (1 - self.diffusion_net.alphas_cumprod[t_pre] - sigma_t ** 2).sqrt()

        n_t_minus_1 = sqrt_ab_pre * n0_hat + coef_noise * noise_n_hat + torch.randn_like(cur_n) * sigma_t
        e_t_minus_1 = sqrt_ab_pre * e0_hat + coef_noise * noise_e_hat + torch.randn_like(
            cur_e) * sigma_t * batchlization

        return (n_t_minus_1, e_t_minus_1)

    def diffusion_process_train_truncated(self, ods, net, batched_dis, batchlization, scalers,
                                          grad_steps: int = 5,
                                          use_amp: bool = True,
                                          use_ckpt: bool = True):
        """
        训练专用：DDIM 采样仍然跑，但只对最后 grad_steps 步保留梯度。
        """
        device = self.diff_config["device"]
        n, e = net

        # masks/condition
        mask_n, mask_e = generate_Gen_mask(n, e)
        mask_n, mask_e = mask_n.to(device), mask_e.to(device)
        masks = (mask_n, mask_e)
        c = (net, masks, batched_dis, batchlization)

        # ---- DDIM schedule (same as WeDAN) ----
        skip = self.diff_config["T"] // self.diff_config["DDIM_T_sample"]
        sample_Ts = list((torch.arange(0, self.diff_config["T"], skip, device=device) + 1).tolist())
        sample_Ts_pre = [0] + sample_Ts[:-1]

        # init pure noises
        cur_n = torch.randn(n.shape, device=device)
        cur_e = torch.randn(e.shape, device=device) * batchlization
        cur_net = (cur_n, cur_e)

        # how many steps total?
        total_steps = len(sample_Ts)
        grad_steps = int(min(max(1, grad_steps), total_steps))

        # split into no_grad prefix + grad suffix
        prefix = total_steps - grad_steps

        # ---- prefix: no graph ----
        if prefix > 0:
            with torch.no_grad():
                for t, t_pre in zip(reversed(sample_Ts[:prefix]), reversed(sample_Ts_pre[:prefix])):
                    t = torch.tensor([t], device=device, dtype=torch.long)
                    t_pre = torch.tensor([t_pre], device=device, dtype=torch.long)
                    # no_grad step (no amp/ckpt needed)
                    cur_n, cur_e = cur_net
                    cur_e = cur_e * batchlization
                    cur_net = (cur_n, cur_e)
                    cur_net = self._ddim_step(cur_net, t, t_pre, c, use_amp=False, use_ckpt=False)

            # IMPORTANT: detach to cut history completely
            cur_net = (cur_net[0].detach(), cur_net[1].detach())

        # ---- suffix: keep graph + amp + checkpoint ----
        for t, t_pre in zip(reversed(sample_Ts[prefix:]), reversed(sample_Ts_pre[prefix:])):
            t = torch.tensor([t], device=device, dtype=torch.long)
            t_pre = torch.tensor([t_pre], device=device, dtype=torch.long)
            cur_n, cur_e = cur_net
            cur_e = cur_e * batchlization
            cur_net = (cur_n, cur_e)
            cur_net = self._ddim_step(cur_net, t, t_pre, c, use_amp=use_amp, use_ckpt=use_ckpt)

        # final prediction
        n_hat, e_hat = cur_net

        # slice into per-city od pairs (training: keep tensors, no inverse)
        od_pairs = []
        l = 0
        for od in ods:
            r = l + od.shape[0]
            od_hat = e_hat[l:r, l:r]
            tmp_mask_e = mask_e[l:r, l:r]
            l = r
            if (tmp_mask_e < 1e-6).sum() == 0:
                continue
            od_gt = od.to(device)
            od_pairs.append((od_hat, od_gt))
        return od_pairs


    def diffusion_process(self, ods, net, batched_dis, batchlization, scalers):
        device = self.diff_config["device"]
        n, e = net

        mask_n, mask_e = generate_Gen_mask(n, e)
        mask_n, mask_e = mask_n.to(device), mask_e.to(device)
        masks = (mask_n, mask_e)
        c = (net, masks, batched_dis, batchlization)

        sample_times = int(self.diff_config["sample_times"])
        n_hat_sum, e_hat_sum = None, None

        for _ in range(sample_times):
            n_hat_t, e_hat_t = self.diffusion_net.DDIM_sample_loop(n.shape, e.shape, c)
            if n_hat_sum is None:
                n_hat_sum, e_hat_sum = n_hat_t, e_hat_t
            else:
                n_hat_sum += n_hat_t
                e_hat_sum += e_hat_t

        inv = 1.0 / sample_times
        n_hat = n_hat_sum * inv
        e_hat = e_hat_sum * inv

        od_pairs = []
        l = 0
        for idx, od in enumerate(ods):
            r = l + od.shape[0]
            od_hat = e_hat[l:r, l:r]
            tmp_mask_e = mask_e[l:r, l:r]
            l = r

            if (tmp_mask_e < 1e-6).sum() == 0:
                continue

            od_gt = od.to(device)

            if not self.training:
                # detach + CPU
                od_gt_np = od_gt.detach().cpu().numpy()
                od_hat_np = od_hat.detach().cpu().numpy()

                # inverse transform
                od_gt_np = scalers["od"].inverse_transform(od_gt_np.reshape([-1, 1])).reshape(
                    [od.shape[0], od.shape[1]])
                od_gt_np = scalers["od_normer"].inverse_transform(od_gt_np)
                od_gt_np[od_gt_np < 0] = 0

                od_hat_np = scalers["od"].inverse_transform(od_hat_np.reshape([-1, 1])).reshape(
                    [od_hat_np.shape[0], od_hat_np.shape[1]])
                od_hat_np = scalers["od_normer"].inverse_transform(od_hat_np)

                # 对角置零 + 去负 + floor
                np.fill_diagonal(od_hat_np, 0)
                od_hat_np[od_hat_np < 0] = 0
                od_hat_np = np.floor(od_hat_np)

                od_hat, od_gt = od_hat_np, od_gt_np

            od_pairs.append((od_hat, od_gt))

        return od_pairs


    def forward(self, poi_ctrl_list, demo_ctrl_list, t_context_list, external_poi_shp, external_demo_shp,
                ods_list, net_list, batched_dis, batchlization_list, scalers,
                poi_control=True, demo_control=True, external_shape=False):
        # ---- 1) stack sequences for ShapeNet ----
        poi_ctrl_seq = torch.stack(poi_ctrl_list, dim=1)  # (N,T,poi_dim)
        demo_ctrl_seq = torch.stack(demo_ctrl_list, dim=1)  # (N,T,demo_dim)
        t_context_seq = torch.stack(t_context_list, dim=1)  # (N,T,t_dim)

        # ---- 2) static node features from day0 ----
        n0, _ = net_list[0]
        poi_feat = n0[:, -self.poi_dim:].clone()
        demo_feat = n0[:, :-self.poi_dim].clone()

        # ---- 3) ShapeNet ----
        shape_out = self.shape_net(poi_feat, demo_feat, poi_ctrl_seq, demo_ctrl_seq, t_context_seq)
        poi_shape_seq = shape_out["poi_shape_seq"]  # (N,T,poi_dim)
        demo_shape_seq = shape_out["demo_shape_seq"]  # (N,T,demo_dim)

        # ====== add weighted smoothness loss (train-only) ======
        reg_loss = None
        if self.training:
            lam = float(self.diff_config.get("lambda_smooth", 0.1))
            alpha = float(self.diff_config.get("alpha_stable", 3.0))
            # 只对“使用 control 的那部分”加约束更合理
            reg_poi = self.shape_net.weighted_smoothness_loss(
                poi_shape_seq, poi_ctrl_seq, alpha=alpha
            ) if poi_control else 0.0
            reg_demo = self.shape_net.weighted_smoothness_loss(
                demo_shape_seq, demo_ctrl_seq, alpha=alpha
            ) if demo_control else 0.0
            reg_loss = lam * (reg_poi + reg_demo)

        od_pairs_list = []
        for t in range(len(poi_ctrl_list)):
            _, e_t = net_list[t]
            ods_t = ods_list[t]
            batchlization_t = batchlization_list[t]

            poi_ctrl_t = poi_ctrl_seq[:, t, :]
            demo_ctrl_t = demo_ctrl_seq[:, t, :]

            if external_shape:
                poi_shp_src = external_poi_shp[t]
                demo_shp_src = external_demo_shp[t]
            else:
                poi_shp_src = poi_shape_seq[:, t, :]
                demo_shp_src = demo_shape_seq[:, t, :]

            poi_new = poi_feat if not poi_control else (poi_feat * (1.0 + poi_shp_src * poi_ctrl_t))
            demo_new = demo_feat if not demo_control else (demo_feat * (1.0 + demo_shp_src * demo_ctrl_t))
            n_new = torch.cat([demo_new, poi_new], dim=-1)
            net_new = (n_new, e_t)

            if self.training:
                od_pairs_t = self.diffusion_process_train_truncated(
                    ods_t, net_new, batched_dis, batchlization_t, scalers,
                    grad_steps=self.diff_config.get("DDIM_grad_steps", 5),
                    use_amp=True,
                    use_ckpt=True)
            else:
                od_pairs_t = self.diffusion_process(ods_t, net_new, batched_dis, batchlization_t, scalers)
            od_pairs_list.append(od_pairs_t)

        if not self.training:
            return od_pairs_list

        return od_pairs_list, reg_loss


class ShapeNet(nn.Module):
    def __init__(self,
                 poi_dim=34, demo_dim=97, t_dim=8,
                 hidden=256,
                 # temporal
                 use_gru=True,
                 # weak conditioning
                 use_feat_cond=True,
                 feat_cond_dim=32,
                 film_strength=0.1,
                 # distribution / sampling
                 K=4,
                 agg_mode="mean",          # "mean" or "mixture"
                 logvar_min=-6.0,
                 logvar_max=2.0):
        super().__init__()
        assert agg_mode in ("mean", "mixture")

        self.poi_dim = poi_dim
        self.demo_dim = demo_dim
        self.t_dim = t_dim
        self.hidden = hidden
        self.use_gru = use_gru

        self.use_feat_cond = use_feat_cond
        self.feat_cond_dim = feat_cond_dim
        self.film_strength = film_strength

        self.K = K
        self.agg_mode = agg_mode
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # --- 1) Ctrl-Time dominant tokenization ---
        # For each modality, token = [ctrl_sign, ctrl_on_mask, t_context]
        self.poi_token_proj = nn.Linear(2 * poi_dim + t_dim, hidden)
        self.demo_token_proj = nn.Linear(2 * demo_dim + t_dim, hidden)

        # --- 2) Temporal modeling over 7 days ---
        if use_gru:
            self.poi_gru = nn.GRU(hidden, hidden, batch_first=True)
            self.demo_gru = nn.GRU(hidden, hidden, batch_first=True)
        else:
            # simple MLP per day (no temporal dependency). kept for ablation.
            self.poi_gru = None
            self.demo_gru = None

        # --- 3) Fuse POI/Demo temporal streams ---
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # --- 4) Weak conditioning from features (FiLM) ---
        if use_feat_cond:
            self.zone_code = nn.Sequential(
                nn.Linear(poi_dim + demo_dim, feat_cond_dim),
                nn.ReLU(inplace=True),
            )
            self.film_gamma = nn.Linear(feat_cond_dim, hidden)
            self.film_beta = nn.Linear(feat_cond_dim, hidden)

        # --- 5) Distribution heads ---
        self.poi_mu = nn.Linear(hidden, poi_dim)
        self.poi_logvar = nn.Linear(hidden, poi_dim)
        self.demo_mu = nn.Linear(hidden, demo_dim)
        self.demo_logvar = nn.Linear(hidden, demo_dim)

        # optional mixture weights over K samples
        if self.agg_mode == "mixture":
            self.sample_weight = nn.Linear(hidden, K)

        # learnable output scaling (amplitude range control)
        self.poi_scale = nn.Parameter(torch.ones(1, 1, poi_dim))
        self.demo_scale = nn.Parameter(torch.ones(1, 1, demo_dim))

        self.reset_parameters(logvar_init=-4.0, scale_init=1.0)

    def reset_parameters(self, logvar_init: float = -4.0, scale_init: float = 1.0):
        """
        Better init for stable training:
        - start from near-zero perturbation
        - small variance sampling
        - FiLM initially identity
        """

        # ---- generic init for Linear layers (except special heads) ----
        def init_linear(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # apply to main trunk
        for m in [self.poi_token_proj, self.demo_token_proj]:
            init_linear(m)
        self.fuse.apply(init_linear)

        if self.use_feat_cond:
            self.zone_code.apply(init_linear)
            # FiLM start as identity: gamma=0, beta=0
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)

        # ---- GRU init (more stable) ----
        def init_gru(gru: nn.GRU):
            for name, p in gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

        if self.use_gru:
            init_gru(self.poi_gru)
            init_gru(self.demo_gru)

        # ---- distribution heads: start from zero perturbation ----
        # mu heads -> exactly 0 at init
        nn.init.zeros_(self.poi_mu.weight)
        nn.init.zeros_(self.poi_mu.bias)
        nn.init.zeros_(self.demo_mu.weight)
        nn.init.zeros_(self.demo_mu.bias)

        # logvar heads -> small variance around mu=0
        nn.init.zeros_(self.poi_logvar.weight)
        nn.init.constant_(self.poi_logvar.bias, logvar_init)
        nn.init.zeros_(self.demo_logvar.weight)
        nn.init.constant_(self.demo_logvar.bias, logvar_init)

        # mixture weights -> uniform at init
        if self.agg_mode == "mixture":
            nn.init.zeros_(self.sample_weight.weight)
            nn.init.zeros_(self.sample_weight.bias)

        # output scales
        with torch.no_grad():
            self.poi_scale.fill_(scale_init)
            self.demo_scale.fill_(scale_init)

    def _clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        return torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

    def _sample_and_aggregate(self, mu: torch.Tensor, logvar: torch.Tensor, weight_logits: torch.Tensor | None):
        """
        mu/logvar: (N,T,D)
        weight_logits: (N,T,K) or None
        returns: shape_seq (N,T,D), norm_weights (N,T,K) or None
        """
        N, T, D = mu.shape
        eps = torch.randn(N, T, self.K, D, device=mu.device, dtype=mu.dtype)
        std = torch.exp(0.5 * logvar).unsqueeze(2)   # (N,T,1,D)
        samples = mu.unsqueeze(2) + eps * std        # (N,T,K,D)

        if self.agg_mode == "mean":
            return samples.mean(dim=2), None

        # mixture
        w = torch.softmax(weight_logits, dim=-1).unsqueeze(-1)  # (N,T,K,1)
        agg = (samples * w).sum(dim=2)                          # (N,T,D)
        return agg, w.squeeze(-1)                               # (N,T,K)

    @staticmethod
    def weighted_smoothness_loss(shape_seq: torch.Tensor,
                                 ctrl_seq: torch.Tensor,
                                 alpha: float = 2.0) -> torch.Tensor:
        """
        shape_seq, ctrl_seq: (N,T,D)
        alpha: how much stronger to enforce smoothness when ctrl==0
        """
        diff = shape_seq[:, 1:, :] - shape_seq[:, :-1, :]  # (N,T-1,D)
        stable = (ctrl_seq[:, :-1, :] == 0).float()  # (N,T-1,D)
        w = 1.0 + alpha * stable
        return (w * (diff ** 2)).mean()

    @staticmethod
    def smoothness_loss(seq: torch.Tensor) -> torch.Tensor:
        diff = seq[:, 1:, :] - seq[:, :-1, :]
        return (diff ** 2).mean()

    def forward(self, poi_feat, demo_feat, poi_ctrl_seq, demo_ctrl_seq, t_context_seq):
        """
        Ctrl-Time dominant, distributional, temporal ShapeNet for 7-day window.

        Inputs:
          poi_feat:      (N, poi_dim)         # weak conditioning only
          demo_feat:     (N, demo_dim)        # weak conditioning only
          poi_ctrl_seq:  (N, 7, poi_dim)      # {-1,0,1}
          demo_ctrl_seq: (N, 7, demo_dim)     # {-1,0,1}
          t_context_seq: (N, 7, t_dim)

        Outputs (dict):
          poi_mu_seq:      (N,7,poi_dim)
          poi_logvar_seq:  (N,7,poi_dim)
          demo_mu_seq:     (N,7,demo_dim)
          demo_logvar_seq: (N,7,demo_dim)
          poi_shape_seq:   (N,7,poi_dim)      # aggregated sample (use for modulation)
          demo_shape_seq:  (N,7,demo_dim)
          (optional) weights: (N,7,K) if agg_mode="mixture"
        """
        N, T, _ = poi_ctrl_seq.shape
        # assert T == 7, "This design targets a fixed 7-day window."

        # ctrl sign & activation mask
        poi_on = (poi_ctrl_seq != 0).float()
        demo_on = (demo_ctrl_seq != 0).float()

        # tokens: (N,7,2*dim+t_dim)
        poi_token = torch.cat([poi_ctrl_seq, poi_on, t_context_seq], dim=-1)
        demo_token = torch.cat([demo_ctrl_seq, demo_on, t_context_seq], dim=-1)

        # project to hidden
        poi_h = F.relu(self.poi_token_proj(poi_token))   # (N,7,H)
        demo_h = F.relu(self.demo_token_proj(demo_token))# (N,7,H)

        # temporal modeling
        if self.use_gru:
            poi_h, _ = self.poi_gru(poi_h)
            demo_h, _ = self.demo_gru(demo_h)

        # fuse
        h = self.fuse(torch.cat([poi_h, demo_h], dim=-1))  # (N,7,H)

        # weak feature conditioning (FiLM, small magnitude)
        if self.use_feat_cond:
            z = self.zone_code(torch.cat([poi_feat, demo_feat], dim=-1))  # (N,z)
            gamma = torch.tanh(self.film_gamma(z)).unsqueeze(1) * self.film_strength
            beta = torch.tanh(self.film_beta(z)).unsqueeze(1) * self.film_strength
            h = h * (1.0 + gamma) + beta

        # distribution params
        poi_mu_seq = torch.tanh(self.poi_mu(h)) * self.poi_scale
        demo_mu_seq = torch.tanh(self.demo_mu(h)) * self.demo_scale
        poi_logvar_seq = self._clamp_logvar(self.poi_logvar(h))
        demo_logvar_seq = self._clamp_logvar(self.demo_logvar(h))

        weight_logits = None
        if self.agg_mode == "mixture":
            weight_logits = self.sample_weight(h)  # (N,7,K)

        # sample aggregation
        poi_shape_seq, norm_w = self._sample_and_aggregate(poi_mu_seq, poi_logvar_seq, weight_logits)
        demo_shape_seq, _ = self._sample_and_aggregate(demo_mu_seq, demo_logvar_seq, weight_logits)

        out = {
            "poi_shape_seq": poi_shape_seq,
            "demo_shape_seq": demo_shape_seq,
            "poi_mu_seq": poi_mu_seq,
            "poi_logvar_seq": poi_logvar_seq,
            "demo_mu_seq": demo_mu_seq,
            "demo_logvar_seq": demo_logvar_seq,
        }
        if norm_w is not None:
            out["weights"] = norm_w
        return out