"""ComfyUI nodes: training-free consistency for ANIMA (Cosmos-Predict2 DiT).

AnimaBatchPrompts      – one prompt per batch element (shared prefix + per-image lines) for Anima.
AnimaSharedAttention   – "Side-Panel Decoupled Attention" (SPDA): every image of the batch also attends to
                         the other images (or to anchors) in a separate softmax, as if they were panels placed
                         to its right (RoPE width shift), blended with weight lambda, in mid blocks only.
See https://github.com/adbrasi/anima-consistency-lab for the research behind the defaults.
"""
import math
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.patcher_extension

N_FREQ_T, N_FREQ_H, N_FREQ_W = 22, 21, 21  # Cosmos rope3d for head_dim 128 (44/42/42 dims)


def _w_freqs(dm, device):
    pe = dm.pos_embedder
    w_theta = 10000.0 * pe.w_ntk_factor
    return 1.0 / (w_theta ** pe.dim_spatial_range.to(device).float())  # (21,)


def _shift_w(k, wf, delta):
    """extra RoPE rotation of already-roped keys k (B,H,S,D) by `delta` tokens along width (split-half pairs)."""
    if delta == 0:
        return k
    half = k.shape[-1] // 2
    ang = wf * float(delta)
    c, s = ang.cos().to(k.dtype), ang.sin().to(k.dtype)
    lo = k[..., half - N_FREQ_W:half]
    hi = k[..., 2 * half - N_FREQ_W:]
    k = k.clone()
    k[..., half - N_FREQ_W:half] = lo * c - hi * s
    k[..., 2 * half - N_FREQ_W:] = hi * c + lo * s
    return k


def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


class SPDA:
    def __init__(self, lam, blocks, sig_range, n_anchors, shift_mode, anchor_mask=None, anchor_bias=0.0, lam_late=None, late_sigma=0.6):
        self.anchor_mask = anchor_mask  # (H,W) float mask of the anchor character (image res) -> anchor mode
        self.anchor_bias = anchor_bias
        self.lam_late = lam_late
        self.late_sigma = late_sigma
        self.lam = lam
        self.blocks = blocks
        self.sig_range = sig_range
        self.n_anchors = n_anchors
        self.shift_mode = shift_mode
        self.state = {}

    # DIFFUSION_MODEL wrapper: records grid + sigma, and the w-freqs
    def wrapper(self, executor, x, timesteps, context, *args, **kwargs):
        dm = executor.class_obj
        self.state["W"] = x.shape[-1] // 2
        if self.anchor_mask is not None and self.state.get("mask_hw") != x.shape[-2:]:
            m = self.anchor_mask.float()[None, None].to(x.device)
            m = F.interpolate(m, size=(x.shape[-2] // 2, x.shape[-1] // 2), mode="bilinear")
            self.state["amask"] = (m.flatten() > 0.5)
            self.state["mask_hw"] = x.shape[-2:]
        self.state["wf"] = _w_freqs(dm, x.device)
        to = kwargs.get("transformer_options", {})
        sig = to.get("sigmas", None)
        self.state["sigma"] = float(sig.max()) if sig is not None else 1.0
        return executor(x, timesteps, context, *args, **kwargs)

    def attention(self, func, q, k, v, heads, *args, **kwargs):
        to = kwargs.get("transformer_options", {})
        is_self = k.shape[-2] == q.shape[-2] and q.ndim == 4
        b = to.get("block_index", -1)
        sig = self.state.get("sigma", 1.0)
        if not (is_self and b in self.blocks and self.sig_range[0] <= sig <= self.sig_range[1]) or self.lam <= 0:
            return func(q, k, v, heads, *args, **kwargs)
        B = q.shape[0]
        cou = to.get("cond_or_uncond", [0])
        nch = max(1, len(cou))
        n = B // nch
        if n < 2:
            return func(q, k, v, heads, *args, **kwargs)
        W, wf = self.state["W"], self.state["wf"]
        if self.anchor_mask is not None:
            return self._anchor(q, k, v, n, nch, W, wf)
        lam = self.lam_late if (self.lam_late is not None and sig < self.late_sigma) else self.lam
        own = _sdpa(q, k, v)  # (B,H,S,D)
        out = own.clone()
        anchors = list(range(n)) if self.n_anchors <= 0 else list(range(min(self.n_anchors, n)))
        for c in range(nch):
            base = c * n
            for i in range(n):
                srcs = [j for j in anchors if j != i]
                if not srcs:
                    continue
                ks, vs = [], []
                for r, j in enumerate(srcs):
                    delta = W * (j - i) if self.shift_mode == "side" else 0  # panel j sits (j-i) widths away
                    ks.append(_shift_w(k[base + j:base + j + 1], wf, delta))
                    vs.append(v[base + j:base + j + 1])
                bor = _sdpa(q[base + i:base + i + 1], torch.cat(ks, 2), torch.cat(vs, 2))
                out[base + i] = own[base + i] + lam * (bor[0] - own[base + i])
        return out.transpose(1, 2).reshape(B, -1, out.shape[1] * out.shape[-1])

    def _anchor(self, q, k, v, n, nch, W, wf):
        """batch = [anchor, targets...]: targets attend own keys + masked anchor keys (+bias) in ONE softmax
        (equivalent to blending gated by how well each query matches the anchor character)."""
        B = q.shape[0]
        out = _sdpa(q, k, v)
        idx = torch.nonzero(self.state["amask"]).flatten()
        for c in range(nch):
            base = c * n
            ka = k[base:base + 1][:, :, idx]
            va = v[base:base + 1][:, :, idx]
            for i in range(1, n):
                kb = _shift_w(ka, wf, -W)  # anchor seen as the panel just left of each target
                K = torch.cat([k[base + i:base + i + 1], kb], 2)
                V = torch.cat([v[base + i:base + i + 1], va], 2)
                bias = torch.zeros(K.shape[2], device=q.device, dtype=q.dtype)
                bias[k.shape[2]:] = self.anchor_bias
                out[base + i] = F.scaled_dot_product_attention(q[base + i:base + i + 1], K, V, attn_mask=bias[None, None, None, :])[0]
        return out.transpose(1, 2).reshape(B, -1, out.shape[1] * out.shape[-1])


class AnimaSharedAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "lambda: blend of borrowed-image attention (0.3 safe, 0.5 strong)"}),
            "block_start": ("INT", {"default": 4, "min": 0, "max": 27}),
            "block_end": ("INT", {"default": 16, "min": 1, "max": 28, "tooltip": "exclusive"}),
            "sigma_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "anchors": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "0 = every image shares with every other; K = only the first K images are sources"}),
            "position": (["side", "same"], {"default": "side", "tooltip": "side: borrowed images seen as panels to the right (keeps pose diversity). same: overlay (copies layout)"}),
            "strength_late": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "lambda once sigma < 0.6. Higher (0.7) = more uniform look but measurably lower aesthetic score"}),
        }, "optional": {
            "anchor_mask": ("MASK", {"tooltip": "ANCHOR MODE: batch image 0 is the anchor (same prompt+seed as your reference); "
                                                "mask = its character (e.g. from a background-removal node). Targets borrow only masked anchor tokens."}),
            "anchor_bias": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.25}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "anima/consistency"

    def patch(self, model, strength, block_start, block_end, sigma_start, sigma_end, anchors, position, strength_late=None,
              anchor_mask=None, anchor_bias=0.0):
        m = model.clone()
        if anchor_mask is not None and anchor_mask.ndim == 3:
            anchor_mask = anchor_mask[0]
        sp = SPDA(strength, set(range(block_start, block_end)), (sigma_end, sigma_start + 1e-3), anchors, position,
                  anchor_mask=anchor_mask, anchor_bias=anchor_bias, lam_late=strength_late)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "anima_spda", sp.wrapper)
        m.model_options["transformer_options"]["optimized_attention_override"] = sp.attention
        return (m,)


class AnimaBatchPrompts:
    """Encode one prompt per image: `prefix` + each non-empty line of `prompts`. Output batch conditioning."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "model": ("MODEL",),
            "prefix": ("STRING", {"multiline": True, "default": "masterpiece, best quality, safe, 1girl, solo, "}),
            "prompts": ("STRING", {"multiline": True, "default": "upper body, smile\nfull body, running, city street"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "batch_size")
    FUNCTION = "encode"
    CATEGORY = "anima/consistency"

    def encode(self, clip, model, prefix, prompts):
        lines = [l.strip() for l in prompts.splitlines() if l.strip()]
        mm.load_models_gpu([model])
        dm = model.model.diffusion_model
        dev = mm.get_torch_device()
        dt = model.model.get_dtype_inference() if hasattr(model.model, "get_dtype_inference") else torch.bfloat16
        ctxs = []
        with torch.no_grad():
            for l in lines:
                out = clip.encode_from_tokens(clip.tokenize(prefix + l), return_dict=True)
                ids = out["t5xxl_ids"].unsqueeze(0).to(dev)
                w = out["t5xxl_weights"].unsqueeze(0).unsqueeze(-1).to(dev, dt)
                ctxs.append(dm.preprocess_text_embeds(out["cond"].to(dev, dt), ids, t5xxl_weights=w).cpu())
        return ([[torch.cat(ctxs), {}]], len(lines))


NODE_CLASS_MAPPINGS = {"AnimaSharedAttention": AnimaSharedAttention, "AnimaBatchPrompts": AnimaBatchPrompts}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaSharedAttention": "Anima Shared Attention (SPDA)", "AnimaBatchPrompts": "Anima Batch Prompts"}
