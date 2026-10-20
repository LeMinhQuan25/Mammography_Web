# Extracted verbatim model architecture / ROI-topology helpers from supplied CBIS_V2.py.
# DO NOT import the full training script: it has top-level training side effects.
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from types import SimpleNamespace
try:
    import gudhi as gd
except ImportError:
    gd = None
CFG = SimpleNamespace()
def set_config(config):
    global CFG
    CFG = SimpleNamespace(**config)
def _robust_lesion_topology_input(image_u8, mask, out_size):
    """
    Build the V25 topology input from the lesion only.

    The ROI is cropped around the union lesion mask with modest padding.
    Intensity is robustly normalized using lesion pixels, while all non-lesion
    pixels are suppressed.  This keeps PH focused on lesion structure instead
    of breast boundary/background/pectoral structures.
    """
    image = np.asarray(image_u8, dtype=np.float32)
    roi = np.asarray(mask) > 0

    if image.ndim != 2 or roi.ndim != 2:
        raise ValueError(
            f"Expected 2-D image/mask, got image={image.shape}, mask={roi.shape}"
        )
    if image.shape != roi.shape:
        raise ValueError(
            f"Image/mask shape mismatch: image={image.shape}, mask={roi.shape}"
        )

    if not roi.any():
        return np.zeros((out_size, out_size), dtype=np.float32), 0

    ys, xs = np.where(roi)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    h = max(1, y1 - y0 + 1)
    w = max(1, x1 - x0 + 1)
    py = int(round(h * float(CFG.topology_roi_padding)))
    px = int(round(w * float(CFG.topology_roi_padding)))

    y0 = max(0, y0 - py)
    y1 = min(image.shape[0] - 1, y1 + py)
    x0 = max(0, x0 - px)
    x1 = min(image.shape[1] - 1, x1 + px)

    crop = image[y0:y1 + 1, x0:x1 + 1]
    crop_mask = roi[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)

    crop = cv2.resize(
        crop, (out_size, out_size), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    crop_mask = cv2.resize(
        crop_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST
    ) > 0

    if not crop_mask.any():
        return np.zeros((out_size, out_size), dtype=np.float32), 0

    fg = crop[crop_mask]
    lo, hi = np.percentile(fg, [1.0, 99.0])
    if float(hi) > float(lo) + 1e-6:
        crop = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)
    else:
        crop = np.zeros_like(crop, dtype=np.float32)

    # Outside-ROI pixels are zero.  The filtration below uses 1-intensity,
    # therefore background enters at the latest filtration value.
    crop *= crop_mask.astype(np.float32)
    return crop.astype(np.float32), 1

def _finite_persistence_intervals(gray01, dimension):
    """
    Cubical Persistent Homology on lesion intensity.

    We use a sublevel filtration of (1 - normalized_intensity): brighter lesion
    structures enter early. Short-lived features below the configured minimum
    persistence are filtered as likely noise.
    """
    filtration = 1.0 - np.clip(
        np.asarray(gray01, dtype=np.float64), 0.0, 1.0
    )

    cc = gd.CubicalComplex(top_dimensional_cells=filtration)
    cc.compute_persistence(
        homology_coeff_field=2,
        min_persistence=float(CFG.topology_min_persistence),
    )

    intervals = cc.persistence_intervals_in_dimension(int(dimension))
    if intervals is None or len(intervals) == 0:
        return np.empty((0, 2), dtype=np.float32)

    intervals = np.asarray(intervals, dtype=np.float64)
    finite = np.isfinite(intervals).all(axis=1)
    intervals = intervals[finite]
    if len(intervals) == 0:
        return np.empty((0, 2), dtype=np.float32)

    return intervals.astype(np.float32)

def _intervals_to_persistence_image(intervals):
    """
    Persistence Diagram -> fixed-size Persistence Image.

    Coordinates are (birth, persistence). Gaussian contributions are weighted
    by persistence, so stable structures contribute more than short-lived ones.
    """
    size = int(CFG.topology_pi_size)
    sigma = max(float(CFG.topology_pi_sigma), 1e-4)

    if intervals is None or len(intervals) == 0:
        return np.zeros((size, size), dtype=np.float32)

    intervals = np.asarray(intervals, dtype=np.float32)
    birth = np.clip(intervals[:, 0], 0.0, 1.0)
    life = np.clip(intervals[:, 1] - intervals[:, 0], 0.0, 1.0)

    keep = life >= float(CFG.topology_min_persistence)
    birth, life = birth[keep], life[keep]
    if len(life) == 0:
        return np.zeros((size, size), dtype=np.float32)

    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    surface = np.zeros((size, size), dtype=np.float32)

    for b, p in zip(birth, life):
        surface += float(p) * np.exp(
            -((xx - float(b)) ** 2 + (yy - float(p)) ** 2)
            / (2.0 * sigma * sigma)
        )

    vmax = float(surface.max())
    if vmax > 0:
        surface /= vmax
    return surface.astype(np.float32)

def replace_convlayers_convnext_author(model_, threshold=300):
    """Same stride-adjustment rule used in the existing BRAIx-compatible notebook."""
    for _, module in model_.named_children():
        if len(list(module.children())) > 0:
            replace_convlayers_convnext_author(module, threshold)
        if isinstance(module, nn.Conv2d) and module.stride[0] == 2 and module.in_channels > threshold:
            module.stride = tuple(max(1, s // 2) for s in module.stride)
    return model_

class TopologyEncoder(nn.Module):
    """Lightweight encoder for the two-channel [PI-H0, PI-H1] representation."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, h0, h1):
        x = torch.cat([h0, h1], dim=1)
        return self.net(x)

class ClinicalSparseBRAIxProtoPNet(nn.Module):
    """
    V26 = BRAIxProtoPNet++-compatible prototype model
          + ROI-guided Clinical-Semantic Sparse prototypes
          + ROI Topological Deep Learning as a bounded teacher enhancement.

    Crucially, topology does NOT redefine prototype identity.  It only supplies
    complementary lesion-structure information to the Global/teacher branch.
    Knowledge Distillation can then transfer useful information toward the
    prototype branch while preserving prototype interpretability.
    """
    def __init__(self, num_prototypes, proto_dim, proto_concept_ids,
                 num_classes=2, temperature=128.0, pretrained=True):
        super().__init__()
        try:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = convnext_tiny(weights=weights)
        except Exception as e:
            print("Pretrained ConvNeXt unavailable; random init:", e)
            base = convnext_tiny(weights=None)

        self.features = replace_convlayers_convnext_author(
            base.features, threshold=300
        )
        self.feature_channels = 768

        self.add_on_layers = nn.Sequential(
            nn.Conv2d(self.feature_channels, proto_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(proto_dim, proto_dim, 1),
            nn.Sigmoid(),
        )

        self.prototype_vectors = nn.Parameter(
            torch.rand(num_prototypes, proto_dim, 1, 1)
        )
        self.register_buffer(
            "ones", torch.ones(num_prototypes, proto_dim, 1, 1)
        )
        self.register_buffer(
            "prototype_concept_ids", proto_concept_ids.clone()
        )
        self.register_buffer(
            "prototype_gate",
            torch.ones(num_prototypes, dtype=torch.float32),
        )

        # Clinical prototypes remain class-agnostic.
        self.last_layer = nn.Linear(num_prototypes, num_classes, bias=True)

        # -------- V25 topology-enhanced Global/teacher branch --------
        self.topology_encoder = TopologyEncoder(
            int(CFG.topology_embedding_dim)
        )
        self.topology_projection = nn.Sequential(
            nn.Linear(CFG.topology_embedding_dim, self.feature_channels),
            nn.LayerNorm(self.feature_channels),
            nn.GELU(),
        )
        self.topology_gate = nn.Sequential(
            nn.Linear(
                self.feature_channels + CFG.topology_embedding_dim, 128
            ),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.topology_fusion_norm = nn.LayerNorm(self.feature_channels)
        self.topology_aux_head = nn.Linear(
            CFG.topology_embedding_dim, num_classes
        )

        # Start conservatively: topology is a small correction, not a replacement.
        final_gate_linear = self.topology_gate[2]
        nn.init.zeros_(final_gate_linear.weight)
        nn.init.constant_(
            final_gate_linear.bias, float(CFG.topology_gate_init_bias)
        )

        self.last_layer_globalnet = nn.Linear(
            self.feature_channels, num_classes, bias=True
        )

        # Existing prototype context gate remains independent from topology.
        self.context_gate = nn.Linear(
            self.feature_channels, num_prototypes, bias=True
        )

        nn.init.normal_(self.last_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.last_layer.bias)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)

        self.num_prototypes = num_prototypes
        self.prototype_shape = (
            num_prototypes, proto_dim, 1, 1
        )
        self.proto_dim = proto_dim
        self.num_classes = num_classes
        self.temperature = temperature

    def _l2_convolution(self, z):
        x2_patch_sum = F.conv2d(z ** 2, self.ones)
        p2 = torch.sum(
            self.prototype_vectors ** 2, dim=(1, 2, 3)
        ).view(-1, 1, 1)
        xp = F.conv2d(z, self.prototype_vectors)
        return F.relu(x2_patch_sum - 2 * xp + p2)

    def distance_2_similarity(self, distances):
        return torch.exp(-distances / self.temperature)

    def forward(self, x, topology_h0=None, topology_h1=None):
        feat = self.features(x)
        z = self.add_on_layers(feat)

        # ---------------- Prototype branch: unchanged in principle ----------------
        dist_maps = self._l2_convolution(z)
        min_dist = -F.max_pool2d(
            -dist_maps,
            kernel_size=(dist_maps.size(2), dist_maps.size(3)),
        ).flatten(1)
        presence_raw = self.distance_2_similarity(min_dist)
        gated = presence_raw * self.prototype_gate.view(1, -1)

        k = min(int(CFG.local_topk_prototypes), gated.size(1))
        if k < gated.size(1):
            delta = (
                self.last_layer.weight[1]
                - self.last_layer.weight[0]
            ).abs().detach()
            evidence_score = gated * (delta.view(1, -1) + 1e-4)
            top_idx = torch.topk(
                evidence_score, k=k, dim=1
            ).indices
            local_mask = torch.zeros_like(gated)
            local_mask.scatter_(1, top_idx, 1.0)
            presence = gated * local_mask
        else:
            local_mask = torch.ones_like(gated)
            presence = gated

        semantic_global_feat = F.adaptive_avg_pool2d(
            feat, 1
        ).flatten(1)

        gate_raw = self.context_gate(semantic_global_feat)
        context_multiplier = (
            1.0
            + float(CFG.context_gate_strength)
            * (2.0 * torch.sigmoid(gate_raw) - 1.0)
        )
        presence_base = presence
        presence = presence_base * context_multiplier

        proto_head_input = F.dropout(
            presence,
            p=float(CFG.proto_head_dropout),
            training=self.training,
        )
        proto_logits = self.last_layer(proto_head_input)

        # ---------------- V25 ROI-topology teacher enhancement ----------------
        if topology_h0 is None or topology_h1 is None:
            # Safe path for prototype-only utilities/dummy calls.
            topology_embedding = torch.zeros(
                x.size(0),
                int(CFG.topology_embedding_dim),
                device=x.device,
                dtype=semantic_global_feat.dtype,
            )
            topology_gate_value = torch.zeros(
                x.size(0), 1,
                device=x.device,
                dtype=semantic_global_feat.dtype,
            )
            topology_logits = self.topology_aux_head(
                topology_embedding
            )
            fused_global_feat = semantic_global_feat
        else:
            # Topology encoder is intentionally run in float32 under AMP because
            # tiny Persistence Images do not need fp16 and numerical stability is cheap.
            with torch.autocast(
                device_type=x.device.type,
                enabled=False,
            ):
                topology_embedding = self.topology_encoder(
                    topology_h0.float(), topology_h1.float()
                )

            topology_embedding = topology_embedding.to(
                dtype=semantic_global_feat.dtype
            )
            topology_context = self.topology_projection(
                topology_embedding
            )
            topology_gate_value = self.topology_gate(
                torch.cat(
                    [semantic_global_feat, topology_embedding], dim=1
                )
            )

            residual = (
                float(CFG.topology_residual_strength)
                * topology_gate_value
                * topology_context
            )
            fused_global_feat = self.topology_fusion_norm(
                semantic_global_feat + residual
            )
            topology_logits = self.topology_aux_head(
                F.dropout(
                    topology_embedding,
                    p=float(CFG.topology_dropout),
                    training=self.training,
                )
            )

        # Clean logits are used only for V26 topology-consistency regularization.
        # The normal GlobalNet path still uses dropout for classification.
        semantic_global_logits = self.last_layer_globalnet(
            semantic_global_feat
        )
        fused_global_logits_clean = self.last_layer_globalnet(
            fused_global_feat
        )

        global_head_input = F.dropout(
            fused_global_feat,
            p=float(CFG.global_head_dropout),
            training=self.training,
        )
        global_logits = self.last_layer_globalnet(global_head_input)

        sim_maps = self.distance_2_similarity(dist_maps)
        return {
            "proto_logits": proto_logits,
            "global_logits": global_logits,
            "topology_logits": topology_logits,
            "topology_embedding": topology_embedding,
            "topology_gate": topology_gate_value,
            "semantic_global_feat": semantic_global_feat,
            "fused_global_feat": fused_global_feat,
            "semantic_global_logits": semantic_global_logits,
            "fused_global_logits_clean": fused_global_logits_clean,
            "presence": presence,
            "presence_base": presence_base,
            "presence_raw": presence_raw,
            "context_multiplier": context_multiplier,
            "context_gate_raw": gate_raw,
            "local_mask": local_mask,
            "min_distances": min_dist,
            "distance_maps": dist_maps,
            "similarity_maps": sim_maps,
            "proto_features": z,
        }

    @torch.no_grad()
    def set_prototype_gate(self, keep_mask):
        keep_mask = torch.as_tensor(
            keep_mask, dtype=self.prototype_gate.dtype, device=self.prototype_gate.device
        )
        if keep_mask.numel() != self.num_prototypes:
            raise ValueError("keep_mask size mismatch")
        self.prototype_gate.copy_(keep_mask)
        pruned = self.prototype_gate <= 0.5
        if pruned.any():
            self.last_layer.weight[:, pruned] = 0.0

    def active_prototype_mask(self):
        return self.prototype_gate > 0.5
