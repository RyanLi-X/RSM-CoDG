import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CHANNELS = 62
NUM_BANDS = 5
FEATURE_DIM = NUM_CHANNELS * NUM_BANDS

CHANNEL_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]

# Six functional regions of the 62-channel SEED montage.
BRAIN_REGION_CHANNELS = {
    'frontal':        ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"],
    'central':        ["FC3", "FC1", "FCZ", "FC2", "FC4", "C3", "C1", "CZ", "C2", "C4"],
    'parietal':       ["CP3", "CP1", "CPZ", "CP2", "CP4", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"],
    'temporal_left':  ["FT7", "FC5", "T7", "C5", "TP7", "CP5"],
    'temporal_right': ["FC6", "FT8", "C6", "T8", "CP6", "TP8"],
    'occipital':      ["PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"],
}
BRAIN_REGIONS = {name: sorted(CHANNEL_NAMES.index(ch) for ch in channels)
                 for name, channels in BRAIN_REGION_CHANNELS.items()}
assert sorted(i for idx in BRAIN_REGIONS.values() for i in idx) == list(range(NUM_CHANNELS)), \
    "BRAIN_REGIONS must partition all electrodes"


class GaussianNoise(nn.Module):
    """Additive Gaussian noise, active only in training mode."""

    def __init__(self, std=0.12):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class ElectrodewiseSubjectCalibration(nn.Module):
    """Subject-specific input calibration.
    """

    def __init__(self, num_source_subjects, num_channels=NUM_CHANNELS, num_bands=NUM_BANDS):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.weight = nn.Parameter(torch.eye(num_bands).repeat(num_source_subjects, num_channels, 1, 1))

    def forward(self, x, subject_ids=None):
        b, t, _ = x.shape
        nodes = x.reshape(b, t, self.num_channels, self.num_bands)
        if self.training:
            if subject_ids is None:
                raise ValueError("subject_ids are required in training mode.")
            out = torch.einsum('btnd,bndk->btnk', nodes, self.weight.index_select(0, subject_ids))
        else:
            out = torch.einsum('btnd,ndk->btnk', nodes, self.weight.mean(dim=0))
        return out.reshape(b, t, -1)


class RGRM(nn.Module):
    """Region-aware Graph Representation Module.
    """

    def __init__(self, num_bands=NUM_BANDS, hidden_dim=64, feature_dim=FEATURE_DIM):
        super().__init__()
        for name, channels in BRAIN_REGIONS.items():
            self.register_buffer(f'{name}_indices', torch.tensor(channels, dtype=torch.long))
        self.W_q = nn.Linear(num_bands, hidden_dim)
        self.W_k = nn.Linear(num_bands, hidden_dim)
        self.W_v = nn.Linear(num_bands, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # RCB weight is sigmoid(alpha)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        self.attention_head = nn.Linear(feature_dim, feature_dim)

    def regions(self):
        return [getattr(self, f'{name}_indices') for name in BRAIN_REGIONS]

    def regional_continuous_branch(self, V):
        """RCB: every electrode receives the mean value feature of its region."""
        out = torch.zeros_like(V)
        for idx in self.regions():
            out[:, idx, :] = V[:, idx, :].mean(dim=1, keepdim=True).expand(-1, len(idx), -1)
        return out

    def regional_sparse_branch(self, Q, K, V):
        """RSB: every electrode receives the value feature of the most similar other electrode
        (top-1 query-key score) in its region."""
        out = torch.zeros_like(V)
        batch_idx = torch.arange(V.size(0), device=V.device)
        for idx in self.regions():
            scores = torch.bmm(Q[:, idx, :], K[:, idx, :].transpose(1, 2))
            self_mask = torch.eye(len(idx), dtype=torch.bool, device=V.device)
            nearest = scores.masked_fill(self_mask, float('-inf')).max(dim=2)[1]  # (B, |region|)
            for i, channel in enumerate(idx):
                out[:, channel, :] = V[batch_idx, idx[nearest[:, i]], :]
        return out

    def forward(self, x):
        b, t, _ = x.shape
        nodes = x.reshape(b, t, NUM_CHANNELS, NUM_BANDS)
        enhanced, attention = [], []
        for step in range(t):
            x_t = nodes[:, step]
            q, k, v = self.W_q(x_t), self.W_k(x_t), self.W_v(x_t)
            alpha = torch.sigmoid(self.alpha)
            fused = alpha * self.regional_continuous_branch(v) + (1 - alpha) * self.regional_sparse_branch(q, k, v)
            x_hat = x[:, step] + self.output_projection(fused.mean(dim=1))
            enhanced.append(x_hat)
            attention.append(torch.sigmoid(self.attention_head(x_hat)))
        return torch.stack(enhanced, dim=1), torch.stack(attention, dim=1).mean(dim=1)


class MSTT(nn.Module):
    """Multi-Scale Temporal Transformer.

    A local branch (each step attends to a +-window_size neighborhood) and a global branch
    (each step attends to itself and every sparse_period-th step) are fused and pooled into
    one vector by temporal attention.
    """

    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, num_heads=8, window_size=5, sparse_period=3):
        super().__init__()
        self.window_size = window_size
        self.sparse_period = sparse_period
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.local_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._masks = {}

    def mask(self, kind, seq_len, device):
        key = (kind, seq_len, device)
        if key not in self._masks:
            pos = torch.arange(seq_len, device=device)
            if kind == 'local':
                allowed = (pos[:, None] - pos[None, :]).abs() <= self.window_size
            else:
                allowed = (pos[:, None] == pos[None, :]) | (pos[None, :] % self.sparse_period == 0)
            self._masks[key] = torch.zeros(seq_len, seq_len, device=device).masked_fill(~allowed, float('-inf'))
        return self._masks[key]

    def forward(self, x):
        h = self.input_projection(x)
        t = h.size(1)
        local_out, _ = self.local_attention(h, h, h, attn_mask=self.mask('local', t, h.device))
        global_out, _ = self.global_attention(h, h, h, attn_mask=self.mask('global', t, h.device))
        fused = self.fusion(torch.cat([local_out, global_out], dim=-1))
        temporal_attention = F.softmax(self.attention_pool(fused), dim=1).squeeze(-1)  # (B, T)
        z = torch.sum(fused * temporal_attention.unsqueeze(-1), dim=1)
        return z, temporal_attention


class CoDG(nn.Module):
    """Collaboratively Optimized Domain Generalization.
    """

    def __init__(self, feature_dim=64, spatial_dim=FEATURE_DIM, temporal_dim=30, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64))
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64))
        self.invariant_extractor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(feature_dim, feature_dim))
        self.projector = nn.Linear(feature_dim, feature_dim, bias=False)

    @staticmethod
    def pairwise_sq_dist(x):
        sq_norm = (x ** 2).sum(dim=1).view(-1, 1)
        return torch.clamp(sq_norm + sq_norm.t() - 2.0 * torch.mm(x, x.t()), min=0.0)

    @staticmethod
    def rbf_kernel(x, y, sigma):
        dist = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
        return torch.exp(-dist / (2 * sigma ** 2))

    def decorrelation_loss(self, features):
        """Squared Frobenius norm of the off-diagonal batch covariance."""
        centered = features - features.mean(dim=0, keepdim=True)
        cov = torch.matmul(centered.T, centered) / (features.size(0) - 1)
        off_diag = 1 - torch.eye(cov.size(0), device=features.device)
        return torch.norm(cov * off_diag, p='fro') ** 2

    def mmd_loss(self, features, subject_ids):
        """RBF-MMD averaged over all training-subject pairs (median-heuristic bandwidth)."""
        subjects = torch.unique(subject_ids)
        if len(subjects) < 2:
            return features.new_zeros(())
        total, count = 0.0, 0
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                x = features[subject_ids == subjects[i]]
                y = features[subject_ids == subjects[j]]
                dist = self.pairwise_sq_dist(torch.cat([x, y], dim=0))
                dist = dist[dist > 0]
                sigma = torch.sqrt(torch.median(dist)) if len(dist) > 0 else features.new_tensor(1.0)
                total += (self.rbf_kernel(x, x, sigma).mean() + self.rbf_kernel(y, y, sigma).mean()
                          - 2 * self.rbf_kernel(x, y, sigma).mean())
                count += 1
        return total / count

    def contrastive_loss(self, embeddings, labels):
        """Supervised contrastive loss: samples with the same emotion label are positives."""
        n = embeddings.size(0)
        embeddings = F.normalize(embeddings, dim=1)
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        not_self = torch.ones_like(logits) - torch.eye(n, device=embeddings.device)
        labels = labels.contiguous().view(-1, 1)
        positives = torch.eq(labels, labels.T).float() * not_self
        log_prob = logits - torch.log((torch.exp(logits) * not_self).sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (positives * log_prob).sum(dim=1) / (positives.sum(dim=1) + 1e-8)
        return -mean_log_prob_pos.mean()

    def forward(self, z, spatial_attention, temporal_attention, subject_ids=None, labels=None):
        f_proj = self.projector(self.invariant_extractor(z))
        e_attn = torch.cat([self.spatial_encoder(spatial_attention),
                            self.temporal_encoder(temporal_attention)], dim=-1)
        losses = {}
        if self.training:
            losses['decorrelation'] = self.decorrelation_loss(f_proj)
            if subject_ids is not None:
                losses['mmd'] = self.mmd_loss(f_proj, subject_ids)
            if labels is not None:
                losses['contrastive'] = self.contrastive_loss(e_attn, labels)
        return f_proj, losses


class RSMCoDG(nn.Module):

    def __init__(self, num_source_subjects, num_classes=3, time_steps=30, dropout_rate=0.4, use_calibration=True):
        super().__init__()
        self.calibration = ElectrodewiseSubjectCalibration(num_source_subjects) if use_calibration else None
        self.noise = GaussianNoise(std=0.12)
        self.rgrm = RGRM(hidden_dim=64)
        self.mstt = MSTT(hidden_dim=64)
        self.codg = CoDG(feature_dim=64, temporal_dim=time_steps)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(64, 32, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, subject_ids=None, labels=None, apply_noise=False):
        """Returns class logits and a dict of CoDG losses (empty in eval mode)."""
        if self.calibration is not None:
            x = self.calibration(x, subject_ids)
        if apply_noise:
            x = self.noise(x)
        x_hat, spatial_attention = self.rgrm(x)
        z, temporal_attention = self.mstt(x_hat)
        features, dg_losses = self.codg(z, spatial_attention, temporal_attention, subject_ids, labels)
        return self.classifier(features), dg_losses
