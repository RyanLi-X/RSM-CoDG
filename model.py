import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Channel mapping aligned to 6 brain regions; ensure channel order matches CHANNEL_NAMES
CHANNEL_NAMES = [
"FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
"FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4","C6","T8",
"TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8",
"PO7","PO5","PO3","POZ","PO4","PO6","PO8","CB1","O1","OZ","O2","CB2"
]

# Six-region partition covering all 62 channels
BRAIN_REGIONS = {
    'frontal':        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],            # FP*, AF*, F*
    'temporal_left':  [14, 23, 32],                                               # FT7, T7, TP7
    'central':        [15,16,17,18,19,20,21, 24,25,26,27,28,29,30],               # FC*, C*
    'temporal_right': [22, 31, 40],                                               # FT8, T8, TP8
    'parietal':       [33,34,35,36,37,38,39, 41,42,43,44,45,46,47,48,49],         # CP*, P*
    'occipital':      [50,51,52,53,54,55,56,57,58,59,60,61]                       # PO*, O*, CB*
}
# Generate name mapping for visualization
BRAIN_REGIONS_NAMES = {k: [CHANNEL_NAMES[i] for i in v] for k, v in BRAIN_REGIONS.items()}

def create_region_mask():
    """Create 62x62 mask based on region membership"""
    mask = torch.zeros(len(CHANNEL_NAMES), len(CHANNEL_NAMES))
    for region_channels in BRAIN_REGIONS.values():
        for i in region_channels:
            for j in region_channels:
                mask[i, j] = 1
    return mask

# Aggregate 310-d spatial attention into 6 regions
def aggregate_spatial_attention_to_regions(spatial_attention, agg='mean'):
    """
    Args:
        spatial_attention: torch.Tensor, shape (batch, 310) or (310,)
        agg: 'mean' or 'sum'  -- aggregation over channels/bands
    Returns:
        region_scores: torch.Tensor, shape (batch, n_regions)
        region_band: torch.Tensor, shape (batch, n_regions, 5)
    Usage:
        region_scores, region_band = aggregate_spatial_attention_to_regions(att_tensor)
    """
    single = False
    if spatial_attention.dim() == 1:
        spatial_attention = spatial_attention.unsqueeze(0)
        single = True
    b = spatial_attention.size(0)
    if spatial_attention.size(1) != 310:
        raise ValueError(f"Expected spatial_attention dim=310, got {spatial_attention.size(1)}")
    # reshape -> (batch, 62, 5)
    ch_band = spatial_attention.view(b, 62, 5)
    region_band_list = []
    for region in BRAIN_REGIONS.keys():
        idxs = BRAIN_REGIONS[region]
        if len(idxs) == 0:
            # Keep shape aligned
            region_band_list.append(torch.zeros((b, 5), device=spatial_attention.device))
            continue
        sel = ch_band[:, idxs, :]  # (b, n_ch_region, 5)
        region_band = sel.mean(dim=1)  # (b, 5)
        region_band_list.append(region_band)
    region_band = torch.stack(region_band_list, dim=1)  # (b, n_regions, 5)
    if agg == 'mean':
        region_scores = region_band.mean(dim=2)  # mean over bands -> (b, n_regions)
    elif agg == 'sum':
        region_scores = region_band.sum(dim=2)
    else:
        raise ValueError("agg must be 'mean' or 'sum'")
    if single:
        region_scores = region_scores.squeeze(0)
        region_band = region_band.squeeze(0)
    return region_scores, region_band
# ------------------------------------------------------------------------------

# Gaussian noise injector
class GaussianNoise(nn.Module):
    """Inject Gaussian noise during training to improve robustness"""
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std
    
    def forward(self, x):
        # Add noise only in training
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

# Region-aware Graph Representation Module (RGRM)
class RegionAwareGraphModule(nn.Module):
    """Region-aware graph representation module"""
    def __init__(self, input_dim=5, hidden_dim=64):
        super(RegionAwareGraphModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Register region mask
        self.register_buffer('region_mask', create_region_mask())
        # Register indices per region
        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 0:
                self.register_buffer(f'{region_name}_indices', torch.tensor(channels, dtype=torch.long))
        # Linear projections for attention
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        # Learnable branch weights
        self.alpha_cont = nn.Parameter(torch.tensor(0.5))
        self.alpha_sparse = nn.Parameter(torch.tensor(0.5))
        
        # Project back to 310-dim
        self.output_projection = nn.Linear(hidden_dim, 310)
        
        # Attention extractor
        self.attention_weights_extractor = nn.Linear(310, 310)
  
    def regional_continuous_attention(self, Q, K, V):
        """Regional continuous branch using in-region averaging"""
        batch_size, num_channels, dim = Q.shape
        output = torch.zeros_like(V)
        
        # Process each region
        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 0:
                region_indices = getattr(self, f'{region_name}_indices')
                region_values = V[:, region_indices, :]
                # Average within region
                avg_values = region_values.mean(dim=1, keepdim=True)
                output[:, region_indices, :] = avg_values.expand(-1, len(channels), -1)
        return output
    
    def regional_sparse_attention(self, Q, K, V):
        """Regional sparse branch selecting the most relevant channel per region"""
        batch_size, num_channels, dim = Q.shape
        output = torch.zeros_like(V)
        
        # Process each region
        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 1:
                region_indices = getattr(self, f'{region_name}_indices')
                region_Q = Q[:, region_indices, :]
                region_K = K[:, region_indices, :]
                region_V = V[:, region_indices, :]
                # Similarity within region
                similarity = torch.bmm(region_Q, region_K.transpose(1, 2))
                # Mask self-connections
                mask = torch.eye(len(channels), device=Q.device, dtype=torch.bool).unsqueeze(0)
                similarity.masked_fill_(mask, float('-inf'))
                
                # Select most similar channel
                _, max_indices = torch.max(similarity, dim=2)
                
                batch_indices = torch.arange(batch_size, device=Q.device).unsqueeze(1)
                channel_indices = torch.arange(len(channels), device=Q.device).unsqueeze(0)
                selected_indices = max_indices[batch_indices, channel_indices]
                
                # Update outputs per selected index
                for i, channel_idx in enumerate(region_indices):
                    # map local selected index to global channel index via region_indices
                    output[:, channel_idx, :] = V[batch_indices.squeeze(), region_indices[selected_indices[:, i]], :]
                        
            elif len(channels) == 1:
                # Single-channel region
                region_indices = getattr(self, f'{region_name}_indices')
                output[:, region_indices, :] = V[:, region_indices, :]
        return output
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        Args:
            x: input features (batch_size, time_steps, 310)
            return_attention: whether to return attention weights
        Returns:
            enhanced_sequence: enhanced features (batch_size, time_steps, 310)
            (optional) aggregated_attention: averaged attention (batch_size, 310)
        """
        batch_size, time_steps, feature_dim = x.shape
        
        if feature_dim != 310:
            raise ValueError(f"Expected feature dimension 310, got {feature_dim}")
        # Reshape 310-d features to (62 channels, 5 bands)
        x_reshaped = x.view(batch_size, time_steps, 62, 5)
        # Iterate over time steps
        enhanced_features = []
        attention_weights_list = []
        
        for t in range(time_steps):
            x_t = x_reshaped[:, t, :, :]
    
            # Attention projections
            Q = self.W_q(x_t)
            K = self.W_k(x_t)
            V = self.W_v(x_t)
            
            # Branch outputs
            cont_output = self.regional_continuous_attention(Q, K, V) # continuous
            sparse_output = self.regional_sparse_attention(Q, K, V) # sparse
            
            # Normalize branch weights and fuse
            alpha_cont_norm = torch.sigmoid(self.alpha_cont)
            alpha_sparse_norm = 1 - alpha_cont_norm
            
            fused_output = alpha_cont_norm * cont_output + alpha_sparse_norm * sparse_output
            
            # Pool and project back to 310-d, then add residual
            pooled_graph_feature = fused_output.mean(dim=1)
            projected_feature = self.output_projection(pooled_graph_feature)
            enhanced_feature = x[:, t, :] + projected_feature
            enhanced_features.append(enhanced_feature)
            if return_attention:
                attention_weights = torch.sigmoid(self.attention_weights_extractor(enhanced_feature))
                attention_weights_list.append(attention_weights)
        
        # Stack time steps
        enhanced_sequence = torch.stack(enhanced_features, dim=1)
        
        if return_attention:
            attention_weights_tensor = torch.stack(attention_weights_list, dim=1)
            # Aggregate attention over time
            aggregated_attention = attention_weights_tensor.mean(dim=1)  # (batch, 310)
            return enhanced_sequence, aggregated_attention
        
        return enhanced_sequence

# Multi-Scale Temporal Transformer (MSTT)
class MultiScaleTemporalTransformer(nn.Module):
    """Multi-scale temporal modeling module."""
    def __init__(self, input_dim=310, hidden_dim=64, num_heads=8, window_size=5, sparse_period=3):
        super(MultiScaleTemporalTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.sparse_period = sparse_period
        
        # Project input to hidden dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # Local Continuous Attention
        self.local_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        # Global Sparse Attention
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Fusion layer
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        # Attention pooling for sequence representation
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._cached_masks = {}
        
    def get_local_mask(self, seq_len, device):
        """Local attention mask limited to a temporal window."""
        cache_key = f"local_{seq_len}_{device}"
        if cache_key not in self._cached_masks:
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
            for i in range(seq_len):
                start = max(0, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)
                mask[i, start:end] = 0
            self._cached_masks[cache_key] = mask
        return self._cached_masks[cache_key]
    
    def get_sparse_mask(self, seq_len, device):
        """Sparse attention mask focusing on periodic steps."""
        cache_key = f"sparse_{seq_len}_{device}"
        if cache_key not in self._cached_masks:
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
            for i in range(seq_len):
                mask[i, i] = 0
                for j in range(0, seq_len, self.sparse_period):
                    mask[i, j] = 0
            self._cached_masks[cache_key] = mask
        return self._cached_masks[cache_key] 
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        Args:
            x: input features (batch_size, seq_len, 310)
            return_attention: whether to return attention weights
        Returns:
            sequence_repr: fused representation (batch_size, hidden_dim)
            (optional) attention_weights: pooled attention weights
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Project to hidden dim
        x_proj = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Build masks
        local_mask = self.get_local_mask(seq_len, x.device)
        sparse_mask = self.get_sparse_mask(seq_len, x.device)
        # Local attention
        local_out, local_attn_weights = self.local_attention(x_proj, x_proj, x_proj, attn_mask=local_mask)
        # Global attention
        global_out, global_attn_weights = self.global_attention(x_proj, x_proj, x_proj, attn_mask=sparse_mask)
        # Fuse outputs
        fused_features = self.fusion(torch.cat([local_out, global_out], dim=-1))
        # Attention pooling
        attention_scores = self.attention_pool(fused_features)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1).squeeze(-1)  # (batch_size, seq_len)
        # Weighted sum
        sequence_repr = torch.sum(fused_features * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)
        
        if return_attention:
            return sequence_repr, attention_weights
        
        return sequence_repr #(batch_size, hidden_dim)

# Collaborative Domain Generalization module
class CollaborativeDomainGeneralization(nn.Module):
    """Domain generalization head (CoDG)."""
    def __init__(self, feature_dim=64, spatial_attention_dim=310, 
                 temporal_attention_dim=30, temperature=0.07): # seed3=30
        super(CollaborativeDomainGeneralization, self).__init__()
        
        self.temperature = temperature
        
        # Spatial attention encoder
        self.spatial_attention_encoder = nn.Sequential(
            nn.Linear(spatial_attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Temporal attention encoder
        self.temporal_attention_encoder = nn.Sequential(
            nn.Linear(temporal_attention_dim, 128), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Domain-invariant projector
        self.domain_invariant_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Orthogonal projector for feature decoupling
        self.orthogonal_projector = nn.Linear(feature_dim, feature_dim, bias=False)
        
    def compute_contrastive_loss(self, attention_emb, subject_ids):
        """InfoNCE-based attention contrastive loss for subject consistency."""
        attention_emb_norm = F.normalize(attention_emb, dim=1)
        sim_matrix = torch.matmul(attention_emb_norm, attention_emb_norm.T) / self.temperature
        
        # Positive mask for same subject
        subject_mask = (subject_ids.unsqueeze(0) == subject_ids.unsqueeze(1)).float()
        subject_mask.fill_diagonal_(0)  # exclude self
        
        # InfoNCE loss
        pos_sim = sim_matrix * subject_mask
        neg_sim = sim_matrix * (1 - subject_mask)
        pos_sim = pos_sim.sum(dim=1) / (subject_mask.sum(dim=1) + 1e-8)
        neg_sim = torch.logsumexp(neg_sim, dim=1)
        
        contrastive_loss = -pos_sim + neg_sim
        return contrastive_loss.mean()
    
    def compute_mmd_loss(self, features, subject_ids):
        """MMD loss to encourage domain-invariant features across subjects."""
        unique_subjects = torch.unique(subject_ids)
        if len(unique_subjects) < 2:
            return torch.tensor(0.0, device=features.device)
        
        mmd_loss = 0
        count = 0
        
        for i, subject_i in enumerate(unique_subjects):
            for j, subject_j in enumerate(unique_subjects):
                if i >= j:
                    continue
                    
                features_i = features[subject_ids == subject_i]
                features_j = features[subject_ids == subject_j]
                
                if len(features_i) > 0 and len(features_j) > 0:
                    # Simplified MMD
                    mean_i = features_i.mean(dim=0)
                    mean_j = features_j.mean(dim=0)
                    mmd_loss += torch.norm(mean_i - mean_j, p=2)
                    count += 1
        
        return mmd_loss / max(count, 1)
    
    
    def compute_orthogonal_loss(self, features):
        """Orthogonality loss to keep feature dimensions independent."""
        # Correlation matrix
        features_centered = features - features.mean(dim=0, keepdim=True)
        correlation_matrix = torch.matmul(features_centered.T, features_centered) / (features.size(0) - 1)
        
        # Encourage off-diagonal terms to zero
        identity = torch.eye(correlation_matrix.size(0), device=features.device)
        orthogonal_loss = torch.norm(correlation_matrix * (1 - identity), p='fro') ** 2
        
        return orthogonal_loss
    
    def forward(self, features, spatial_attention_weights, temporal_attention_weights, 
                subject_ids=None, labels=None):
        """
        Domain generalization forward
        Args:
            features: MSTT output (batch_size, feature_dim)
            spatial_attention_weights: RGRM attention weights (batch_size, 310)
            temporal_attention_weights: MSTT attention weights (batch_size, time_steps)
            subject_ids: optional subject IDs
            labels: optional emotion labels
        Returns:
            orthogonal_features: decoupled features
            losses: DG loss dict
        """
        batch_size = features.size(0)
        
        # 1. Domain-invariant projection
        domain_invariant_features = self.domain_invariant_projector(features)
        # 2. Feature decoupling
        orthogonal_features = self.orthogonal_projector(domain_invariant_features)
        # 3. Encode attention patterns
        spatial_attention_emb = self.spatial_attention_encoder(spatial_attention_weights)
        temporal_attention_emb = self.temporal_attention_encoder(temporal_attention_weights)
        attention_fusion = torch.cat([spatial_attention_emb, temporal_attention_emb], dim=-1)
        # 4. Compute DG losses
        losses = {}
        if self.training:
            # Feature orthogonality
            orthogonal_loss = self.compute_orthogonal_loss(orthogonal_features)
            losses['feature_orthogonal'] = orthogonal_loss
            if subject_ids is not None:
                # Subject-level contrastive loss on attention embeddings
                contrastive_loss = self.compute_contrastive_loss(attention_fusion, subject_ids)
                losses['attention_contrastive'] = contrastive_loss
                # MMD domain alignment
                mmd_loss = self.compute_mmd_loss(orthogonal_features, subject_ids)
                losses['feature_mmd'] = mmd_loss
        
        return orthogonal_features, losses

# Unified EEG emotion model
class RSMCoDGModel(nn.Module):
    """RSM-CoDG main model for EEG emotion recognition."""
    def __init__(self, cuda, number_of_category=3, dropout_rate=0.4, time_steps=30): # seed3=30
        super(RSMCoDGModel, self).__init__()
        # Noise injection
        self.noise_layer = GaussianNoise(std=0.12)
        # Region-aware graph module (RGRM)
        self.region_graph_module = RegionAwareGraphModule(input_dim=5, hidden_dim=64)
        # Multi-scale temporal transformer (MSTT)
        self.temporal_transformer = MultiScaleTemporalTransformer(input_dim=310, hidden_dim=64)
        # Collaborative domain generalization (CoDG)
        self.codg = CollaborativeDomainGeneralization(
            feature_dim=64,
            spatial_attention_dim=310,
            temporal_attention_dim=time_steps  # based on dataset setting
        )
        # Emotion classifier
        self.cls_fc = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, number_of_category, bias=True)
        )

    def forward(self, x, subject_ids=None, labels=None, apply_noise=False, return_feats=False):
        """
        Forward pass
        Args:
            x: input features (batch_size, time_steps, 310)
            subject_ids: subject IDs
            labels: emotion labels
            apply_noise: apply noise during training
            return_feats: if True, also return intermediate outputs
        Returns:
            x_pred: log probabilities
            x_logits: classification logits
            dg_losses: DG loss dict
            cls_loss: classification loss (or None)
            if return_feats is True, also returns domain_generalized_features, spatial_attention_weights, temporal_attention_weights
        """
        # Noise injection
        if apply_noise and self.training:
            x = self.noise_layer(x)
        # Region-aware enhancement (RGRM)
        x_enhanced, spatial_attention_weights = self.region_graph_module(x, return_attention=True)
        # Multi-scale temporal modeling (MSTT)
        sequence_repr, temporal_attention_weights = self.temporal_transformer(x_enhanced, return_attention=True)
        # Collaborative domain generalization (CoDG)
        domain_generalized_features, dg_losses = self.codg(
            sequence_repr, spatial_attention_weights, temporal_attention_weights, 
            subject_ids, labels
        )
        # Emotion classification
        x_logits = self.cls_fc(domain_generalized_features)
        x_pred = F.log_softmax(x_logits, dim=1)  # log_softmax for classification
        
        # Compute classification loss when labels are provided
        cls_loss = None
        if labels is not None:
            cls_loss = F.nll_loss(x_pred, labels)
        
        if return_feats:
            return x_pred, x_logits, dg_losses, cls_loss, domain_generalized_features, spatial_attention_weights, temporal_attention_weights
        
        return x_pred, x_logits, dg_losses, cls_loss

# Unified EEG test model
class RSMCoDGTestModel(nn.Module):
    """Test model without noise injection or DG losses."""
    def __init__(self, baseModel):
        super(RSMCoDGTestModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)

    def forward(self, x):
        """Forward pass for testing without noise or labels."""
        x_pred, x_logits, _, _ = self.baseModel(x, subject_ids=None, labels=None, apply_noise=False)
        return x_logits

# Helper to create model
def create_rsm_codg_model(num_classes=3, dropout_rate=0.4):
    """Convenience wrapper to build RSM-CoDG model."""
    return RSMCoDGModel(
        cuda=True, 
        number_of_category=num_classes, 
        dropout_rate=dropout_rate
    )

# Helper to compute total loss
def calculate_total_loss(cls_loss, dg_losses, loss_weights=None):
    """
    Combine classification and DG losses.
    """
    if loss_weights is None:
        loss_weights = {
            'feature_orthogonal': 0.05,
            'attention_contrastive': 0.2,
            'feature_mmd': 0.15
        }
    
    total_loss = cls_loss
    
    for loss_name, loss_value in dg_losses.items():
        if loss_name in loss_weights:
            total_loss += loss_weights[loss_name] * loss_value
    
    return total_loss
