import torch.nn as nn

class MetadataEmbedding(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, meta):
        # meta shape: [B, 2]
        return self.mlp(meta)