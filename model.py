import torch
from torch import nn
import configs
from mipm import MetaboliteInteractionPerception, DropPath
from kan import KAN


class METRON(nn.Module):
    def __init__(self, dropout=configs.dropout, droprate=configs.droprate, no_off=False) -> None:
        super().__init__()
        self.n_channels = 1
        self.input_embedding_dim = 34
        self.dropout = dropout
        self.droprate = droprate
        self.no_off = no_off
        self.layer_norm = nn.LayerNorm(self.input_embedding_dim)
        self.mipm = MetaboliteInteractionPerception(self.input_embedding_dim, self.n_channels, no_off=self.no_off)
        self.kan = KAN(self.input_embedding_dim)
        self.drop_path1d = DropPath(self.droprate) if self.droprate > 0.0 else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(self.input_embedding_dim, 32),
            nn.LeakyReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x_norm = self.layer_norm(x)
        x0 = x_norm
        x_1d = self.mipm(x0)
        x_1d = self.drop_path1d(x_1d) + x0
        x0 = x_1d
        x_1d = self.kan(x_1d)
        x_1d = self.drop_path1d(x_1d) + x0
        x_fc = self.fc(x_1d)
        out = self.projection(x_fc)

        return out


def get_model():
    model = METRON()
    return model.to(configs.DEVICE)