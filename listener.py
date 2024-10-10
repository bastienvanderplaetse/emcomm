import torch
from torch import nn
from weightinit import custom_init

class Listener(nn.Module):
    def __init__(self, msg_length, vocab_size, emb_dim, emb_img, core_state_dim, target_proj_dim, core_state_proj_dim, transform_func) -> None:
        super().__init__()

        self.length = msg_length
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.emb_img = emb_img
        self.core_state_dim = core_state_dim
        self.target_proj_dim = target_proj_dim
        self.core_state_proj_dim = core_state_proj_dim

        self.embedder = nn.Embedding(self.vocab_size, self.emb_dim)
        self.mlp = nn.Linear(self.emb_dim, self.emb_dim)

        self.lstm = nn.LSTM(self.length, self.core_state_dim//2, batch_first=True)

        self.target_proj = nn.Linear(self.emb_img, self.target_proj_dim)
        if transform_func == 'custom':
            custom_init(self.target_proj.weight.data)
        self.core_state_proj = nn.Linear(self.core_state_dim, self.core_state_proj_dim)

        self.relu = nn.ReLU()

    
    def forward(self, x, action):
        batch_size = x.shape[0]

        embedded_msg = self.embedder(action)

        output, (h_n, c_n) = self.lstm(embedded_msg)
        core_state = torch.concat((h_n, c_n), axis=-1).squeeze()

        predictions = self.core_state_proj(core_state)
        targets = self.target_proj(x)

        return predictions, targets

    def image_proj(self, x):
        return self.target_proj(x)

    def message_proj(self, action):
        embedded_msg = self.embedder(action)
        output, (h_n, c_n) = self.lstm(embedded_msg)
        core_state = torch.concat((h_n, c_n), axis=-1).squeeze()

        predictions = self.core_state_proj(core_state)

        return predictions