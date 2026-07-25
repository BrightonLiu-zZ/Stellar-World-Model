from __future__ import annotations

import torch
import torch.nn as nn


class Dynamics(nn.Module):
    """
    GRU latent-dynamics predictor, the term that makes Variant B a world model.
    Given the latent means of the first T-1 windows in a sequence it predicts the
    latent of the next window at each step (one-step teacher forcing). A linear head
    maps the GRU hidden state back to latent space. The prediction target is supplied
    stop-gradient'd by the caller, so only this prediction side learns.
    """

    def __init__(self, z_dim: int, gru_hidden: int, gru_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(z_dim, gru_hidden, num_layers=gru_layers, batch_first=True)
        self.head = nn.Linear(gru_hidden, z_dim)

    def forward(self, mu_seq_in: torch.Tensor) -> torch.Tensor:
        # mu_seq_in: (B, T-1, z) -- the inputs z_1..z_{T-1}
        h, _ = self.gru(mu_seq_in) # (B, T-1, gru_hidden)
        pred_next = self.head(h) # (B, T-1, z) -- predicted z_2..z_T
        return pred_next

    def rollout(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Free-running multi-step rollout (exp05 multistep mode): from the real z_1 only, predict
        z_2, then feed the PREDICTION back to predict z_3, ... up to z_{n_steps+1}. No teacher
        forcing -- errors compound, so this is a much harder objective than one-step forward and
        matches the rollout-vs-persistence eval. Only z0 is a real latent; every later input is the
        model's own output, so gradient flows through the predictions into the GRU/head and back to
        the encoder via z0. The caller stop-gradients the targets.
        """
        # z0: (B, z) -- the real first-window latent z_1
        inp = z0.unsqueeze(1) # (B, 1, z)
        h_state = None # GRU hidden carried across steps so context accumulates
        preds = []
        for _ in range(n_steps):
            out, h_state = self.gru(inp, h_state) # (B, 1, gru_hidden)
            step = self.head(out[:, -1, :]) # (B, z) -- predicted next latent
            preds.append(step)
            inp = step.unsqueeze(1) # feed the prediction back (free-running)
        return torch.stack(preds, dim=1) # (B, n_steps, z) -- predicted z_2..z_{n_steps+1}
