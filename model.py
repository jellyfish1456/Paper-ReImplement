import torch.nn as nn


class Score(nn.Module):
    def __init__(self):
        super().__init__()

        nef = 32
        self.u_net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(3, nef, 16, stride=2, padding=2),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.Conv2d(nef, nef * 2, 4, stride=2, padding=2),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.Conv2d(nef * 2, nef * 4, 5, stride=1, padding=2),
            nn.GroupNorm(4, nef * 4),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*4) x 3 x 3
            nn.ConvTranspose2d(nef * 4, nef * 2, 5, stride=1, padding=1),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.ConvTranspose2d(nef * 2, nef, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.ConvTranspose2d(nef, 3, 4, stride=2, padding=1),
            # nn.Softplus()

            nn.ELU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 3 * 32 * 32)
        )

    def forward(self, x):

        score = self.u_net(x)
        score = self.fc(score.view(x.shape[0], -1)).view(
            x.shape[0], 3, 32, 32)

        return score
