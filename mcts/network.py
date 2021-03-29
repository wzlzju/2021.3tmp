import torch.nn as nn
import torch.nn.functional as F


def get_conv(dim_in, dim_out):
    layers = [
        nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(dim_out),
        nn.ReLU(inplace=True)
    ]
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim_out),
        )

    def forward(self, x):
        return self.block(x) + x


class Net(nn.Module):
    def __init__(self, value_branch=False):
        super(Net, self).__init__()
        self.input_len = 17190
        self.dim_in = 1
        self.dim_hidden = 256
        self.dim_out = 600
        self.num_res_block = 1
        self.value_branch = value_branch

        # network structure
        self.conv = get_conv(self.dim_in, self.dim_hidden)
        self.res_blocks = nn.Sequential(
            *[ResBlock(self.dim_hidden, self.dim_hidden) for i in range(self.num_res_block)]
        )
        self.policy_head_1 = nn.Sequential(
            nn.Conv1d(self.dim_hidden, 2, kernel_size=1, stride=1),
            nn.BatchNorm1d(2),
            nn.ReLU(),
        )
        self.policy_head_2 = nn.Linear(self.input_len * 2, self.dim_out)

        if self.value_branch:
            self.value_head_1 = nn.Sequential(
                nn.Conv1d(self.dim_hidden, 1, kernel_size=1, stride=1),
                nn.BatchNorm1d(1),
                nn.ReLU(),
            )
            self.value_head_2 = nn.Sequential(
                nn.Linear(self.input_len, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Tanh()
            )

    def forward(self, x, softmax=True):
        x = self.conv(x)
        x = self.res_blocks(x)

        policy = self.policy_head_1(x)
        policy = policy.view(policy.size()[0], -1)
        policy = self.policy_head_2(policy)


        if self.value_branch:
            value = self.value_head_1(x)
            value = value.view(value.size()[0], -1)
            value = self.value_head_2(value)

            return policy, value
        else:
            return policy


class LinearNet(nn.Module):
    def __init__(self, value_branch=False, in_dim=784, n_hidden_1=300, n_hidden_2=100, out_dim=10):
        super(LinearNet, self).__init__()
        self.value_branch = value_branch

        self.net = nn.Sequential(
                    nn.Linear(19732, 400),
                    nn.ReLU(),
                    nn.Linear(400, 200),
                    nn.ReLU(),
                    nn.Linear(200, 400),
                    nn.ReLU(),
                    nn.Linear(400, 600),
                )

    def forward(self, x):
        x = x.view(-1, 19732)
        x = self.net(x)

        return x
