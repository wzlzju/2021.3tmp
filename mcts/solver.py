import network
import torch
from torch import optim
import torch.nn.functional as F


class Solver:
    def __init__(self, use_gpu=False):
        # parameters
        self.regularization = 0.00001
        self.learn_rate = 0.01
        self.value_branch = False
        self.use_gpu = use_gpu
        self.gpu = 3

        self.model = network.Net(value_branch=self.value_branch)
        # self.optimizer = optim.SGD(self.model.parameters(),lr=self.learn_rate, momentum=0.9, weight_decay=self.regularization)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        self.ce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss()
        self.mse = torch.nn.MSELoss()
        self.myCe = lambda x, y: -torch.mean(y * F.log_softmax(x, dim=1))

        if self.use_gpu:
            self.set_gpu(self.use_gpu, gpu=self.gpu)

    def set_gpu(self, use_gpu, gpu=0):
        """
        use this function to change to gpu mode
        """
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = use_gpu
            self.gpu = gpu

            torch.cuda.set_device(device=self.gpu)
            self.model = self.model.cuda()

    def train(self, data, vector_label, scalar_label=None):
        # data = torch.tensor(data, dtype=torch.float32)
        # vector_label = torch.tensor(vector_label, dtype=torch.float32)
        if self.value_branch:
            scalar_label = torch.tensor(scalar_label, dtype=torch.float32)
        if self.use_gpu:
            data = data.cuda()
            vector_label = vector_label.cuda()
            if self.value_branch:
                scalar_label = scalar_label.cuda()

        self.model.train()

        if self.value_branch:
            vector_pred, scalar_pred = self.model(data)
            scalar_pred = scalar_pred.squeeze()
            loss2 = self.mse(scalar_pred, scalar_label)
        else:
            vector_pred = self.model(data)

        loss1 = self.myCe(vector_pred, vector_label)

        # print(scalar_pred, scalar_label, loss2)
        # print(vector_pred.max(dim=1)[1], vector_label, loss1)
        if self.value_branch:
            loss = loss1 + loss2
        else:
            loss = loss1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()

    def test(self, data):
        # input shape: numpy array [1, 17191]
        # output shape: policy=numpy array [600]
        if self.use_gpu:
            data = data.cuda()

        self.model.eval()

        if self.value_branch:
            vector_pred, scalar_pred = self.model(data.unsqueeze(dim=0))
        else:
            vector_pred = self.model(data.unsqueeze(dim=0))

        # policy = F.softmax(vector_pred, dim=1).
        policy = vector_pred
        policy = policy[0].detach().cpu().numpy()

        if self.value_branch:
            value = scalar_pred[0].cpu().item()
            return policy, value
        else:
            return policy

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
