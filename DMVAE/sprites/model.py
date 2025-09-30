import sys

import torch
import torch.nn as nn

sys.path.append("../")
import probtorch
from probtorch.util import expand_inputs
from util import kaiming_init

EPS = 1e-9
TEMP = 0.66

layers = [1024, 512, 512, 2048, 1024, 512]


class EncoderA(nn.Module):
    def __init__(self, seed, zShared_dim=10, zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.seed = seed

        self.fc = nn.Sequential()
        self.fc.append(nn.Linear(64 * 64 * 3, layers[0], bias=False))
        self.fc.append(nn.ReLU(True))
        for i in range(1, len(layers)):
            self.fc.append(nn.Linear(layers[i - 1], layers[i], bias=False))
            self.fc.append(nn.ReLU(True))
        self.fc.append(
            nn.Linear(layers[-1], 2 * zPrivate_dim + 2 * zShared_dim, bias=False)
        )

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        stats = self.fc(x)
        stats = stats.unsqueeze(0)
        muPrivate = stats[:, :, : self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim : (2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)

        muShared = stats[
            :, :, (2 * self.zPrivate_dim) : (2 * self.zPrivate_dim + self.zShared_dim)
        ]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim) :]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muPrivate, scale=stdPrivate, name="privateA")

        q.normal(loc=muShared, scale=stdShared, name="sharedA")

        return q


class DecoderA(nn.Module):
    def __init__(self, seed, zShared_dim=10, zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.num_digits = zShared_dim
        self.seed = seed

        self.dec_hidden = nn.Sequential()
        self.dec_hidden.append(
            nn.Linear(zPrivate_dim + zShared_dim, layers[-1], bias=False)
        )
        self.dec_hidden.append(nn.ReLU(True))
        for i in range(1, len(layers)):
            self.dec_hidden.append(
                nn.Linear(layers[::-1][i - 1], layers[::-1][i], bias=False)
            )
            self.dec_hidden.append(nn.ReLU(True))
        self.dec_hidden.append(nn.Linear(layers[0], 64 * 64 * 3, bias=False))

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        private_mean = torch.zeros_like(q["privateA"].dist.loc)
        private_std = torch.ones_like(q["privateA"].dist.scale)
        shared_mean = torch.zeros_like(q["sharedA"].dist.loc)
        shared_std = torch.ones_like(q["sharedA"].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(
            private_mean, private_std, value=q["privateA"], name="privateA"
        )
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(
                shared_mean, shared_std, value=shared[shared_name], name=shared_name
            )

            recon = self.dec_hidden(torch.cat([zPrivate, zShared], -1))

            # TODO: change this to gaussian NLL
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(
                lambda x_hat, x: ((x - x_hat) ** 2).sum(-1),
                recon,
                images,
                name="images1_" + shared_name,
            )
        return p

    def forward2(self, z):
        recon = self.dec_hidden(z)

        return recon


class EncoderB(nn.Module):
    def __init__(self, seed, zShared_dim=10, zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.seed = seed

        self.fc = nn.Sequential()
        self.fc.append(nn.Linear(64 * 64 * 3, layers[0], bias=False))
        self.fc.append(nn.ReLU(True))
        for i in range(1, len(layers)):
            self.fc.append(nn.Linear(layers[i - 1], layers[i], bias=False))
            self.fc.append(nn.ReLU(True))
        self.fc.append(
            nn.Linear(layers[-1], 2 * zPrivate_dim + 2 * zShared_dim, bias=False)
        )

        self.weight_init()

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        stats = self.fc(x)
        stats = stats.unsqueeze(0)
        muPrivate = stats[:, :, : self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim : (2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)

        muShared = stats[
            :, :, (2 * self.zPrivate_dim) : (2 * self.zPrivate_dim + self.zShared_dim)
        ]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim) :]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muPrivate, scale=stdPrivate, name="privateB")

        q.normal(loc=muShared, scale=stdShared, name="sharedB")

        return q


class DecoderB(nn.Module):
    def __init__(self, seed, zShared_dim=10, zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.num_digits = zShared_dim
        self.seed = seed

        self.dec_hidden = nn.Sequential()
        self.dec_hidden.append(
            nn.Linear(zPrivate_dim + zShared_dim, layers[-1], bias=False)
        )
        self.dec_hidden.append(nn.ReLU(True))
        for i in range(1, len(layers)):
            self.dec_hidden.append(
                nn.Linear(layers[::-1][i - 1], layers[::-1][i], bias=False)
            )
            self.dec_hidden.append(nn.ReLU(True))
        self.dec_hidden.append(nn.Linear(layers[0], 64 * 64 * 3, bias=False))

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        private_mean = torch.zeros_like(q["privateB"].dist.loc)
        private_std = torch.ones_like(q["privateB"].dist.scale)
        shared_mean = torch.zeros_like(q["sharedB"].dist.loc)
        shared_std = torch.ones_like(q["sharedB"].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(
            private_mean, private_std, value=q["privateB"], name="privateB"
        )
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(
                shared_mean, shared_std, value=shared[shared_name], name=shared_name
            )

            recon = self.dec_hidden(torch.cat([zPrivate, zShared], -1))

            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(
                lambda x_hat, x: ((x - x_hat) ** 2).sum(-1),
                recon,
                images,
                name="images2_" + shared_name,
            )
        return p

    def forward2(self, z):
        recon = self.dec_hidden(z)

        return recon
