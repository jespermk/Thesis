import torch.nn as nn
import torch.nn.functional as F
import torch as t
from VI_models.simple_models import radial, MFG
import math
import numpy as np
from torch.autograd import Variable

def maxpool2d(x, kernel, step):

    x = x.unfold(3, kernel, step)
    x = x.unfold(4, kernel, step)
    x = x.max(6)[0].max(5)[0]

    return x

def sigma_T(rho):
    return t.log(1 + t.exp(rho))


class unit_log_MFG(nn.Module):
    def __init__(self, device):
        super(unit_log_MFG, self).__init__()
        pi = math.pi
        self.pi = t.tensor(pi).to(device)

    def forward(self, x):

        return - t.log(2*self.pi) * x.size()[1] / 2 - t.sum(x**2) / 2.



class NN(nn.Module):
    def __init__(self, s):
        super(NN, self).__init__()

        self.s = s
        self.mat_index = t.tensor([])
        self.bias_index = t.tensor([])
        self.m = []

        stop = 0
        start = 0

        for i in range(self.s.size()[0]):

            self.m += [nn.Linear(self.s[i, 0], self.s[i, 1])]

            stop += self.s[i, 0] * self.s[i, 1]

            self.mat_index = t.cat((self.mat_index, t.tensor([ [start, stop] ])), 0)

            start += self.s[i, 0] * self.s[i, 1]
            stop += self.s[i, 1]

            self.bias_index = t.cat((self.bias_index, t.tensor([[start, stop]])), 0)

            start += self.s[i, 1]

        self.mat_index = self.mat_index.long()
        self.bias_index = self.bias_index.long()


    def evl(self, x, p):


        for i in range(self.s.size()[0]):


            w = t.reshape(p[self.mat_index[i, 0]: self.mat_index[i, 1]], (self.s[i, 1], self.s[i, 0]))
            b = p[self.bias_index[i,0]:self.bias_index[i,1]]

            self.m[i].weight = nn.Parameter(w)
            self.m[i].bias = nn.Parameter(b)

            x = F.relu( self.m[i](x) )
            # we dont want the relu transform in the last layer

        return x

class radial_aff(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(radial_aff, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.parameter_dim = in_dim * out_dim + out_dim

        self.w_param_index = in_dim * out_dim

        k = t.sqrt(t.tensor(1.92 / in_dim))

        self.mu_w = nn.Parameter(t.randn(out_dim * in_dim) * k)
        self.rho_w = nn.Parameter(t.randn(out_dim * in_dim) * 0.5)

        self.mu_b = nn.Parameter(t.randn(out_dim) * k)
        self.rho_b = nn.Parameter(t.randn(out_dim) * 0.5)

        self.prior = unit_log_MFG(device)

    def forward(self, x, e):

        s_n = e.size()[0]

        w_e = e[:, 0: self.w_param_index]
        b_e = e[:, self.w_param_index: self.parameter_dim]

        w = self.mu_w + sigma_T(self.rho_w) * w_e
        b = self.mu_b + sigma_T(self.rho_b) * b_e

        entropy_q = t.sum(t.log(sigma_T(self.rho_w))) + t.sum(t.log(sigma_T(self.rho_b)))

        cross_entropy_qp = t.mean( self.prior(w) ) + t.mean( self.prior(b) )

        kl_qp = - entropy_q - cross_entropy_qp

        w = t.reshape(w, (s_n, self.out_dim, self.in_dim))
        x_w = t.einsum('spi,poi->spo', x, w)
        x_w_b = x_w + b

        return x_w_b, kl_qp



class radialNN(nn.Module):
    def __init__(self,device):
        super(radialNN, self).__init__()

        self.layers = nn.ModuleList(
            [radial_aff(784, 200, device),
             radial_aff(200, 200, device),
             radial_aff(200, 10, device)
             ])

        layer_parameter = [m.parameter_dim for m in self.layers]
        self.index = []
        n = 0

        for i in layer_parameter:
            self.index += [n]
            n += i

        self.index += [n]

        self.e = radial(n, device)

    def forward(self, x):
        # set up
        e_ = self.e.radial_normalized_sample(100)
        kl_qp_sum = 0
        index = 0

        x = x.unsqueeze(1)

        x = t.flatten(x, start_dim=2)

        e = e_[:, self.index[index]: self.index[index + 1]]
        x, kl_qp = self.layers[index](x, e)
        x = F.relu(x)
        kl_qp_sum += kl_qp
        index += 1

        e = e_[:, self.index[index]: self.index[index + 1]]
        x, kl_qp = self.layers[index](x, e)
        x = F.relu(x)
        kl_qp_sum += kl_qp
        index += 1

        e = e_[:, self.index[index]: self.index[index + 1]]
        x, kl_qp = self.layers[index](x, e)
        x = F.log_softmax(x, dim=-1)
        kl_qp_sum += kl_qp
        index += 1

        return x, kl_qp_sum

class radial_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, device):
        super(radial_conv2d, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel = kernel
        self.w_param_index = out_channels * in_channels * kernel**2

        self.parameter_dim = out_channels * in_channels * kernel**2 + out_channels

        k = t.sqrt(t.tensor(1/(in_channels * kernel ** 2) ) )

        self.mu_w = nn.Parameter( t.rand(out_channels * in_channels * kernel ** 2)*k - 0.5*k )
        self.rho_w = nn.Parameter( t.randn(out_channels * in_channels * kernel ** 2)*0.5 )

        self.mu_b = nn.Parameter( t.rand(out_channels)*k - 0.5*k )
        self.rho_b = nn.Parameter(t.randn(out_channels)*0.5)

        self.prior = unit_log_MFG(device)

    def forward(self, x, e):

        s_n = e.size()[0]

        w_e = e[:, 0: self.w_param_index]
        b_e = e[:, self.w_param_index: self.parameter_dim]

        w = self.mu_w + sigma_T(self.rho_w) * w_e
        b = self.mu_b + sigma_T(self.rho_b) * b_e
        entropy_q = t.sum(t.log(sigma_T(self.rho_w))) + t.sum(t.log(sigma_T(self.rho_b)))

        cross_entropy_qp = t.mean(self.prior(w)) + t.mean(self.prior(b))

        kl_qp = - entropy_q - cross_entropy_qp

        w = t.reshape(w, (s_n, self.out_channels, self.in_channels, self.kernel, self.kernel))

        x = x.unfold(3, self.kernel, 1)
        x = x.unfold(4, self.kernel, 1)

        x_w = t.einsum('ijklmno,jpkno->ijplm', x, w)
        x_w_b = x_w + b.unsqueeze(0).unsqueeze(3).unsqueeze(4)

        return x_w_b, kl_qp



class image_model(nn.Module):
    def __init__(self, VI_model, device):
        super(image_model, self).__init__()


        self.layers = nn.ModuleList(
            [radial_conv2d(1, 20, 5, device),
            radial_conv2d(20, 50, 5, device),
            radial_aff(4*4*50, 500, device),
            radial_aff(500, 10, device)
        ])


        layer_parameter = [m.parameter_dim for m in self.layers]
        self.index = []
        n = 0

        for i in layer_parameter:
            self.index += [n]
            n += i

        self.index += [n]

        self.e = VI_model(n, device)

    def forward(self, x, train=False):

        # set up
        e_ = self.e.normalized_sample(100)
        kl_qp_sum = 0
        index = 0

        x = x.unsqueeze(1)

        # layers
        e = e_[:, self.index[index]: self.index[index+1]]
        x, kl_qp = self.layers[index](x, e)
        x = F.relu(x)
        kl_qp_sum += kl_qp
        index += 1

        x = maxpool2d(x, 2, 2)

        e = e_[:, self.index[index]: self.index[index+1]]
        x, kl_qp = self.layers[index](x, e)
        x = F.relu(x)
        kl_qp_sum += kl_qp
        index += 1

        x = maxpool2d(x, 2, 2)

        x = t.flatten(x, start_dim= 2)

        e = e_[:, self.index[index]: self.index[index+1]]
        x, kl_qp = self.layers[index](x, e)
        x = F.relu(x)
        kl_qp_sum += kl_qp
        index += 1

        e = e_[:, self.index[index]: self.index[index+1]]
        x, kl_qp = self.layers[index](x, e)
        x = F.log_softmax(x, dim=-1)
        kl_qp_sum += kl_qp
        index += 1

        if train:
            return x, kl_qp_sum
        else:
            return x.mean(1)

class ensemble_model(nn.Module):
    def __init__(self, ensemble, device, weighted=False):
        super(ensemble_model, self).__init__()
        self.device = device
        self.weighted = weighted
        if weighted:
            self.w = nn.Parameter(t.ones(len(ensemble)))
        else:
            self.w = t.ones(len(ensemble), device=device)
        self.ensemble = ensemble
        [ensemble[i].to('cpu') for i in range(len(ensemble))]

    def forward(self, x):

        with t.no_grad():
            y = []
            for i in range(len(self.ensemble)):
                self.ensemble[i].to(self.device)
                y += [self.ensemble[i](x)]
                self.ensemble[i].to('cpu')

        x = t.stack(y).requires_grad_()

        w = self.w

        w = w / w.sum()

        x = t.einsum('mso,m->so', x, w)

        return x



# reformed

class vi_conv2d(nn.Module):
    def __init__(self, shape):
        super(vi_conv2d, self).__init__()

        self.shape = shape

    def forward(self, x, w, b):

        w = t.reshape(w, (w.size()[0], self.shape[1], self.shape[0], self.shape[2], self.shape[3]))

        x = x.unfold(3, self.shape[2], 1)
        x = x.unfold(4, self.shape[3], 1)

        x_w = t.einsum('ijklmno,jpkno->ijplm', x, w)
        x_w_b = x_w + b.unsqueeze(0).unsqueeze(3).unsqueeze(4)

        return x_w_b



class vi_aff(nn.Module):
    def __init__(self, shape):
        super(vi_aff, self).__init__()

        self.shape = shape

    def forward(self, x, w, b):

        w = t.reshape(w, (w.size()[0], self.shape[1], self.shape[0]))
        x_w = t.einsum('spi,poi->spo', [x, w])
        x_w_b = x_w + b

        return x_w_b



class vi_image_model(nn.Module):
    def __init__(self, vi_model, prior, device):
        super(vi_image_model, self).__init__()

        self.shape = [
            [1, 20, 5, 5],
            [20, 50, 5, 5],
            [4*4*50, 500],
            [500, 10]

        ]
        self.layer_kind = ['conv', 'conv', 'aff', 'aff']

        mus_w = []
        rhos_w = []
        mus_b = []
        rhos_b = []
        w_size = []
        b_size = []
        for i in range(len(self.layer_kind)):

            if self.layer_kind[i] == 'conv':

                p = self.shape[i]

                k = t.sqrt(t.tensor(1 / (p[0] * p[2] * p[3])))

                mus_w += [t.rand(p[1] * p[0] * p[2] * p[3]) * k - 0.5 * k]
                rhos_w += [t.ones(p[1] * p[0] * p[2] * p[3]) * 0.5]
                w_size += [p[1] * p[0] * p[2] * p[3]]

                mus_b += [t.rand(p[1]) * k - 0.5 * k]
                rhos_b += [t.ones(p[1]) * 0.5]
                b_size += [p[1]]

            elif self.layer_kind[i] == 'aff':

                p = self.shape[i]

                k = t.sqrt(t.tensor(1.92 / p[0]))

                mus_w += [t.randn(p[1] * p[0]) * k]
                rhos_w += [t.ones(p[1] * p[0]) * 0.5]
                w_size += [p[1] * p[0]]

                mus_b += [t.randn(p[1]) * k]
                rhos_b += [t.ones(p[1]) * 0.5]
                b_size += [p[1]]

            else:
                print('typo in parameter creation')

        w_index = [0]
        b_index = [0]

        w_sum = 0
        b_sum = 0

        for i in range(len(self.shape)):
            w_sum += w_size[i]
            b_sum += b_size[i]

            w_index += [w_sum]
            b_index += [b_sum]

        self.w_index = w_index
        self.b_index = b_index

        self.mu_w = nn.Parameter(t.cat(mus_w))
        self.rho_w = nn.Parameter(t.cat(rhos_w))

        self.mu_b = nn.Parameter(t.cat(mus_b))
        self.rho_b = nn.Parameter(t.cat(rhos_b))

        layers = []
        for i in range(len(self.layer_kind)):
            if self.layer_kind[i] == 'conv':
                layers += [vi_conv2d(self.shape[i])]
            elif self.layer_kind[i] == 'aff':
                layers += [vi_aff(self.shape[i])]
            else:
                print('typo in module creation')

        self.layers = nn.ModuleList(layers)

        dim = self.w_index[-1] + self.b_index[-1]

        self.vi_model = vi_model(dim, device)
        self.prior = prior(dim, device)


    def forward(self, x, train=False):
        # sample
        e = self.vi_model.normalized_sample(100)

        # kl_qp
        w = self.mu_w + sigma_T(self.rho_w) * e[:, :self.rho_w.size()[0]]
        b = self.mu_b + sigma_T(self.rho_b) * e[:, self.rho_w.size()[0]:]

        if train:

            entropy_q = self.vi_model.entropy(sigma_T(self.rho_w), sigma_T(self.rho_b))

            p = t.cat((w, b), dim=1)

            cross_entropy_qp = t.mean(self.prior.log_prob(p))

            kl_qp = entropy_q - cross_entropy_qp

        # for likelihood
        x = x.unsqueeze(1)
        i = 0


        # layers
        x = self.layers[i](x,
                          w[:, self.w_index[i]:self.w_index[i+1]],
                          b[:, self.b_index[i]:self.b_index[i+1]])
        x = F.relu(x)
        i += 1

        x = maxpool2d(x, 2, 2)

        x = self.layers[i](x,
                            w[:, self.w_index[i]:self.w_index[i+1]],
                            b[:, self.b_index[i]:self.b_index[i+1]])
        x = F.relu(x)
        i += 1

        x = maxpool2d(x, 2, 2)

        x = t.flatten(x, start_dim=2)

        x = self.layers[i](x,
                            w[:, self.w_index[i]:self.w_index[i+1]],
                            b[:, self.b_index[i]:self.b_index[i+1]])
        x = F.relu(x)
        i += 1

        x = self.layers[i](x,
                            w[:, self.w_index[i]:self.w_index[i+1]],
                            b[:, self.b_index[i]:self.b_index[i+1]])
        x = F.log_softmax(x, dim=-1)

        if train:
            return x, kl_qp
        else:
            return x.mean(1)

################

shape_default = [[1, 20, 5, 5],
                 [20, 50, 5, 5],
                 [4*4*50, 500],
                 [500, 10]]
kind_default = ['conv', 'conv', 'aff', 'aff']

class radial_VCL_image_model(nn.Module):
    def __init__(self, vi_model,
                 layer_kind=kind_default,
                 shape=shape_default,
                 device='cuda',
                 prior=None,
                 other_prior=None,
                 rho=-4,
                 vi_samples_n=100):
        super(radial_VCL_image_model, self).__init__()

        self.layer_kind = layer_kind
        self.shape = shape

        # Init parameters
        mus_w = []
        rhos_w = []
        mus_b = []
        rhos_b = []
        w_size = []
        b_size = []

        for i in range(len(self.layer_kind)):

            if self.layer_kind[i] == 'conv':

                p = self.shape[i]

                k = t.sqrt(t.tensor(1 / (p[0] * p[2] * p[3]), dtype=t.float64))

                mus_w += [t.rand(p[1] * p[0] * p[2] * p[3], dtype=t.float64) * k - 0.5 * k]
                rhos_w += [t.zeros(p[1] * p[0] * p[2] * p[3], dtype=t.float64) + rho]
                w_size += [p[1] * p[0] * p[2] * p[3]]

                mus_b += [t.rand(p[1], dtype=t.float64) * k - 0.5 * k]
                rhos_b += [t.zeros(p[1], dtype=t.float64) + rho]
                b_size += [p[1]]

            elif self.layer_kind[i] == 'aff':

                p = self.shape[i]

                k = t.sqrt(t.tensor(1.92, dtype=t.float64) / p[0])

                mus_w += [t.randn(p[1] * p[0], dtype=t.float64) * k]
                rhos_w += [t.zeros(p[1] * p[0], dtype=t.float64) + rho]
                w_size += [p[1] * p[0]]

                mus_b += [t.randn(p[1], dtype=t.float64) * k]
                rhos_b += [t.zeros(p[1], dtype=t.float64) + rho]
                b_size += [p[1]]

            else:
                print('typo in parameter creation')

        w_index = [0]
        b_index = [0]

        w_sum = 0
        b_sum = 0

        for i in range(len(self.shape)):
            w_sum += w_size[i]
            b_sum += b_size[i]

            w_index += [w_sum]
            b_index += [b_sum]

        self.w_index = w_index
        self.b_index = b_index

        # Temporarily store VI parameters for w and b
        mu_w_ = t.cat(mus_w)
        rho_w_ = t.cat(rhos_w)

        mu_b_ = t.cat(mus_b)
        rho_b_ = t.cat(rhos_b)

        # Init layers
        layers = []
        for i in range(len(self.layer_kind)):
            if self.layer_kind[i] == 'conv':
                layers += [vi_conv2d(self.shape[i])]
            elif self.layer_kind[i] == 'aff':
                layers += [vi_aff(self.shape[i])]
            else:
                print('typo in module creation')

        self.layers = nn.ModuleList(layers)

        self.dim = self.w_index[-1] + self.b_index[-1]

        self.device = device

        self.vi_model = vi_model(self.dim, self.device)
        self.vi_samples_n = vi_samples_n

        if other_prior:
            self.other_prior = other_prior(self.dim, self.device)
        else:
            self.other_prior = other_prior

        # Init prior parameters
        if not prior:

            self.mu_prior = t.zeros(self.dim, device=self.device, dtype=t.float64)
            self.rho_prior = t.ones(self.dim, device=self.device, dtype=t.float64)

            # Set VI parameters
            self.mu_w = nn.Parameter(mu_w_)
            self.rho_w = nn.Parameter(rho_w_)

            self.mu_b = nn.Parameter(mu_b_)
            self.rho_b = nn.Parameter(rho_b_)

        else:

            mu_w_prev, mu_b_prev = prior.mu_w.detach().requires_grad_(),\
                                   prior.mu_b.detach().requires_grad_()
            rho_w_prev, rho_b_prev = prior.rho_w.detach().requires_grad_(),\
                                     prior.rho_b.detach().requires_grad_()

            mu_prev = t.cat([mu_w_prev, mu_b_prev]).to(device)
            rho_prev = t.cat([rho_w_prev, rho_b_prev]).to(device)

            if self.dim > prior.dim:

                # Set prior
                mu_prior = t.zeros(self.dim - prior.dim, device=device, dtype=t.float64)
                rho_prior = t.ones(self.dim - prior.dim, device=device, dtype=t.float64)

                self.mu_prior = t.cat([mu_prev, mu_prior]).to(self.device)
                self.rho_prior = t.cat([rho_prev, rho_prior]).to(self.device)

                # Set VI parameters
                self.mu_w = nn.Parameter(t.cat([mu_w_prev,
                                                mu_w_[mu_w_prev.size(0):]])
                                         )
                self.rho_w = nn.Parameter(t.cat([rho_w_prev,
                                                 rho_w_[rho_w_prev.size(0):]])
                                          )

                self.mu_b = nn.Parameter(t.cat([mu_b_prev,
                                                mu_b_[mu_b_prev.size(0):]])
                                         )
                self.rho_b = nn.Parameter(t.cat([rho_b_prev,
                                                 rho_b_[rho_b_prev.size(0):]])
                                          )

            elif self.dim == prior.dim:

                # Set prior
                self.mu_prior = mu_prev.to(self.device).requires_grad_()
                self.rho_prior = rho_prev.to(self.device).requires_grad_()

                # Set VI parameters
                self.mu_w = nn.Parameter(mu_w_prev)
                self.rho_w = nn.Parameter(rho_w_prev)

                self.mu_b = nn.Parameter(mu_b_prev)
                self.rho_b = nn.Parameter(rho_b_prev)

            else:
                print('error in dimensionality of prior seems it is smaller that its prior')

            prior.to('cpu')


    def forward(self, x, train=False):
        # sample
        e = self.vi_model.normalized_sample(self.vi_samples_n)

        # kl_qp
        e_w, e_b = e.split([self.w_index[-1], self.b_index[-1]], dim=1)

        w = t.addcmul(self.mu_w.unsqueeze(0), self.sigma_w().unsqueeze(0), Variable(e_w))
        b = t.addcmul(self.mu_b.unsqueeze(0), self.sigma_b().unsqueeze(0), Variable(e_b))

        if train:

            p_sample = t.cat((w, b), dim=1)

            entropy = self.vi_model.entropy(self.sigma_w(), self.sigma_b())

            if self.other_prior:
                cross_entropy = self.other_prior.cross_entropy(p_sample,
                                                              self.mu_prior,
                                                              sigma_T(self.rho_prior))
            else:
                cross_entropy = self.vi_model.cross_entropy(p_sample,
                                                           self.mu_prior,
                                                           sigma_T(self.rho_prior))


            kl_qp = entropy - cross_entropy

        # for likelihood
        x = x.unsqueeze(dim=1)

        for i in range(len(self.layer_kind)):

            if self.layer_kind[i] == 'conv':

                x = self.layers[i](x,
                                   w[:, self.w_index[i]:self.w_index[i + 1]],
                                   b[:, self.b_index[i]:self.b_index[i + 1]])
                x = F.relu(x)
                x = maxpool2d(x, 2, 2)

            elif self.layer_kind[i] == 'aff':

                if self.layer_kind[i-1] == 'conv':
                    x = t.flatten(x, start_dim=2)

                x = self.layers[i](x,
                                   w[:, self.w_index[i]:self.w_index[i + 1]],
                                   b[:, self.b_index[i]:self.b_index[i + 1]])

                if i == len(self.layer_kind)-1:
                    x = x

                else:
                    x = F.relu(x)

            else:
                print('Error in layer kind text')

        if train:
            return x, kl_qp
        else:
            return x.mean(1)

    def sample_posterior(self, n):

        # sample
        e = self.vi_model.normalized_sample(n)

        # apply change of varible
        e_w, e_b = e.split([self.w_index[-1], self.b_index[-1]], dim=1)

        w = t.addcmul(self.mu_w.unsqueeze(0), self.sigma_w().unsqueeze(0), e_w)
        b = t.addcmul(self.mu_b.unsqueeze(0), self.sigma_b().unsqueeze(0), e_b)

        return t.cat((b, w), dim=1)

    def sigma_w(self):
        return t.log(1 + t.exp(self.rho_w))

    def sigma_b(self):
        return t.log(1 + t.exp(self.rho_b))


##################################################################################################################################################

class non_bayes_base(nn.Module):
    def __init__(self, shape, bias, std):
        super(non_bayes_base, self).__init__()

        self.shape = shape
        self.bias = bias


class non_bayes_conv(non_bayes_base):
    def __init__(self,
                 shape,
                 pad=0,
                 stride=1,
                 bn=False,
                 relu=False,
                 PreAc=False,
                 bias=False,
                 std=None):

        super(non_bayes_conv, self).__init__(t.tensor([shape[i] for i in [1, 0, 2, 3]]), bias, std)

        self.relu = relu
        self.pad = pad
        self.PreAc = PreAc
        self.conv = nn.Conv2d(
            self.shape[1], self.shape[0],
            self.shape[2:4],
            stride=stride,
            padding=pad,
            bias=self.bias,
            dtype=t.float64
        )
        if bn:
            if PreAc:
                self.bn = nn.BatchNorm2d(shape[0], dtype=t.float64)
            else:
                self.bn = nn.BatchNorm2d(shape[1], dtype=t.float64)
        else:
            self.bn = None

    def forward(self, x):

        if self.PreAc:
            if self.bn is not None:
                x = self.bn(x)
            if self.relu:
                x = F.relu(x)

        x = self.conv(x)

        if not self.PreAc:
            if self.bn is not None:
                x = self.bn(x)
            if self.relu:
                x = F.relu(x)

        return x

class non_bayes_aff(non_bayes_base):
    def __init__(self, shape, relu=False, bias=False, std=None):

        super(non_bayes_aff, self).__init__(t.tensor([shape[i] for i in [1, 0]]), bias, std)
        self.relu = relu

        self.aff = nn.Linear(
            self.shape[1],
            self.shape[0],
            bias=self.bias,
            dtype=t.float64
        )

    def forward(self, x):

        x = self.aff(x)

        if self.relu:
            x = F.relu(x)

        return x


#################################################################################################################################################

import radial_bnn_from_paper.radial_layers.variational_bayes as fac


class non_fac_base(nn.Module):
    def __init__(self, shape, bias, std, prior='gaus'):
        super(non_fac_base, self).__init__()

        if t.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device
        self.prior = prior
        self.shape = shape
        self.bias = bias
        self.std = std
        self.df = 3

        self.w = Variable(t.empty(*shape, dtype=t.float64, device=device))
        if self.bias:
            self.b = Variable(t.empty(shape[0], dtype=t.float64, device=device))

        self.register_parameter('w_mu', nn.Parameter(t.empty(*shape, dtype=t.float64)))
        self.register_parameter('w_rho', nn.Parameter(t.empty(*shape, dtype=t.float64)))
        if self.bias:
            self.register_parameter('b_mu', nn.Parameter(t.empty(shape[0], dtype=t.float64)))
            self.register_parameter('b_rho', nn.Parameter(t.empty(shape[0], dtype=t.float64)))

        self.w_mu_prior = t.zeros(*shape, device=device)
        self.w_sigma_prior = std * t.ones(*shape, device=device)
        if self.bias:
            self.b_mu_prior = t.zeros(shape[0], device=device)
            self.b_sigma_prior = std * t.ones(shape[0], device=device)

    def param(self):
        return (
            self.shape,
            [self.shape[0]]
        )

    def kl(self):
        entropy = t.sum(t.log(self.w_sigma()))

        if self.prior in {'gaus', 'radial'}:
            cross_entropy = 0.5 * (
                    (self.w_sigma() / self.w_sigma_prior).square().sum()
                    + ((self.w_mu - self.w_mu_prior) / self.w_sigma_prior).square().sum()
            )

        elif self.prior == 'exp':
            cross_entropy = ((self.w - self.w_mu_prior) / self.w_sigma_prior).norm(dim=-1).square().mean()

        elif self.prior == 't':
            cross_entropy = (
                    (self.df + 1) / 2
                    * t.log(1 + ((self.w - self.w_mu_prior) / (self.w_sigma_prior / math.sqrt(3) )).norm(dim=-1).square() / self.df)
            ).mean()

        if self.bias:

            entropy += t.sum(t.log(self.b_sigma()))
            if self.prior in {'gaus', 'radial'}:
                cross_entropy += 0.5 * (
                        (self.b_sigma() / self.b_sigma_prior).square().sum()
                        + ((self.b_mu - self.b_mu_prior) / self.b_sigma_prior).square().sum()
                )

            elif self.prior == 'exp':
                cross_entropy += ((self.b - self.b_mu_prior) / self.b_sigma_prior).norm(dim=-1).square().mean()

            elif self.prior == 't':
                cross_entropy += (
                ((self.df + 1) / 2)
                * t.log(1 + ((self.b - self.b_mu_prior) / (self.b_sigma_prior / math.sqrt(3) )).norm(dim=-1).square() / self.df)
                ).mean()


        return cross_entropy - entropy

    def param_num(self):

        if self.bias:
            return [
                self.shape.prod(),
                self.shape[0]
            ]
        else:
            return [
                self.shape.prod()
            ]

    def w_sigma(self):
        return t.log(1 + t.exp(self.w_rho))

    def b_sigma(self):
        return t.log(1 + t.exp(self.b_rho))

class non_fac_conv(non_fac_base):
    def __init__(self,
                 shape,
                 pad=None,
                 stride=1,
                 bn=False,
                 relu=False,
                 PreAc=False,
                 bias=False,
                 std=None):

        super(non_fac_conv, self).__init__(t.tensor([shape[i] for i in [1, 0, 2, 3]]), bias, std)
        self.reset_parameters()

        self.relu = relu
        self.pad = pad
        self.stride = stride
        self.PreAc = PreAc
        if bn:
            if PreAc:
                self.bn = nn.BatchNorm2d(shape[0], dtype=t.float64)
            else:
                self.bn = nn.BatchNorm2d(shape[1], dtype=t.float64)
        else:
            self.bn = None

    def forward(self, x, p):

        if x.dim() == 4:
            x = x.unsqueeze(dim=1)

        if self.PreAc:
            if self.bn is not None:
                n, s = x.shape[:2]
                x = t.flatten(x, start_dim=0, end_dim=1)
                x = self.bn(x)
                x = x.reshape([n, s, *x.shape[1:]])
            if self.relu:
                x = F.relu(x)


        if self.pad is not None:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        x = x.unfold(3, self.shape[2], self.stride)
        x = x.unfold(4, self.shape[3], self.stride)


        self.w = t.addcmul(
            self.w_mu,
            self.w_sigma(),
            Variable(p[0])
        )

        x = t.einsum('ijklmno,jpkno->ijplm', x, self.w)

        if self.bias:
            self.b = t.addcmul(
                self.b_mu,
                self.b_sigma(),
                Variable(p[1])
            )

            x += self.b.unsqueeze(0).unsqueeze(3).unsqueeze(4)

        if not self.PreAc:
            if self.bn is not None:
                n, s = x.shape[:2]
                x = t.flatten(x, start_dim=0, end_dim=1)
                x = self.bn(x)
                x = x.reshape([n, s, *x.shape[1:]])
            if self.relu:
                x = F.relu(x)
        return x

    def reset_parameters(self):

        self.w_rho.data.normal_(-4, std=0.5)
        nn.init.kaiming_uniform_(self.w_mu, math.sqrt(1.92))

        if self.bias:
            self.b_rho.data.normal_(-4, std=0.5)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_mu, -bound, bound)



class non_fac_aff(non_fac_base):
    def __init__(self, shape, relu=False, bias=False, std=None):

        super(non_fac_aff, self).__init__(t.tensor([shape[i] for i in [1, 0]]), bias, std)
        self.reset_parameters()
        self.relu = relu

    def forward(self, x, p):

        if x.ndim == 2:
            x = x.unsqueeze(dim=1)

        self.w = t.addcmul(
            self.w_mu,
            self.w_sigma(),
            Variable(p[0])
        )

        x = t.einsum('spi,poi->spo', x, self.w)

        if self.bias:
            self.b = t.addcmul(
                self.b_mu,
                self.b_sigma(),
                Variable(p[1])
            )

            x += self.b

        if self.relu:
            x = F.relu(x)

        return x

    def reset_parameters(self):
        std = math.sqrt(1.92 / self.w_mu.shape[1])

        self.w_rho.data.normal_(-4, std=0.5)
        self.w_mu.data.normal_(std=std)

        if self.bias:
            self.b_rho.data.normal_(-4, std=0.5)
            self.b_mu.data.normal_(std=std)

    def continual_learning(self, expantion=None):

        if expantion is None:
            self.w_mu_prior = self.w_mu.detach()
            self.w_sigma_prior = self.w_sigma().detach()
        else:
            self.w_mu_prior = t.concat(
                [self.w_mu.detach(), t.zeros([expantion, self.w_mu.size(1)], device=self.device)]
            )
            self.w_sigma_prior = t.concat(
                [self.w_sigma().detach(), t.ones([expantion, self.w_rho.size(1)], device=self.device) * self.std]
            )

        if self.bias:
            if expantion is None:
                self.b_mu_prior = self.b_mu.detach()
                self.b_sigma_prior = self.b_sigma().detach()
            else:
                self.b_mu_prior = t.concat(
                    [self.b_mu.detach(), t.zeros([expantion], device=self.device)]
                )
                self.b_sigma_prior = t.concat(
                    [self.b_sigma().detach(), t.ones([expantion], device=self.device) * self.std]
                )

        if expantion is not None:

            std = math.sqrt(1.92 / self.w_mu.shape[1])

            self.w_mu = nn.Parameter(t.concat(
                [self.w_mu.clone(), t.randn([expantion, self.w_mu.size(1)], device=self.device) * std]
            ))
            self.w_rho = nn.Parameter(t.concat(
                [self.w_rho.clone(), 0.5 * t.randn([expantion, self.w_mu.size(1)], device=self.device) - 4]
            ))

            if self.bias:
                self.b_mu = nn.Parameter(t.concat(
                    [self.b_mu.clone(), t.randn([expantion], device=self.device) * std]
                ))
                self.b_rho = nn.Parameter(t.concat(
                    [self.b_rho.clone(), 0.5 * t.randn([expantion], device=self.device) - 4]
                ))




class fac_conv(nn.Module):
    def __init__(self, shape, pad=0, stride=1, bn=False, relu=False, PreAc=False, bias=False, std=None):
        super(fac_conv, self).__init__()

        self.PreAc = PreAc
        self.relu = relu
        if bn:
            if PreAc:
                self.bn = nn.BatchNorm2d(shape[0], dtype=t.float64)
            else:
                self.bn = nn.BatchNorm2d(shape[1], dtype=t.float64)
        else:
            self.bn = None

        self.conv = fac.SVIConv2D(
            in_channels=shape[0],
            out_channels=shape[1],
            kernel_size=shape[2:],
            variational_distribution='radial',
            prior={"name": "gaussian_prior",
                   "sigma": std,
                   "mu": 0},
            initial_rho=-4,
            mu_std="he",
            stride=(stride, stride),
            padding=pad,
            bias=bias,
        )


    def forward(self, x, vi_sample_n):

        if x.dim() == 4:
            x = x.unsqueeze(dim=1)

        if self.PreAc:
            if self.bn is not None:
                n, s = x.shape[:2]
                x = t.flatten(x, start_dim=0, end_dim=1)
                x = self.bn(x)
                x = x.reshape([n, s, *x.shape[1:]])

            if self.relu:
                x = F.relu(x)

        x = self.conv(x, vi_sample_n)

        if not self.PreAc:
            if self.bn is not None:
                n, s = x.shape[:2]
                x = t.flatten(x, start_dim=0, end_dim=1)
                x = self.bn(x)
                x = x.reshape([n, s, *x.shape[1:]])

            if self.relu:
                x = F.relu(x)

        return x

    def kl(self):

        cross_entropy_sum = 0
        entropy_sum = 0

        for module in self.modules():
            if hasattr(module, "cross_entropy"):
                cross_entropy_sum += module.cross_entropy()
            if hasattr(module, "entropy"):
                entropy_sum += module.entropy()

        return cross_entropy_sum - entropy_sum


class fac_aff(nn.Module):
    def __init__(self, shape, relu=False, bias=False, std=None):
        super(fac_aff, self).__init__()
        self.relu = relu
        self.std = std
        self.cl = False
        self.bias = bias

        self.aff = fac.SVI_Linear(
            in_features=shape[0],
            out_features=shape[1],
            initial_rho=-4,
            initial_mu='he',
            variational_distribution='radial',
            prior={"name": "gaussian_prior",
                   "sigma": std,
                   "mu": 0,},
            use_bias=bias,
        )

        if t.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.device = device

        self.w_mu_prior = t.zeros(*shape, device=device)
        self.w_sigma_prior = std * t.ones(*shape, device=device)
        if self.bias:
            self.b_mu_prior = t.zeros(shape[0], device=device)
            self.b_sigma_prior = std * t.ones(shape[0], device=device)

    def forward(self, x, vi_sample_n):

        if x.ndim == 2:
            x = x.unsqueeze(dim=1)

        x = self.aff(x, vi_sample_n)

        if self.relu:
            x = F.relu(x)

        return x

    def kl(self):
        if self.cl:
            entropy = t.sum(t.log(sigma_T(self.aff.weight_rhos)))
            cross_entropy = 0.5 * ((self.aff.weight - self.w_mu_prior) / self.w_sigma_prior).norm(dim=-1).square().mean()


            if self.bias:
                entropy += t.sum(t.log(sigma_T(self.aff.bias_rhos)))
                cross_entropy += 0.5 * ((self.aff.bias - self.b_mu_prior) / self.b_sigma_prior).norm(dim=-1).square().mean()

            return cross_entropy - entropy

        else:
            cross_entropy_sum = 0
            entropy_sum = 0

            for module in self.modules():
                if hasattr(module, "cross_entropy"):
                    cross_entropy_sum += module.cross_entropy()
                if hasattr(module, "entropy"):
                    entropy_sum += module.entropy()

            return cross_entropy_sum - entropy_sum

    def continual_learning(self, expantion=None):
        self.cl = True

        if expantion is None:
            self.w_mu_prior = self.aff.weight_mus.detach()
            self.w_sigma_prior = sigma_T(self.aff.weight_rhos.detach())
        else:
            self.w_mu_prior = t.concat(
                [self.aff.weight_mus.detach(), t.zeros([expantion, self.aff.weight_mus.shape[1]], device=self.device)]
            )
            self.w_sigma_prior = t.concat(
                [sigma_T(self.aff.weight_rhos.detach()), t.ones([expantion, self.aff.weight_rhos.shape[1]], device=self.device) * self.std]
            )

        if self.bias:
            if expantion is None:
                self.b_mu_prior = self.aff.bias_mus.detach()
                self.b_sigma_prior = self.aff.bias_rhos.detach()
            else:
                self.b_mu_prior = t.concat(
                    [self.aff.bias_mus.detach(), t.zeros([expantion], device=self.device)]
                )
                self.b_sigma_prior = t.concat(
                    [sigma_T(self.aff.bias_rhos.detach()), t.ones([expantion], device=self.device) * self.std]
                )

        if expantion is not None:

            std = math.sqrt(1.92 / self.aff.weight_mus.shape[1])

            self.aff.weight_mus = nn.Parameter(t.concat(
                [self.aff.weight_mus.clone(), t.randn([expantion, self.aff.weight_mus.shape[1]], device=self.device) * std]
            ))
            self.aff.weight_rhos = nn.Parameter(t.concat(
                [self.aff.weight_rhos.clone(), 0.5 * t.randn([expantion, self.aff.weight_rhos.shape[1]], device=self.device) - 4]
            ))

            if self.bias:
                self.aff.bias_mus = nn.Parameter(t.concat(
                    [self.aff.bias_mus.detach().clone(), t.randn([expantion], device=self.device) * std]
                ))
                self.aff.bias_rhos = nn.Parameter(t.concat(
                    [self.aff.bias_rhos.detach().clone(), 0.5 * t.randn([expantion], device=self.device) - 4]
                ))



class AvgPool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(AvgPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        if stride == None:
            self.stride = kernel_size
        else:
            self.stride = stride
        if dilation != 1:
            raise NotImplementedError

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = x.unfold(3, self.kernel_size, self.stride)
        x = x.unfold(4, self.kernel_size, self.stride) #  Now this is [examples, samples, channels, pooled_H, pooled_W, size_H, size_W)
        x = x.mean(6).mean(5)
        return x

class MaxPool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(MaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        if stride == None:
            self.stride = kernel_size
        else:
            self.stride = stride
        if dilation != 1:
            raise NotImplementedError

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = x.unfold(3, self.kernel_size, self.stride)
        x = x.unfold(4, self.kernel_size,
                     self.stride)  # Now this is [examples, samples, channels, pooled_H, pooled_W, size_H, size_W)
        x = x.max(6)[0].max(5)[0]
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, inference=None, PreAc=False, std=None):
        super(Block, self).__init__()

        self.PreAc = PreAc
        self.shortcut = True if downsample or in_channels != out_channels else False
        self.inference = inference

        if inference == 'non_bayes':
            conv = non_bayes_conv
        elif inference == 'fac':
            conv = fac_conv
        else:
            conv = non_fac_conv

        if self.shortcut:
            shortcut = nn.ModuleList(
                [
                    conv([in_channels, out_channels, 1, 1], stride=2, std=std,
                         bn=True),
                ]
            )

        block = nn.ModuleList(
            [
                conv([in_channels, out_channels, 3, 3], stride=2 if downsample else 1, pad=1, std=std,
                     bn=True,
                     relu=True,
                     PreAc=PreAc),
                conv([out_channels, out_channels, 3, 3], stride=1, pad=1, std=std,
                     bn=True,
                     relu=PreAc,
                     PreAc=PreAc)
            ]
        )

        if self.shortcut:
            self.block = nn.ModuleList([block, shortcut])
        else:
            self.block = nn.ModuleList([block])

    def forward(self, x, p_block, vi_sample_n):
        if self.inference == 'non_bayes':
            if self.shortcut:
                x_temp = x
                for f in self.block[1]:
                    x_temp = f(x_temp)
                x_shortcut = x_temp
            else:
                x_shortcut = x

            for f in self.block[0]:
                x = f(x)

        elif self.inference == 'fac':
            if self.shortcut:
                x_temp = x
                for f in self.block[1]:
                    x_temp = f(x_temp, vi_sample_n)
                x_shortcut = x_temp
            else:
                x_shortcut = x

            for f in self.block[0]:
                x = f(x, vi_sample_n)

        else:
            if self.shortcut:
                x_temp = x
                for f, p in zip(self.block[1], p_block[1]):
                    x_temp = f(x_temp, p)
                x_shortcut = x_temp
            else:
                x_shortcut = x

            for f, p in zip(self.block[0], p_block[0]):
                x = f(x, p)
        if self.PreAc:
            return x + x_shortcut
        else:
            return F.relu(x + x_shortcut)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, inference=None, PreAc=False, std=None):
        super(Bottleneck, self).__init__()

        self.PreAc = PreAc
        self.shortcut = True if downsample or in_channels != out_channels else False
        self.inference = inference

        if inference == 'non_bayes':
            conv = non_bayes_conv
        elif inference == 'fac':
            conv = fac_conv
        else:
            conv = non_fac_conv

        if self.shortcut:
            shortcut = nn.ModuleList(
                [
                    conv([in_channels, out_channels, 1, 1], stride=2 if downsample else 1, std=std,
                         bn=True),
                ]
            )

        block = nn.ModuleList(
            [
                conv([in_channels, out_channels // 4, 1, 1], stride=1, std=std,
                     bn=True,
                     relu=True,
                     PreAc=PreAc),
                conv([out_channels // 4, out_channels // 4, 3, 3], stride=2 if downsample else 1, pad=1, std=std,
                     bn=True,
                     relu=True,
                     PreAc=PreAc),
                conv([out_channels // 4, out_channels, 1, 1], stride=1, std=std,
                     bn=True,
                     relu=True if PreAc else False,
                     PreAc=PreAc),
            ]
        )
        if self.shortcut:
            self.block = nn.ModuleList([block, shortcut])
        else:
            self.block = nn.ModuleList([block])

    def forward(self, x, p_block, vi_sample_n):
        if self.inference == 'non_bayes':
            if self.shortcut:
                x_temp = x
                for f in self.block[1]:
                    x_temp = f(x_temp)
                x_shortcut = x_temp
            else:
                x_shortcut = x

            for f in self.block[0]:
                x = f(x)

        elif self.inference == 'fac':
            if self.shortcut:
                x_temp = x
                for f in self.block[1]:
                    x_temp = f(x_temp, vi_sample_n)
                x_shortcut = x_temp
            else:
                x_shortcut = x

            for f in self.block[0]:
                x = f(x, vi_sample_n)

        else:
            if self.shortcut:
                x_temp = x
                for f, p in zip(self.block[1], p_block[1]):
                    x_temp = f(x_temp, p)
                x_shortcut = x_temp
            else:
                x_shortcut = x

            for f, p in zip(self.block[0], p_block[0]):
                x = f(x, p)

        if self.PreAc:
            return x + x_shortcut
        else:
            return F.relu(x + x_shortcut)



class ResNet(nn.Module):

    def __init__(self,
                 vi_model=None,
                 device='cuda',
                 vi_samples_n=100,
                 std=1,
                 c_in=None,
                 out_dim=10,
                 resblock=None,
                 repeats=None,
                 PreAc=False,):

        super(ResNet, self).__init__()

        self.PreAc = PreAc
        if vi_model == 'non_bayes':
            aff = non_bayes_aff
            conv = non_bayes_conv
            inference_model = vi_model
        elif vi_model == 'fac':
            aff = fac_aff
            conv = fac_conv
            inference_model = vi_model
        else:
            aff = non_fac_aff
            conv = non_fac_conv
            inference_model = None

        if resblock == Bottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        zero = nn.ModuleList([
          nn.ModuleList([
            nn.ModuleList([
              conv([c_in, 64, 3, 3], stride=1, pad=1, std=std,
                   bn=True if not PreAc else False,
                   relu=True if not PreAc else False,
                   PreAc=PreAc)])])])

        one_ = [resblock(
            filters[0], filters[1],
            downsample=False, std=std,
            inference=inference_model,
            PreAc=PreAc)]
        for repetitions in range(repeats[0] - 1):
            one_ += [resblock(
                filters[1], filters[1],
                downsample=False,
                inference=inference_model, std=std,
                PreAc=PreAc)]

        one = nn.ModuleList(one_)

        two_ = [resblock(
            filters[1], filters[2],
            downsample=True, std=std,
            inference=inference_model,
            PreAc=PreAc)]
        for repetitions in range(repeats[1] - 1):
            two_ += [resblock(
                filters[2], filters[2],
                downsample=False,
                inference=inference_model, std=std,
                PreAc=PreAc)]

        two = nn.ModuleList(two_)

        three_ = [resblock(
            filters[2], filters[3],
            downsample=True, std=std,
            inference=inference_model,
            PreAc=PreAc)]
        for repetitions in range(repeats[2] - 1):
            three_ += [resblock(
                filters[3], filters[3],
                downsample=False,
                inference=inference_model, std=std,
                PreAc=PreAc)]

        three = nn.ModuleList(three_)

        four_ = [resblock(
            filters[3], filters[4],
            downsample=True, std=std,
            inference=inference_model,
            PreAc=PreAc)]
        for repetitions in range(repeats[3] - 1):
            four_ += [resblock(
                filters[4], filters[4],
                downsample=False,
                inference=inference_model, std=std,
                PreAc=PreAc)]

        four = nn.ModuleList(four_)

        self.five_pre = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if vi_model == 'non_bayes' else AvgPool2D(4),
            nn.Flatten(start_dim=1 if vi_model == 'non_bayes' else 2),
        )

        five = nn.ModuleList([
          nn.ModuleList([
            nn.ModuleList([
              aff([filters[4], out_dim], std=std, bias=True, relu=False)])])])

        self.model = nn.ModuleList([zero, one, two, three, four, five]).to(device)

        self.vi_model = vi_model
        self.device = device
        self.vi_samples_n = vi_samples_n

        if vi_model not in {'non_bayes', 'fac'}:

            # init
            model_set = []

            for i, super_block in enumerate(self.model):
                super_block_set = []

                for block in super_block:
                    block_set = []

                    for sub_block in block.block if i in [1, 2, 3, 4] else block:
                        sub_block_set = []

                        for f in sub_block:
                            f_set = []

                            for p, p_num in zip(f.param(), f.param_num()):
                                f_set += [[p_num, p]]
                            sub_block_set += [[sum([e[0] for e in f_set]), f_set]]
                        block_set += [[sum([e[0] for e in sub_block_set]), sub_block_set]]
                    super_block_set += [[sum([e[0] for e in block_set]), block_set]]
                model_set += [[sum([e[0] for e in super_block_set]), super_block_set]]

            self.param_recipe = model_set
            dim = sum([l[0] for l in model_set])
            self.vi_model = vi_model(dim, device)


    def forward(self, x, train=False):

        x = x.double()

        if self.vi_model == 'non_bayes':

            for block in self.model[0]:
                for sub_block in block:
                    for f in sub_block:
                        x = f(x)

            for block in self.model[1]:
                x = block(x, None, None)
            for block in self.model[2]:
                x = block(x, None, None)
            for block in self.model[3]:
                x = block(x, None, None)
            for block in self.model[4]:
                x = block(x, None, None)

            x = self.five_pre(x)

            for block in self.model[5]:
                for sub_block in block:
                    for f in sub_block:
                        x = f(x)



        elif self.vi_model == 'fac':

            for block in self.model[0]:
                for sub_block in block:
                    for f in sub_block:
                        x = f(x, self.vi_samples_n)

            for block in self.model[1]:
                x = block(x, None, self.vi_samples_n)
            for block in self.model[2]:
                x = block(x, None, self.vi_samples_n)
            for block in self.model[3]:
                x = block(x, None, self.vi_samples_n)
            for block in self.model[4]:
                x = block(x, None, self.vi_samples_n)

            x = self.five_pre(x)

            for block in self.model[5]:
                for sub_block in block:
                    for f in sub_block:
                        x = f(x, self.vi_samples_n)


        else:

            # sample

            p_set = self.prep_param(
                self.vi_model.normalized_sample(self.vi_samples_n)
            )

            for (block, p_block) in zip(self.model[0], p_set[0]):
                for (sub_block, p_sub_block) in zip(block, p_block):
                    for f, p_f in zip(sub_block, p_sub_block):
                        x = f(x, p_f)

            for (block, p_block) in zip(self.model[1], p_set[1]):
                x = block(x, p_block, None)
            for (block, p_block) in zip(self.model[2], p_set[2]):
                x = block(x, p_block, None)
            for (block, p_block) in zip(self.model[3], p_set[3]):
                x = block(x, p_block, None)
            for (block, p_block) in zip(self.model[4], p_set[4]):
                x = block(x, p_block, None)

            x = self.five_pre(x)

            for (block, p_block) in zip(self.model[5], p_set[5]):
                for (sub_block, p_sub_block) in zip(block, p_block):
                    for f, p_f in zip(sub_block, p_sub_block):
                        x = f(x, p_f)

        if train and self.vi_model != 'non_bayes':
            kl_qp = 0

            for (i, super_block) in enumerate(self.model):
                for block in super_block:
                    for sub_block in block.block if i in [1, 2, 3, 4] else block:
                        for f in sub_block:
                            kl_qp += f.kl()

            return x, kl_qp

        else:

            return x

    def prep_param(self, s):

        model_set = []
        for n_super_block, spec_super_block in self.param_recipe:
            super_block_set = []
            super_block_s = s[:, :n_super_block]
            s = s[:, n_super_block:]

            for n_block, spec_block in spec_super_block:
                block_set = []
                block_s = super_block_s[:, :n_block]
                super_block_s = super_block_s[:, n_block:]

                for n_sub_block, spec_sub_block in spec_block:
                    sub_block_set = []
                    sub_block_s = block_s[:, :n_sub_block]
                    block_s = block_s[:, n_sub_block:]

                    for n_func, spec_func in spec_sub_block:
                        func_set = []
                        func_s = sub_block_s[:, :n_func]
                        sub_block_s = sub_block_s[:, n_func:]

                        for n_p, spec_p in spec_func:
                            p_s = func_s[:, :n_p]
                            func_s = func_s[:, n_p:]

                            func_set += [p_s.reshape([self.vi_samples_n, *spec_p])]
                        sub_block_set += [func_set]
                    block_set += [sub_block_set]
                super_block_set += [block_set]
            model_set += [super_block_set]

        return model_set


class ImageDN(nn.Module):

    def __init__(self,
                 vi_model=None,
                 device='cuda',
                 std=None,
                 vi_samples_n=100):

        super(ImageDN, self).__init__()
        self.vi_model = vi_model
        self.vi_samples_n = vi_samples_n

        if vi_model == 'non_bayes':
            aff = non_bayes_aff
        elif vi_model == 'fac':
            aff = fac_aff
        else:
            aff = non_fac_aff

        self.model = nn.ModuleList([
            aff([784, 200], relu=True, bias=True, std=std),
            aff([200, 200], relu=True, bias=True, std=std),
            aff([200, 200], relu=True, bias=True, std=std),
            aff([200, 200], relu=True, bias=True, std=std),
            aff([200, 200], relu=True, bias=True, std=std),
            aff([200, 10], bias=True, std=std)
        ]).to(device)

        if vi_model not in {'non_bayes', 'fac'}:
            # init
            model_set = []
            for f in self.model:
                f_set = []
                for p, p_num in zip(f.param(), f.param_num()):
                    f_set += [[p_num, p]]
                model_set += [[sum([e[0] for e in f_set]), f_set]]

            self.param_recipe = model_set
            dim = sum([l[0] for l in model_set])
            self.vi_model = vi_model(dim, device)

    def forward(self, x, train=False):

        x = x.double().flatten(start_dim=1)

        if self.vi_model == 'non_bayes':
            for f in self.model:
                x = f(x)

        elif self.vi_model == 'fac':
            for f in self.model:
                x = f(x, self.vi_samples_n)

        else:
            # sample
            p_set = self.prep_param(
                self.vi_model.normalized_sample(self.vi_samples_n)
            )

            for (f, p_f) in zip(self.model, p_set):
                x = f(x, p_f)

        if train and self.vi_model != 'non_bayes':
            kl_qp = 0
            for f in self.model:
                kl_qp += f.kl()
            return x, kl_qp

        else:

            return x

    def prep_param(self, s):

        model_set = []
        for n_func, spec_func in self.param_recipe:
            func_set = []
            func_s = s[:, :n_func]
            s = s[:, n_func:]
            for n_p, spec_p in spec_func:
                p_s = func_s[:, :n_p]
                func_s = func_s[:, n_p:]

                func_set += [p_s.reshape([self.vi_samples_n, *spec_p])]
            model_set += [func_set]

        return model_set


class CLImageDN(nn.Module):

    def __init__(self,
                 vi_model=None,
                 device='cuda',
                 std=None,
                 vi_samples_n=100):

        super(CLImageDN, self).__init__()
        self.device = device
        self.vi_model = vi_model
        self.vi_samples_n = vi_samples_n

        if vi_model == 'non_bayes':
            aff = non_bayes_aff
        elif vi_model == 'fac':
            aff = fac_aff
        else:
            aff = non_fac_aff

        self.model = nn.ModuleList([
            aff([784, 200], relu=True, bias=True, std=std),
            aff([200, 200], relu=True, bias=True, std=std),
            aff([200, 200], relu=True, bias=True, std=std),
            aff([200, 200], relu=True, bias=True, std=std),
            aff([200, 200], relu=True, bias=True, std=std),
            aff([200, 1], relu=False, bias=True, std=std)
        ]).to(device)

        if vi_model not in {'non_bayes', 'fac'}:
            # init
            model_set = []
            for f in self.model:
                f_set = []
                for p, p_num in zip(f.param(), f.param_num()):
                    f_set += [[p_num, p]]
                model_set += [[sum([e[0] for e in f_set]), f_set]]

            self.param_recipe = model_set
            dim = sum([l[0] for l in model_set])
            self.vi_model = vi_model(dim, device)

    def forward(self, x, train=False):

        x = x.double().flatten(start_dim=1)

        if self.vi_model == 'non_bayes':
            for f in self.model:
                x = f(x)

        elif self.vi_model == 'fac':
            for f in self.model:
                x = f(x, self.vi_samples_n)

        else:
            # sample
            p_set = self.prep_param(
                self.vi_model.normalized_sample(self.vi_samples_n)
            )

            for (f, p_f) in zip(self.model, p_set):
                x = f(x, p_f)

        if train and self.vi_model != 'non_bayes':
            kl_qp = 0
            for f in self.model:
                kl_qp += f.kl()
            return x, kl_qp

        else:

            return x

    def prep_param(self, s):

        model_set = []
        for n_func, spec_func in self.param_recipe:
            func_set = []
            func_s = s[:, :n_func]
            s = s[:, n_func:]
            for n_p, spec_p in spec_func:
                p_s = func_s[:, :n_p]
                func_s = func_s[:, n_p:]

                func_set += [p_s.reshape([self.vi_samples_n, *spec_p])]
            model_set += [func_set]

        return model_set

    def con_learn(self, vi_model=None, expansion=None, prior='gaus'):
        for i in range(len(self.model)):
            if i == len(self.model) - 1:
                self.model[i].continual_learning(expansion)
                if vi_model != 'fac':
                    self.model[i].prior = prior
            else:
                self.model[i].continual_learning()
                if vi_model != 'fac':
                    self.model[i].prior = prior

        if vi_model not in {'non_bayes', 'fac'}:
            model_set = []
            for f in self.model:
                f_set = []
                for p, p_num in zip(f.param(), f.param_num()):
                    f_set += [[p_num, p]]
                model_set += [[sum([e[0] for e in f_set]), f_set]]

            self.param_recipe = model_set
            dim = sum([e[0] for e in model_set])
            self.vi_model = vi_model(dim, self.device)














