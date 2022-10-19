import torch as t
import torch.nn as nn
import math

class radial(nn.Module):
    def __init__(self,dimension, device):
        super(radial, self).__init__()

        self.dimension = dimension

        #init_interval = t.tensor(init_interval)

        # pi = t.tensor(math.pi)

        #self.C_d = t.log( t.tensor(2) ) - (self.dimension/2) * t.log(2 * pi) + t.lgamma(self.dimension/2) - t.log(t.sqrt(2 * pi))

        """
        if random_init: # see py.torch documentation for Linear
            self.mu = nn.Parameter( t.randn(self.dimension) * init_interval  + init_interval )
            self.sigma = nn.Parameter( t.ones(self.dimension) )
        else:
            self.mu = nn.Parameter( t.zeros(self.dimension) )
            self.sigma = nn.Parameter( t.ones(self.dimension) )
        """

        self.dim = dimension
        self.device = device


    def normalized_sample(self,n):

        '''
        Samples from ( e_MFVI / || e_MFVI || ) * r, ie: d * r

            Where : d = e_MFVI / || e_MFVI || ie: The normalized direction
        '''

        size = [n, self.dim]

        epsilon_mfvi = t.randn(size, device=self.device, dtype=t.float64)
        distance = t.randn((size[0]), device=self.device, dtype=t.float64)

        normalizing_factor = t.norm(epsilon_mfvi.view(size[0], -1), p=2, dim=1).unsqueeze(1)
        distance = distance.unsqueeze(1)

        direction = epsilon_mfvi / normalizing_factor
        epsilon_radial = direction * distance

        return epsilon_radial

    def entropy(self, sigma_w, sigma_b):

        entropy = - t.sum(t.log(sigma_w))
        if sigma_b is not None:
            entropy -= t.sum(t.log(sigma_b))

        return entropy

    def cross_entropy(self, p_sample, mu, sigma):
        return \
               (
                - 0.5 * ((p_sample - mu) / sigma).norm(dim=-1).square()
               ).mean()

    """
    def sample(self, n):

        '''
        w_radial \sim  \mu + \sigma * e

        Where: e = e_MFVI / || e_MFVI || * r ie: The normalized dirction
        '''

        e_r = self.radial_normalized_sample(n)

        sample = self.mu + t.abs( self.sigma ) * e_r

        return sample


    def log_prob(self,x):

        '''
        Evaluates x by constructing a gaussian from the model parameters and then evaluating that
        '''


        log_p_x = self.C_d - (1/2) * t.norm(x) ** 2 - t.sum( t.log( t.abs( self.sigma) ) )

        return log_p_x

    def loss(self, m, p, x, y, n):

        e_r = self.radial_normalized_sample(n)

        p_sample = self.mu + t.abs( self.sigma ) * e_r

            # Likelihood

        nl = []
        l = nn.GaussianNLLLoss()

        for i in range(p_sample.size()[0]):

            nl += [l(m.evl(x, p_sample[i, :]), y, t.ones(y.size()))]

        nl = t.stack(nl)

        E_nl = t.mean(nl)

            # KL[ q | p ]

        entropy_q = self.C_d - (1 / 2) - t.sum(t.log(t.abs(self.sigma)))

        cross_entropy_qp = t.mean( p.log_prob(p_sample) )

        kl_qp = entropy_q - cross_entropy_qp

            # Loss

        loss = E_nl # + kl_qp

        return loss

        """

class MFG(nn.Module):
    def __init__(self, dimension, device):
        super(MFG, self).__init__()

        self.dim = dimension
        self.device = device


    def normalized_sample(self, n):

        return t.randn([n, self.dim], device=self.device, dtype=t.float64)

    def entropy(self, sigma_w, sigma_b):
        entropy = - t.sum(t.log(sigma_w))
        if sigma_b is not None:
            entropy -= t.sum(t.log(sigma_b))

        return entropy

    def cross_entropy(self, p_sample, mu, sigma):
        return \
               (
                - 0.5 * ((p_sample - mu) / sigma).norm(dim=-1).square()
               ).mean()

    def log_prob(self, x):

        return - 0.5 * t.sum(x ** 2, dim=-1)




##############

class exponential_radial(nn.Module):
    def __init__(self, dim, device):
        super(exponential_radial, self).__init__()

        #init_interval = t.tensor(init_interval)

        # pi = t.tensor(math.pi)

        #self.C_d = t.log( t.tensor(2) ) - (self.dimension/2) * t.log(2 * pi) + t.lgamma(self.dimension/2) - t.log(t.sqrt(2 * pi))

        """
        if random_init: # see py.torch documentation for Linear
            self.mu = nn.Parameter( t.randn(self.dimension) * init_interval  + init_interval )
            self.sigma = nn.Parameter( t.ones(self.dimension) )
        else:
            self.mu = nn.Parameter( t.zeros(self.dimension) )
            self.sigma = nn.Parameter( t.ones(self.dimension) )
        """

        self.dim = dim
        self.device = device

        self.r = t.distributions.exponential.Exponential(
            t.ones(1, device=device, dtype=t.float64)
        )

    def normalized_sample(self, n):

        n = t.tensor(n)

        size = [n, self.dim]
        # First we find a random direction (\epsilon_{\text{MFVI}} in equation (3) on page 4)
        epsilon_mfvi = t.randn(size, device=self.device, dtype=t.float64)

        # Then we pick a distance (r in equation (3) on page 4)
        distance = self.r.sample([size[0]])

        normalizing_factor = t.norm(epsilon_mfvi.view(size[0], -1), p=2, dim=1).unsqueeze(1)
        distance = distance

        direction = epsilon_mfvi / normalizing_factor
        epsilon_radial = direction * distance

        return epsilon_radial

    def entropy(self, sigma_w, sigma_b):
        entropy = - t.sum(t.log(sigma_w))
        if sigma_b is not None:
            entropy -= t.sum(t.log(sigma_b))

        return entropy

    def cross_entropy(self, p_sample, mu, sigma):

        return \
               (
                - ((p_sample - mu) / sigma).norm(dim=-1).square()
               ).mean()


class t_radial(nn.Module):
    def __init__(self, dim, device):
        super(t_radial, self).__init__()

        self.dim = dim
        self.device = device

        self.df = t.ones(1, device=device, dtype=t.float64)*3

        self.r = t.distributions.studentT.StudentT(
            self.df,
            loc=t.zeros(1, device=device, dtype=t.float64),
            scale=1 / t.sqrt(3 * t.ones(1, device=device, dtype=t.float64))
        )

    def normalized_sample(self, n):


        size = [n, self.dim]
        # First we find a random direction (\epsilon_{\text{MFVI}} in equation (3) on page 4)
        epsilon_mfvi = t.randn(size, device=self.device, dtype=t.float64)

        # Then we pick a distance (r in equation (3) on page 4)
        distance = self.r.sample([size[0]])

        normalizing_factor = t.norm(epsilon_mfvi.view(size[0], -1), p=2, dim=1).unsqueeze(1)
        distance = distance

        direction = epsilon_mfvi / normalizing_factor
        epsilon_radial = direction * distance

        return epsilon_radial

    def entropy(self, sigma_w, sigma_b):
        entropy = - t.sum(t.log(sigma_w))
        if sigma_b is not None:
            entropy -= t.sum(t.log(sigma_b))

        return entropy

    def cross_entropy(self, p_sample, mu, sigma):

        return \
               (
                - (self.df+1)/2 * t.log(
                                        1 +
                                        ((p_sample - mu) / sigma).norm(dim=-1).square()
                                        / self.df
                                        )
                ).mean()











