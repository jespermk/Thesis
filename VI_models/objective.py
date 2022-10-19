import torch as t
import torch.nn as nn

class likelihood(nn.Module):

    def __init__(self, m):
        super(likelihood, self).__init__()
        self.m = m
        self.l = nn.GaussianNLLLoss()

    def load_data(self,d_in,d_out):

        self.d_in = d_in
        self.d_out = d_out
        self.var = t.ones(self.d_out.size()) # self.var has been removed

    def log_eval(self,p):

        """
        log l(theta| D)
        """

        nl = 0

        for i in range(p.size()[0]):

            nl += self.l( self.m.evl(self.d_in, p[i, :]), self.d_out, self.var )


        return nl / p.size()[0]





class ELBO(nn.Module):

    """
    ELBO(theta) = E_q[ log L(theta, D) ] - KL[q|p], theta \sim q

        Where : KL[q|p] = E_q[ log q(theta) ] - E_q[ log p(theta) ]


    """
    def __init__(self, q, prior, nlikelihood):
        super(ELBO, self).__init__()
        self.q = q
        self.p = prior
        self.nl = nlikelihood


    def KL_qp(self,q_samp):

        """
        KL[q|p] = E_q[ log q(theta) ] - E_q[ log p(theta) ]

        """

        #entropy_q = - t.sum( t.log( t.abs(self.q.sigma) )) - 1/2 + self.q.C_d # from radial paper
        #cross_entropy_qp = t.mean( - (1/2) * t.norm( (q_samp - self.q.mu) / self.q.sigma ) ** 2 )  # from paper

        #entropy_q = t.mean( self.q.log_prob(q_samp) ) # unbiased estimate
        cross_entropy_qp = t.mean( self.p.log_prob(q_samp) )

        kl_qp = entropy_q - cross_entropy_qp

        return kl_qp

    def E_nl(self,q_samp):

        """
        - E_q [ log L(theta, D) ]

        """

        e_nl = self.nl.log_eval(q_samp)

        return e_nl


    def evl(self,n):

        """
         - ELBO(theta) = -E_q [ log L(theta, D) ] + KL[q|p]

        """

        q_samp = self.q.sample(n)

        return self.E_nl(q_samp) # + self.KL_qp(q_samp)






