from models.NN import radial_VCL_image_model
from VI_models.simple_models import radial, MFG, t_radial, exponential_radial
import numpy as np
import numpy.random as npr
import torch
import numpy.linalg as npl
from torch.nn import Module
from torch.nn import functional as F
from radial_bnn_from_paper.radial_layers.variational_bayes import SVI_Linear
from scipy.spatial.distance import mahalanobis

def data_gen_x_t_alphaI(dim = 1, data_n = 30, alpha_inv_rho = 1):

    alpha_inv = sigma_T(alpha_inv_rho) ** 2

    w_true = npr.randn(1, dim) * sigma_T(alpha_inv_rho)
    b_true = npr.randn(1, 1) * sigma_T(alpha_inv_rho)

    W_true = np.concatenate([b_true, w_true], axis=1)

    x = npr.randn(data_n, dim)
    Phi = np.column_stack((np.ones(data_n), x))

    noise = npr.randn(data_n, 1)

    t = (Phi @ W_true.T) + noise

    return x, t, alpha_inv, Phi

def take_diag(x):
    return np.diag(np.diag(x))


def get_best_MF_approx(S):
    S_inv = npl.inv(S)
    S_inv_diag = take_diag(S_inv)
    S_inv_diag_inv = npl.inv(S_inv_diag)

    return S_inv_diag_inv

def get_best_MF_approx_inv(S):
    S_inv = npl.inv(S)
    S_inv_diag = take_diag(S_inv)

    return S_inv_diag


def sigma_T(rho):
    return np.log(1 + np.exp(rho))


def get_sigma_np(m):
    if m.from_paper:
        return sigma_T(np.concatenate([
            m.first_layer.bias_rhos.detach().cpu().squeeze().numpy(),
            m.first_layer.weight_rhos.detach().cpu().squeeze().numpy()
        ]))
    else:
        return np.concatenate([
            m.sigma_b().detach().cpu().numpy(),
            m.sigma_w().detach().cpu().numpy()
        ])


def get_mu_np(m):
    if m.from_paper:
        return np.concatenate([
            m.first_layer.bias_mus.detach().cpu().flatten().numpy(),
            m.first_layer.weight_mus.detach().cpu().flatten().numpy()
        ])
    else:
        return np.concatenate([
            m.mu_b.detach().cpu().flatten().numpy(),
            m.mu_w.detach().cpu().flatten().numpy()
        ])

def kl_NN(m0, S0, m1, S1,):
    k = m0.size
    m0, m1 = m0.squeeze(),\
             m1.squeeze()
    S0, S1 = S0.squeeze(),\
             S1.squeeze()

    Z1 = npl.inv(S1)

    _, logdetS0 = npl.slogdet(S0)
    _, logdetS1 = npl.slogdet(S1)

    tra_Z1S0 = np.trace(Z1 @ S0)

    m1_m0_MahalDist = mahalanobis(m1, m0, Z1)

    return 0.5 * (tra_Z1S0 + m1_m0_MahalDist - k + logdetS1 - logdetS0)




def get_emperical_std_per_dim_np(m):
    with torch.no_grad():
        if m.from_paper:
            weight_epsilon = m.first_layer.noise_distribution((1000,) + m.first_layer.weight_mus.size())
            bias_epsilon = m.first_layer.noise_distribution((1000,) + m.first_layer.bias_mus.size())
            weight = torch.addcmul(
                m.first_layer.weight_mus,
                m.first_layer._rho_to_sigma(m.first_layer.weight_rhos),
                weight_epsilon
            ).std(dim=0).detach().cpu().flatten().numpy()
            bias = torch.addcmul(
                m.first_layer.bias_mus,
                m.first_layer._rho_to_sigma(m.first_layer.bias_rhos),
                bias_epsilon
            ).std(dim=0).detach().cpu().flatten().numpy()

            return np.concatenate([bias, weight])
        else:
            samples = m.sample_posterior(1000)
            return samples.std(dim=0).detach().cpu().numpy()


def get_dist_mu_np(m, mu_analyt):
    size_diff = npl.norm(get_mu_np(m)) - npl.norm(mu_analyt)
    return np.sign(size_diff) * npl.norm(get_mu_np(m) - mu_analyt)


def get_dist_sigma_np(m, sigma_analyt):
    size_diff = npl.norm(get_sigma_np(m)) - npl.norm(sigma_analyt)
    return np.sign(size_diff) * npl.norm(get_sigma_np(m) - sigma_analyt)


def get_dist_emperical_sigma_np(m, sigma_analyt):
    size_diff = npl.norm(get_emperical_std_per_dim_np(m)) - npl.norm(sigma_analyt)
    return np.sign(size_diff) * npl.norm(get_emperical_std_per_dim_np(m) - sigma_analyt)


def get_diff_mean_sigma_np(m, sigma_analyt):
    return np.mean(get_sigma_np(m)) - np.mean(sigma_analyt)


def get_diff_mean_emperical_sigma_np(m, sigma_analyt):
    return np.mean(get_emperical_std_per_dim_np(m)) - np.mean(sigma_analyt)

def get_analyt_m_S(Phi, t, alpha_inv=None):

    N, M = Phi.shape

    alpha = 1/alpha_inv

    A = alpha * np.identity(M) + Phi.T @ Phi

    # compute mean and covariance
    m = np.linalg.solve(A, Phi.T) @ t
    S = np.linalg.inv(A)

    return m, S


# Training set up
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



class radial_paper(Module):
    """
    Very basic stochastic variational inference MLP for classification.
    You can create this from JSON args in the full train.py pipeline.
    """

    def __init__(self, dim):
        super(radial_paper, self).__init__()
        initial_rho = -1.5  # This is a reasonable value, but not very sensitive.
        initial_mu_std = (
            "he"  # Uses Kaiming He init. Or pass float for a Gaussian variance init.
        )
        variational_distribution = "radial"  # You can use 'gaussian' to do normal MFVI.
        prior = {
            "name": "gaussian_prior",
            "sigma": sigma_T(1),
            "mu": 0,
        }  # Just a unit Gaussian prior.

        self.from_paper = True
        self.first_layer = SVI_Linear(
            in_features=dim,
            out_features=1,
            initial_rho=initial_rho,
            initial_mu=initial_mu_std,
            variational_distribution=variational_distribution,
            prior=prior,
        )

    def forward(self, x):

        x = self.first_layer(x)

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


def get_analyt_posterior_stats(analyt, iteration=2000, output='mean_var_norm_diff', all_it=False, kl_output=False):

    m, S = analyt


    if output == 'mean_var_raw':

        S_MF = get_best_MF_approx(S)
        sigma_MF = np.sqrt(np.diag(S_MF))

        if all_it:
            m = np.squeeze(np.repeat(np.expand_dims(m, axis=0), iteration, axis=0))
            s_MF_it = np.squeeze(np.repeat(np.expand_dims(sigma_MF, axis=0), iteration, axis=0))

        if kl_output:
            kl = kl_NN(m, S_MF, m, S)
            return \
                m, \
                s_MF_it, \
                kl*np.ones([iteration, 1])
        else:
            return \
                m, \
                s_MF_it, \

    elif output == 'mean_var_norm_diff':

        sigma_norm_error = np.zeros([1, 1])
        mu_norm_error = np.zeros([1, 1])

        if all_it:
            mu_norm_error = np.ones([iteration,1])*mu_norm_error
            sigma_norm_error = np.ones([iteration,1])*sigma_norm_error

        if kl_output:
            S_MF = get_best_MF_approx(S)
            kl = kl_NN(m, S_MF, m, S)
            return \
                mu_norm_error, \
                sigma_norm_error, \
                kl*np.ones([iteration, 1])
        else:
            return \
                mu_norm_error, \
                sigma_norm_error,


def compute_posterior_stats(x, t, analyt, dim=1, model=MFG, iteration=1000, output = 'mean_var_norm_diff',
                            all_it=False,
                            kl_output=False,
                            grad_var_output=False):
    mu1, S1 = analyt
    mu1 = mu1.squeeze()

    if output == 'mean_var_norm_diff':
        Z1 = get_best_MF_approx_inv(S1)

    # Radial
    layer_kind = [
        'aff',
    ]
    shape = [
        [dim, 1]
    ]


    if model == radial_paper:
        from_paper = radial_paper(dim)
        from_paper.to(device)
        optimizer_radialNN = torch.optim.Adam(from_paper.parameters(), lr=0.1, amsgrad=True)

    else:
        radialNN = radial_VCL_image_model(model,
                                          layer_kind=layer_kind,
                                          shape=shape,
                                          device=device,
                                          vi_samples_n=1,
                                          rho=-1.5)

        radialNN.from_paper = False
        radialNN.to(device)
        optimizer_radialNN = torch.optim.Adam(radialNN.parameters(), lr=0.1, amsgrad=True)

    nll = torch.nn.GaussianNLLLoss(reduction='none')

    x = torch.from_numpy(x).to(device).double()
    t = torch.from_numpy(t).to(device).double()
    dataset_size = x.size(0)
    var = torch.ones(dataset_size).to(device)

    mu = []
    Sigma_emperical = []
    kl = []

    for i in range(iteration):
        # training
        for e in range(1000):
            optimizer_radialNN.zero_grad()

            if model == radial_paper:
                t_pred, kl_qp = from_paper(x.unsqueeze(1).double()), from_paper.kl()

            else:
                t_pred, kl_qp = radialNN(x, train=True)

            t_pred = t_pred.squeeze(dim=2)

            nll_vector = nll(
                t_pred,
                t,
                var
            ).mean(dim=1)
            nll_normalized = nll_vector.mean(dim=0)
            loss = nll_normalized + kl_qp / dataset_size

            loss.backward()
            optimizer_radialNN.step()

        if all_it:

            if model == radial_paper:
                m0 = get_mu_np(from_paper)
                STD0 = np.diag(get_emperical_std_per_dim_np(from_paper))
                mu += [m0]
                Sigma_emperical += [STD0]
            else:
                m0 = get_mu_np(radialNN)
                STD0 = np.diag(get_emperical_std_per_dim_np(radialNN))
                mu += [m0]
                Sigma_emperical += [STD0]

            if kl_output:
                kl += [kl_NN(m0, np.square(STD0), mu1, S1)]

    if grad_var_output:

        grads = []

        for e in range(1000):

            optimizer_radialNN.zero_grad()

            if model == radial_paper:
                t_pred, kl_qp = from_paper(x.unsqueeze(1).double()), from_paper.kl()

            else:
                t_pred, kl_qp = radialNN(x, train=True)

            t_pred = t_pred.squeeze(dim=2)

            nll_vector = nll(
                t_pred,
                t,
                var
            ).mean(dim=1)
            nll_normalized = nll_vector.mean(dim=0)
            loss = nll_normalized + kl_qp / dataset_size
            loss.backward()

            with torch.no_grad():
                g = []

                if model == radial_paper:

                    g += [from_paper.first_layer.weight_rhos.grad.flatten()]
                    g += [from_paper.first_layer.bias_rhos.grad.flatten()]

                    g += [from_paper.first_layer.weight_mus.grad.flatten()]
                    g += [from_paper.first_layer.bias_mus.grad.flatten()]

                else:

                    g += [radialNN.rho_w.grad.flatten()]
                    g += [radialNN.rho_b.grad.flatten()]

                    g += [radialNN.mu_w.grad.flatten()]
                    g += [radialNN.mu_b.grad.flatten()]

                grads += [torch.cat(g, dim=0).cpu().numpy()]

        grads_var_norm = npl.norm(
            np.var(
                np.stack(grads, axis=0),
                axis=0
            )
        )

    if all_it:
        mu0 = np.stack(mu)
        Sigma0 = np.stack(Sigma_emperical)

        if kl_output:
            kl = np.stack(kl)

    else:

        if model == radial_paper:
            mu0 = get_mu_np(from_paper)
            mu0 = np.expand_dims(mu0, axis=0)
            Sigma0 = np.diag(get_emperical_std_per_dim_np(from_paper))
            Sigma0 = np.expand_dims(Sigma0, axis=0)

        else:
            mu0 = get_mu_np(radialNN)
            mu0 = np.expand_dims(mu0, axis=0)
            Sigma0 = np.diag(get_emperical_std_per_dim_np(radialNN))
            Sigma0 = np.expand_dims(Sigma0, axis=0)

        if kl_output:
            kl += [kl_NN(mu0, np.square(Sigma0), mu1, S1)]

    if output == 'mean_var_raw':

        if kl_output:
            return \
                mu0.squeeze(), \
                np.diagonal(Sigma0, axis1=1, axis2=2).squeeze(), \
                kl.squeeze()
        else:
            return \
                mu0.squeeze(),\
                np.diagonal(Sigma0, axis1=1, axis2=2).squeeze(),\

    elif output == 'mean_var_norm_diff':

        k = dim + 1
        mu_error = (mu1 - mu0)*np.diag(Z1)
        mu_norm_error = npl.norm(mu_error, axis=1)
        sigma_error = np.identity(k) - (Z1 @ np.square(Sigma0))
        sigma_norm_error = npl.norm(sigma_error, axis=(1, 2), ord='fro')

        if kl_output:
            return\
                mu_norm_error.squeeze(),\
                sigma_norm_error.squeeze(),\
                kl.squeeze()

        elif grad_var_output:
            return\
                mu_norm_error.squeeze(),\
                sigma_norm_error.squeeze(),\
                grads_var_norm.squeeze()

        else:
            return\
                mu_norm_error.squeeze(),\
                sigma_norm_error.squeeze(),\

dims = 32**np.arange(4)

mean_ens = []
var_ens = []
grad_var_ens = []

for _ in range(10):

    mean_ = []
    var_ = []
    grad_var_ = []

    for dim in dims:

        x, t, alpha_inv, Phi = data_gen_x_t_alphaI(dim=dim)

        [m, S] = get_analyt_m_S(Phi, t, alpha_inv)

        [
            (r_paper_m, r_paper_v, r_paper_gv)
        ] = [

           compute_posterior_stats(
               x, t,
               [m, S],
               dim=dim,
               model=model,
               iteration=6 * np.sqrt(dim).astype(int),
               all_it=False,
               kl_output=False,
               grad_var_output=True,
               output='mean_var_norm_diff'
           )

           for model in [radial_paper]
        ]

        mean_ += [r_paper_m]
        var_ += [r_paper_v]
        grad_var_ += [r_paper_gv]

    mean_ens += [np.stack(mean_)]
    var_ens += [np.stack(var_)]
    grad_var_ens += [np.stack(grad_var_)]


np.save(
    'generated_data/blr_paper_mean',
    np.stack(mean_ens)
)
np.save(
    'generated_data/blr_paper_var',
    np.stack(var_ens)
)
np.save(
    'generated_data/blr_paper_grad_var',
    np.stack(grad_var_ens)
)


"""

    plot_specification_mean = [
        (N_mean, 'MFG', 'b'),
        (r_mean, 'radial', 'r'),
        (r_paper_mean, 'radial paper', 'g'),
    ]

    plot_specification_var = [
        (N_var, 'MFG', 'b'),
        (r_var, 'radial', 'r'),
        (r_paper_var, 'radial paper', 'g'),
    ]

    plot_specification_grad_var = [
        (N_grad_var, 'MFG', 'b'),
        (r_grad_var, 'radial', 'r'),
        (r_paper_grad_var, 'radial paper', 'g'),
    ]

    plt.figure(figsize=(20, 30))

    plt.subplot2grid((1, 3), (0, 0))
    for mean, label, color in plot_specification_mean:
        plt.plot(mean, c=color, label=label)
    plt.yscale('log')
    plt.title('Norm of error in means')
    plt.ylabel('Norm of error in means')
    plt.xlabel('dimension')

    plt.subplot2grid((1, 3), (0, 1))
    for var, label, color in plot_specification_var:
        plt.plot(var, c=color, label=label)
    plt.yscale('log')
    plt.title('Norm of error in variances')
    plt.ylabel('Norm of error in variances')
    plt.xlabel('dimension')

    plt.subplot2grid((1, 3), (0, 2))
    for grad_var, label, color in plot_specification_grad_var:
        plt.plot(grad_var, c=color, label=label)
    plt.yscale('log')
    plt.title('Norm of gradient variance')
    plt.ylabel('Norm of gradient variance')
    plt.xlabel('dimension')

    plt.legend()
    plt.savefig('plots/bayesian_lin_reg_per_dim', dpi=400)
    """




"""
plot_specification_mean += [[
 (N_mean, 'MFG', 0.3, 'b', '-'),
 (r_mean, 'radial', 0.3, 'r', '-'),
 (r_paper_mean, 'radial paper', 0.3, 'g', '-'),
 (analyt_mean, 'analytical', 1, 'k', '--'),
 ]]
plot_specification_var += [[
    (N_var, 'MFG', 0.3, 'b', '-'),
    (r_var, 'radial', 0.3, 'r', '-'),
    (r_paper_var, 'radial', 0.3, 'g', '-'),
    (analyt_var, 'analytical', 1, 'k', '--'),
]]
plot_specification_kl += [[
    (N_kl, 'MFG', 0.3, 'b', '-'),
    (r_kl, 'radial', 0.3, 'r', '-'),
    (r_paper_kl, 'radial paper', 0.3, 'g', '-'),
    (analyt_kl, 'analytical', 1, 'k', '--'),
]]
row = idx + 1
col = 3
plt.figure(figsize=(20, 30))

for it in range(row):
    plt.subplot2grid((row, col), (it, 1))
    plt.title('mus')

    for mean, label, transparency, color, ls in plot_specification_mean[it]:

        plt.plot(mean, alpha=transparency, c=color, ls=ls, )

    if it == row - 1:
        plt.xlabel('iterations')
    plt.ylabel('means')


for it in range(row):
    plt.subplot2grid((row, col), (it, 2))
    plt.title('sigmas')
    for var, label, transparency, color, ls in plot_specification_var[it]:

        plt.plot(var, alpha=transparency, c=color, ls=ls,)

    if it == row-1:
        plt.xlabel('iterations')
    plt.ylabel('standard deviations')


for it in range(row):
    plt.subplot2grid((row, col), (it, 0))
    plt.title('kl')

    for kl, label, transparency, color, ls in plot_specification_kl[it]:

        if it == 0:
            plt.plot(kl, alpha=transparency, c=color, label=label, ls=ls)
        else:
            plt.plot(kl, alpha=transparency, c=color, ls=ls)

    if it == row-1:
        plt.xlabel('iterations')
    plt.yscale('log')
    plt.ylabel('KL as between Gaussians - dim=%d' % dims[it])
    plt.legend()

plt.savefig('plots/bayesian_lin_reg_per_dim', dpi=400)
"""
