from VI_models.simple_models import MFG, radial, exponential_radial, t_radial
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm


vi_dist = [radial, exponential_radial, t_radial]
vi_names = ['Gaussian radial', 'Exponential radial', 'Students t radial']
cs = ['tab:blue', 'tab:red', 'tab:purple']

dim = [2, 10, 100]
plt.figure(figsize=(12, 12))
for i, (vi, name, c) in tqdm(enumerate(zip(vi_dist, vi_names, cs))):
    for ii, d in enumerate(dim):
        plt.subplot2grid((3, 3), (i, ii))
        dist = vi(d, 'cpu')
        marginal_samples = dist.normalized_sample(1000000)[:, 0].numpy()
        marginal_std = np.std(marginal_samples)
        n, bins, patches = plt.hist(marginal_samples, 1000, density=True, facecolor=c, label=name)
        plt.plot(
            np.arange(-5 * marginal_std, 5 * marginal_std, 0.01 * marginal_std),
            norm.pdf(np.arange(-5 * marginal_std, 5 * marginal_std, 0.01 * marginal_std), scale=marginal_std),
            ls='--',
            alpha=0.7,
            label='Gaussian with same std'
        )
        plt.ylabel('density')
        plt.xlabel('first dimension')
        plt.xlim(-3 * marginal_std, 3 * marginal_std)
        plt.title('Marginal for %i d. ' % d + name, fontsize=10)
    plt.legend()
plt.tight_layout()
plt.savefig('plots/profiles', dpi=300)
plt.show()

plt.close()

plt.figure(figsize=(12, 12))
for i, (vi, name, c) in tqdm(enumerate(zip(vi_dist, vi_names, cs))):
    for ii, d in enumerate(dim):
        plt.subplot2grid((3, 3), (i, ii))
        dist = vi(d, 'cpu')
        marginal_samples = dist.normalized_sample(1000000)[:, 0].numpy()
        marginal_std = np.std(marginal_samples)
        n, bins, patches = plt.hist(marginal_samples, 1000, density=True, facecolor=c, label=name)
        plt.plot(
            np.arange(-10 * marginal_std, 10 * marginal_std, 0.01 * marginal_std),
            norm.pdf(np.arange(-10 * marginal_std, 10 * marginal_std, 0.01 * marginal_std), scale=marginal_std),
            ls='--',
            alpha=0.7,
            label='Gaussian with same std'
        )
        plt.xlim(-10 * marginal_std, 10 * marginal_std)
        plt.yscale('log')
        plt.ylabel('log density')
        plt.xlabel('first dimension')
        plt.title(' log marginal for %i d. ' % d + name, fontsize=9, fontweight='bold')
    plt.legend()
plt.tight_layout()
plt.savefig('plots/log_profiles',  dpi=300)
plt.show()
