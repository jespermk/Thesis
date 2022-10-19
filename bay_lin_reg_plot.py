import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as transforms

N_mean, r_mean, r_paper_mean = [
    np.load('generated_data/blr_%s_mean.npy' % model)
    for model in ['MFG', 'r', 'paper']
]
N_var, r_var, r_paper_var = [
    np.load('generated_data/blr_%s_var.npy' % model)
    for model in ['MFG', 'r', 'paper']
]
N_grad_var, r_grad_var, r_paper_grad_var =[
    np.load('generated_data/blr_%s_grad_var.npy' % model)
    for model in ['MFG', 'r', 'paper']
]

plot_specification_mean = [
    (r_mean, 'Gaussian radial', 'tab:blue'),
    (r_paper_mean, 'Factorized gaussian radial', 'tab:orange'),
    (N_mean, 'Mean field gaussian', 'tab:green'),
]

plot_specification_var = [
    (r_var, 'Gaussian radial', 'tab:blue'),
    (r_paper_var, 'Factorized gaussian radial', 'tab:orange'),
    (N_var, 'Mean field gaussian', 'tab:green'),
]

plot_specification_grad_var = [
    (r_grad_var, 'Gaussian radial', 'tab:blue',),
    (r_paper_grad_var, 'Factorized gaussian radial', 'tab:orange'),
    (N_grad_var, 'Mean field gaussian', 'tab:green')
]
plot_names = ['Gaussian radial', 'Factorized gaussian radial', 'Mean field gaussian']

plt.figure(figsize=(12, 6))

x = np.repeat(
    np.expand_dims(np.arange(4), axis=0),
    10,
    axis=0
)
mean_grad_var = np.mean(
    np.concatenate((N_grad_var, r_grad_var, r_paper_grad_var), axis=0),
    axis=0
)

offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData

plt.subplot2grid((2, 3), (0, 0))
for it, [mean, label, color] in enumerate(plot_specification_mean):
    plt.scatter(x + it/5 - 0.2, mean, c=color, marker='_', linewidths=1, s=50, alpha=0.35, zorder=0)
    plt.scatter(x[0, :] + it/5 - 0.2, np.mean(mean, axis=0), c=color, label=label, marker='+', s=50, linewidths=2, alpha=1)
    plt.xticks([0, 1, 2, 3], ['1', '32^1', '32^2', '32^3'])
plt.yscale('log')
plt.title('Norm of error in means', fontweight='bold')
plt.ylabel('Norm of error in means')

plt.subplot2grid((2, 3), (0, 1))
for it, [var, label, color] in enumerate(plot_specification_var):
    plt.scatter(x + it/5 - 0.2, var, c=color, marker='_', linewidths=1, s=50, alpha=0.35, zorder=0)
    plt.scatter(x[0, :] + it/5 - 0.2, np.mean(var, axis=0), c=color, label=label, marker='+', s=50, linewidths=2, alpha=1)
    plt.xticks([0, 1, 2, 3], ['1', '32^1', '32^2', '32^3'])
plt.yscale('log')
plt.title('Norm of error in variances', fontweight='bold')
plt.ylabel('Norm of error in variances')

plt.subplot2grid((2, 3), (0, 2))
for it, [grad_var, label, color] in enumerate(plot_specification_grad_var):
    plt.scatter(x + it/5 - 0.2, grad_var, c=color, marker='_', linewidths=1, s=50, alpha=0.35, zorder=0)
    plt.scatter(x[0, :] + it/5 - 0.2, np.mean(grad_var, axis=0), c=color, label=label, marker='+', s=50, linewidths=2, alpha=1)
    plt.xticks([0, 1, 2, 3], ['1', '32^1', '32^2', '32^3'])
plt.yscale('log')
plt.title('Norm of gradient variance', fontweight='bold')
plt.ylabel('Norm of gradient variance')
plt.xlabel('Dimension')

plt.subplot2grid((2, 3), (1, 0))
for it, [grad_var, label, color] in enumerate(plot_specification_grad_var):
    plt.scatter(x + it/5 - 0.2, grad_var / mean_grad_var, c=color, marker='_', linewidths=1, s=50, alpha=0.35, zorder=0)
    plt.scatter(x[0, :] + it/5 - 0.2, np.mean(grad_var, axis=0) / mean_grad_var, c=color, label=label, marker='+', s=50, linewidths=2, alpha=1)
    plt.xticks([0, 1, 2, 3], ['1', '32^1', '32^2', '32^3'])
plt.title('Normalized norm of gradient variance', fontweight='bold')
plt.ylabel('Norm of gradient variance')
plt.xlabel('Dimension')


plt.subplot2grid((2, 3), (1, 1))
for it, [grad_var, label, color] in enumerate(plot_specification_grad_var):
    plt.scatter(x + it/5 - 0.2, grad_var / 32 ** np.arange(4), c=color, marker='_', linewidths=1, s=50, alpha=0.35, zorder=0)
    plt.scatter(x[0, :] + it/5 - 0.2, np.mean(grad_var, axis=0) / 32 ** np.arange(4), c=color, label=label, marker='+', s=50, linewidths=2, alpha=1)
    plt.xticks([0, 1, 2, 3], ['1', '32^1', '32^2', '32^3'])

plt.title('Gradient variance per dimension', fontweight='bold')
plt.ylabel('Gradient variance per dimension')
plt.xlabel('Dimension')
plt.legend()
plt.tight_layout()
plt.savefig('plots/bayes_lin_reg', dpi=400)
plt.show()

