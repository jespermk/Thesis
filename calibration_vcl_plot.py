import matplotlib.pyplot as plt
from VI_models.simple_models import radial, MFG, exponential_radial, t_radial
import numpy as np
import torch as t

model_names = ['DDN', 'DDN', 'ResNet18', 'ResNet50', 'PreResNet50']

data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR10', 'CIFAR10']

vi_names = ['non_frac', 'frac', 'gaus', 'e', 't']

vi_models = [radial, 'fac', MFG, exponential_radial, t_radial]

plot_names = ['Gaussian radial', 'Factorized gaussian radial', 'Mean field gaussian', 'Exponential radial', 'Students t radial']

cs = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple')
def conf_int(ps, xs, n, c, w, conf=0.95):
    ps = np.array(ps)
    xs = np.array(xs)
    ints = conf * np.sqrt((ps * (1 - ps)) / n)

    for x, int, p in zip(xs, ints, ps):
        plt.fill_between([x - w, x + w], p - int, p + int, color=c, alpha=0.2)


plt.figure(figsize=(12, 10))
for i, (model_name, data_name) in enumerate(zip(model_names, data_names)):
    plt.subplot2grid((3, 2), (i if i < 2 else i - 2, 0 if i < 2 else 1))
    for vi_name, vi_model, plot_name in zip(vi_names, vi_models, plot_names):
        true = np.load('generated_data/calibration_plot_x_' + vi_name + '_' + model_name + '_' + data_name + '.npy')
        pred = np.load('generated_data/calibration_plot_y_' + vi_name + '_' + model_name + '_' + data_name + '.npy')
        y = pred - true
        plt.plot(
            np.arange(0, 1, 1/len(y)),
            y,
            label=plot_name
        )
    plt.plot(np.arange(0, 1, 0.01), np.zeros(100), label='optimal', ls='--', c='k')
    plt.ylim(-0.4, 0.4)
    plt.title('Model: ' + model_name + ',' + '  ' + 'Data: ' + data_name, fontweight='bold')
    if i in {0, 1, 4}:
        plt.ylabel('Calibration discrepancy')
    if i in {1, 4}:
        plt.xlabel('Prediction confidence - Occurrence rate')
plt.tight_layout()

plt.legend(bbox_to_anchor=(-0.5, 0.8))
plt.savefig('plots/calibration_second', dpi=300)
plt.show()


data = [np.load('generated_data/VCL_correct_' + vi_name + '.npy') for vi_name in vi_names]
X = np.arange(5)
plt.figure(figsize=(12, 6))
plt.subplot2grid((1, 2), (0, 0))
for d, ind, c, name in zip(data, [i*1/5 + 0.1 for i in range(5)], cs, plot_names):
    plt.scatter(X + ind, d, marker='_', s=70, linewidths=2, label=name)
    conf_int(d, X + ind, 2000, c, 0.07)

plt.grid()
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Tasks')
plt.ylim(0.7, 1)
plt.title('Individual task accuracy', fontweight='bold', fontsize=12)

plt.subplot2grid((1, 2), (0, 1))
mean = [np.mean(d) for d in data]
plt.scatter(plot_names, mean, color=('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'), marker='_', s=3000, linewidths=2)
plt.xticks([])

for mu, ind, c in zip(mean, X, cs):
    conf_int([mu], [ind], 10000, c, 0.5)

plt.ylabel('Mean accuracy')
plt.xlabel('Variational inference distributions')
plt.ylim(0.88, 0.96)
plt.title('Mean task accuracy', fontweight='bold', fontsize=12)
plt.savefig('plots/CVL', dpi=300)
plt.show()

