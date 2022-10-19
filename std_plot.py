import matplotlib.pyplot as plt
import numpy as np

def conf_int(ps, xs, n, c, conf=0.95):
    ps = np.array(ps)
    xs = np.array(xs)
    ints = conf * np.sqrt((ps * (1 - ps)) / n)

    for x, int, p in zip(xs, ints, ps):
        plt.fill_between([x - 0.1, x + 0.1], p - int, p + int, color=c, alpha=0.3)


stds = ['1', '10', '100']

model_names = ['DDN', 'DDN', 'ResNet18', 'ResNet50', 'PreResNet50']

data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR10', 'CIFAR10']

vi_names = ['non_frac', 'frac', 'gaus']

plot_names = ['Gaussian radial', 'Factorized gaussian radial', 'Mean field gaussian']

cs = ['tab:blue', 'tab:orange', 'tab:green']

x_ints = [[0.95, 1], [0.85, 0.925], [0.825, 0.925], [0.79, 0.9], [0.79, 0.925]]

ns = [5000, 5000, 2000, 2000, 2000]

plt.figure(figsize=(12, 8))
for i, (model_name, data_name, n, x_int) in enumerate(zip(model_names, data_names, ns, x_ints)):
    plt.subplot2grid((2, 3), (0 if i < 3 else 1, i if i < 3 else i - 3))
    plt.xticks([1, 2, 3], ['1', '10', '100'])
    for j, (vi_name, plot_name, c) in enumerate(zip(vi_names, plot_names, cs)):
        acc = []
        for std in stds:

            acc += [np.load('generated_data/acc_'+vi_name+'_'+model_name+'_'+data_name+'_std_'+std+'.npy').squeeze()]

        plt.scatter([3/4 + j/4, 7/4 + j/4, 11/4 + j/4], acc, label=plot_name, marker='+', s=100, linewidths=3)


        conf_int(acc, [3/4 + j/4, 7/4 + j/4, 11/4 + j/4], n, c)

    if i == 0:
        plt.ylabel('Accuracy')
    plt.xlabel('Prior standard deviation')

    plt.plot(
        [3/4, 5/3, 14/4],
        np.ones(3)*np.load('generated_data/acc_non_bayes_'+model_name+'_'+data_name+'_std_1.npy').squeeze(),
        label='non_bayesian', ls='--', alpha=0.5, c='tab:red')

    plt.title('Model: ' + model_name + ',   ''Data: ' + data_name, fontweight='bold', fontsize=10)
    plt.ylim(x_int)

plt.tight_layout()
plt.legend()
plt.savefig('plots/std', dpi=300)
plt.show()

