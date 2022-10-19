import matplotlib.pyplot as plt
import numpy as np

model_names = [
    'DDN',
    'DDN',
    'ResNet18',
    'ResNet50',
    'PreResNet50',
]

vi_names = [
'non_frac',
    'frac',
    'gaus',
]
data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR10', 'CIFAR10']
data_type = ['eval_set_nlls_', 'eval_set_correct_rates_']
metrics = ['Negative log likelihood', 'Accuracy']
plot_names = ['Gaussian radial', 'Factorized Gaussian radial', 'Mean field Gaussian']
colors = ['tab:blue', 'tab:red', 'tab:purple']

plt.figure(figsize=(10, 12))
for i, (stat, metric) in enumerate(zip(data_type, metrics)):
    for ii, (data, model_name) in enumerate(zip(data_names, model_names)):
        plt.subplot2grid((len(model_names), len(data_type)), (ii, i))
        for vi_name, plot_name, color in zip(vi_names, plot_names, colors):

            plt.grid(True)
            plt.plot(
                2*np.arange(len(np.load('generated_data/' + stat + vi_name + '_' + model_name + '_' + data + '_std_1' + '.npy'))),
                np.load('generated_data/' + stat + vi_name + '_' + model_name + '_' + data + '_std_1' + '.npy'),
                label=plot_name,
                c=color

            )


        if stat == 'train_set_losses_':
            plt.yscale('log')

        if stat == 'eval_set_correct_rates_':
            plt.ylim(0, 1)
        if metric == 'Accuracy':
            if data == 'MNIST':
                plt.ylim(0.95, 1)
            if data == 'FashionMNIST':
                plt.ylim(0.85, 1)
            if data == 'CIFAR10':
                plt.ylim(0.6, 1)


        plt.title('Model: ' + model_name + ',' + '  ' + 'Data: ' + data)
        plt.ylabel(metric)
        plt.xlabel('epochs')
plt.legend()
plt.tight_layout()
plt.savefig('plots/full_run_second', dpi=400)
plt.show()