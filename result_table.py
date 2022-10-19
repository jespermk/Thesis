import matplotlib.pyplot as plt
import numpy as np

model_names = ['DDN', 'DDN', 'ResNet18', 'ResNet50', 'PreResNet50']

data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR10', 'CIFAR10']

vi_names = ['non_frac', 'frac', 'gaus']

plot_names = ['Gaussian radial', 'Factorized gaussian radial', 'Mean field gaussian']

all_res = []

for vi_name in vi_names:
    mod_res = []
    for model_name, data_name in zip(model_names, data_names):

        stats = np.load('generated_data/stat_' + vi_name + '_' + model_name + '_' + data_name + '.npy')

        mod_res += [stats]

    all_res += [mod_res]


all_res = np.concatenate(all_res)
print(len([[res for res in row] for row in all_res]))

fig, ax = plt.subplots(figsize=(10, 3))
columns = ['Accuracy', 'LPD', 'ECE', 'Expected grad std']
rows = [[name if e == 0 else ' '] for name in plot_names for e in range(5)]
print(rows, len(rows))
ax.table(
    cellText=[['%.2e' % res for res in row] for row in all_res],
    rowLabels=[name + ' - ' + model + ' on ' + data if e == 0 else ' - | | -   ' + model + ' on ' + data for name in plot_names for e, (model, data) in enumerate(zip(model_names, data_names))],
    colLabels=['Accuracy', 'LPD', 'ECE', 'Expected grad std'],
    fontsize=50,
    cellLoc='center',
    loc='upper left'
)
ax.axis('off')
plt.tight_layout()
plt.savefig('plots/table_first', dpi=400)
plt.show()
