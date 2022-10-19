import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

model_names = ['DDN', 'DDN', 'ResNet18', 'ResNet50', 'PreResNet50']

data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR10', 'CIFAR10']

test_set_size = [10000, 10000, 8000, 8000, 8000]

vi_names = ['non_frac', 'frac', 'gaus', 'e', 't']

plot_names = ['Gaussian radial', 'Factorized gaussian radial', 'Mean field gaussian', 'Exponential radial', 'Students t radial']

def conf_int(p, n, conf=0.95):
    ints = conf * np.sqrt((p * (1 - p)) / n)

    return ints

all_res = []
data_n = []
for model_name, data_name, n in zip(model_names, data_names, test_set_size):
    mod_res = []
    for vi_name in vi_names:

        stats = np.load('generated_data/stat_' + vi_name + '_' + model_name + '_' + data_name + '.npy')

        mod_res += [stats]
    all_res += [mod_res]
    data_n += [n for _ in model_names]

all_res = np.concatenate(all_res)

fig, ax = plt.subplots(figsize=(10, 3))
columns = ['Acc.', 'ELPD', 'ECE', 'Expect. grad. std.']
rows = [[name if e == 0 else ' '] for name in plot_names for e in range(5)]

data = {
'Variational Distribution': [vi for _ in model_names for vi in plot_names],
'Model': [model for model in model_names for _ in plot_names],
'Data': [data for data in data_names for _ in plot_names],
'Acc.': ['%.3f' % row[-4] + '±%.3f' % conf_int(row[-4], n) for row, n in zip(all_res, data_n)],
'ELPD': ['%.3f' % row[-3] for row in all_res],
'ECE': ['%.3f' % row[-2] for row in all_res],
'Ex. grad std': ['%.3f' % row[-1] for row in all_res],
}

df = pd.DataFrame(data=data)
file_name = 'table_first' if len(plot_names) == 3 else 'table_second'
df.to_latex('plots/' + file_name + '.tex', index=False)


"""
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
"""