from criteria.criteria import Test
from VI_models.simple_models import radial, MFG, exponential_radial, t_radial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

model_names = ['DDN', 'DDN', 'ResNet18', 'ResNet50', 'PreResNet50']

data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR10', 'CIFAR10']

vi_names = ['non_frac', 'frac', 'gaus', 'e', 't']

vi_models = [radial, 'fac', MFG, exponential_radial, t_radial]

plot_names = ['Gaussian radial', 'Factorized gaussian radial', 'Mean field gaussian', 'Exponential radial', 'Students t radial']

stat = {'accuracy', 'lpd', 'ece', 'grad_std', 'calibration_plot', 'corruption'}


plt.figure(figsize=(12, 10))
for i, (model_name, data_name) in enumerate(zip(model_names, data_names)):
    plt.subplot2grid((3, 2), (i if i < 2 else i - 2, 0 if i < 2 else 1))
    for vi_name, vi_model, plot_name in zip(vi_names, vi_models, plot_names):
        y = np.load('generated_data/corruption_nlls' + vi_name + '_' + model_name + '_' + data_name + '.npy')

        plt.plot(
            np.logspace(start=-1.5, stop=0.5, num=10),
            y,
            label=plot_name
        )


    plt.title('Model: ' + model_name + ',' + '  ' + 'Data: ' + data_name, fontweight='bold')
    plt.yscale('log')
    plt.xscale('log')
    if data_name == 'CIFAR10':
        plt.ylim(0.4, 8)
    if i in {0, 1, 4}:
        plt.ylabel('Negative log likelihood')
    if i in {1, 4}:
        plt.xlabel('Standard deviation of added noise')
plt.tight_layout()

plt.legend(bbox_to_anchor=(-0.5, 0.8))
plt.savefig('plots/corruption_nlls', dpi=300)


fig = plt.figure(figsize=(45/4, 60/4))
ax = fig.add_gridspec(54, 70, hspace=0, wspace=0)
for j, (model_name, data_name) in enumerate(zip(model_names, data_names)):

    confidences = [np.load(
        'generated_data/corrupted_prob' + vi_name + '_' + model_name + '_' + data_name + '.npy'
    ) for vi_name in vi_names]

    for i in range(10):

        ax1 = fig.add_subplot(ax[11*j: 11*j + 7, 7*i: 7*(i+1)])
        ax2 = fig.add_subplot(ax[11*j + 6: 11*(j+1) - 2, 1 + 7*i: 7*i + 6])

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)


        if i == 0:
            ax1.set_title('Model: ' + model_name + ',  Data: ' + data_name, fontsize=12, fontweight='bold', loc='left')
            ax2.set_ylabel('confidence', fontsize=10)

        img = np.load(
            'generated_data/corrupted_img' + 'non_frac' + '_' + model_name + '_' + data_name + '.npy'
        )[i].squeeze()
        if len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        max, min = img.max(axis=(-1, -2), keepdims=True), img.min(axis=(-1, -2), keepdims=True)


        ax1.imshow(
            (img - min) / np.abs(min-max)
        )
        r = np.arange(-1, len(vi_names), 0.01)

        confidence = [np.load(
                'generated_data/corrupted_prob' + vi_name + '_' + model_name + '_' + data_name + '.npy'
            )[i] for vi_name in vi_names]


        ax2.bar(
            vi_names,
            [np.load(
                'generated_data/corrupted_prob' + vi_name + '_' + model_name + '_' + data_name + '.npy'
            )[i] for vi_name in vi_names],
            color=('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple')
        )
        confidence += confidence
        if i != 0:
            ax2.get_yaxis().set_visible(False)
        else:
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax2.set_ylim(np.min(confidences), np.max(confidences))

    print(np.min(confidences), np.max(confidences))

plt.tight_layout()
plt.savefig('plots/corruption_image_predictions', dpi=300)
plt.show()





