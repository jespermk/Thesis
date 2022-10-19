from criteria.criteria import Test
from VI_models.simple_models import radial, MFG, exponential_radial, t_radial
import numpy as np
import torch as t


model_names = ['DDN', 'DDN', 'ResNet18', 'ResNet50', 'PreResNet50']

data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR10', 'CIFAR10']

vi_names = ['t']

vi_models = [t_radial]

stat = {'accuracy', 'lpd', 'ece', 'grad_std', 'calibration_plot', 'corruption'}

if t.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

for model_name, data_name in zip(model_names, data_names):
    for vi_name, vi_model in zip(vi_names, vi_models):


        test = Test(
            param='trained_models/m_' + vi_name + '_' + model_name + '_' + data_name,
            model=model_name,
            data=data_name,
            vi_model=vi_model,
            use_test_set=True
        )

        acc, lpd, ece, grad_std, calibration_plot, corruption_nlls, corruption_prob_ind = test.run_stats(stat)

        stats = np.array([acc, lpd, ece, grad_std])
        calibration_plot_x, calibration_plot_y = calibration_plot
        corrupted_prob, corrupted_img = corruption_prob_ind


        np.save(
            'generated_data/stat_' + vi_name + '_' + model_name + '_' + data_name,
            stats
        )
        np.save(
            'generated_data/calibration_plot_x_' + vi_name + '_' + model_name + '_' + data_name,
            calibration_plot_x
        )
        np.save(
            'generated_data/calibration_plot_y_' + vi_name + '_' + model_name + '_' + data_name,
            calibration_plot_y
        )
        np.save(
            'generated_data/corruption_nlls' + vi_name + '_' + model_name + '_' + data_name,
            corruption_nlls
        )
        np.save(
            'generated_data/corrupted_prob' + vi_name + '_' + model_name + '_' + data_name,
            corrupted_prob
        )
        np.save(
            'generated_data/corrupted_img' + vi_name + '_' + model_name + '_' + data_name,
            corrupted_img
        )


