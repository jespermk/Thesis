from criteria.criteria import Test
from VI_models.simple_models import radial, MFG, exponential_radial, t_radial
import numpy as np
import torch as t


model_names = ['DDN', 'DDN', 'ResNet18', 'ResNet50', 'PreResNet50']

data_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR10', 'CIFAR10']

vi_names = ['non_frac', 'frac', 'gaus']

vi_models = [radial, 'fac', MFG, exponential_radial, t_radial]

stds = ['1', '10', '100']

if t.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

for model_name, data_name in zip(model_names, data_names):
    for vi_name, vi_model in zip(vi_names, vi_models):
        for std in stds:
            print(model_name, data_name, vi_name, std)

            test = Test(
                param='trained_models/m_' + vi_name + '_' + model_name + '_' + data_name + '_std_' + std,
                model=model_name,
                data=data_name,
                vi_model=vi_model,
                use_test_set=False
            )

            acc = test.run_stats({
                'accuracy',
            })

            np.save(
                'generated_data/acc_'+vi_name+'_'+model_name+'_'+data_name+'_std_' + std,
                acc
            )
            print(model_name, data_name)


