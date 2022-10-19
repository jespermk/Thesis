import torch as t
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
from torchvision import datasets
from sklearn.calibration import CalibrationDisplay
from torch.utils.data import DataLoader
from sklearn.base import ClassifierMixin
from sklearn.calibration import calibration_curve
from tqdm import tqdm

class clibration_test(ClassifierMixin):
    def __init__(self, model):

        self.model = model

    def prob(self, x):
        with t.no_grad():
            proba = F.softmax(self.model(x), dim=-1).cpu().numpy()
        return proba


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + t.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

"""
# FASHIONMNIST
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                AddGaussianNoise(mean=0, std=0.25,)])

test_set = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)
loader_test = DataLoader(test_set, batch_size=100, shuffle=True)
"""

folder = 'trained_models/'
models_loc = ['m_normal_FM_less',
              'm_radial_FM_less']
[Normal,
 Radial,] = [t.load(
    folder + m,
    map_location=t.device('cpu')
    ) for m in models_loc]
clf_list = [
    (Normal, "Normal"),
    (Radial, "Radial"),
]
folder = 'generated_data/'

for clf, name in clf_list:
    clf = clibration_test(clf)
    y_true_ = []
    y_prob_ = []

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    AddGaussianNoise(mean=0, std=0.5, )])

    for it in range(15):

        test_set = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=transform
        )
        loader_test = DataLoader(test_set, batch_size=100, shuffle=True)
        for x, y in tqdm(loader_test):
            y_prob_ += [clf.prob(x)]
            y_true_ += [y.cpu().numpy()]

    y_prob = np.concatenate(y_prob_, 0)
    y_true = np.concatenate(y_true_, 0)


    np.save(folder + 'y_prob_' + name, y_prob)
    np.save(folder + 'y_true_' + name, y_true)



