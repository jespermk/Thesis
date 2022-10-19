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
from sklearn.metrics import ConfusionMatrixDisplay

def calibration_plot(y_true, y_prob):
    n_classes = 10
    class_list = np.arange(n_classes)
    x = []
    y = []
    prob = []
    for cls in class_list:
        y_true_ = (y_true == cls).astype(float)
        y_prob_ = y_prob[:, cls]
        y_, x_ = calibration_curve(y_true_, y_prob_, n_bins=5)
        prob += [y_prob_]
        x += [x_]
        y += [y_]
        print(np.shape(y_prob_),np.shape(x_),np.shape(y_))
    return np.stack(x), np.stack(y), np.stack(prob).flatten()


def plot_calibration_curve(y_true, y_prob, n_bins=5, ax=None, hist=True, normalize=False):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, normalize=normalize)
    if ax is None:
        ax = plt.gca()
    if hist:
        ax.hist(y_prob, weights=np.ones_like(y_prob) / len(y_prob), alpha=.4,
               bins=np.maximum(10, n_bins))
    ax.plot([0, 1], [0, 1], ':', c='k')
    curve = ax.plot(prob_pred, prob_true, marker="o")

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("fraction of positive samples")

    ax.set(aspect='equal')
    return curve


folder = 'generated_data/'
models = ["Normal",
          "Radial",
          'tRadial',
          'expRadial']

prob = [np.load(folder + 'y_prob_%s.npy' % m) for m in models]
true = [np.load(folder + 'y_true_%s.npy' % m) for m in models]

model = 1


x, y, probs = calibration_plot(true[model], prob[model])


fig, ax = plt.subplots()
ax.plot(np.mean(x,0), np.mean(y,0), c='b')
[ax.plot(x[i], y[i], alpha=0.2, c='y') for i in range(10)]
ax.plot(np.arange(0,1, 0.01), np.arange(0,1, 0.01), ls='--', c='r')
ax.fill_between(x[0],
                          (np.mean(y,0) - np.std(y,0)),
                          (np.mean(y,0) + np.std(y,0)),
                            color='b',
                            alpha=.1)
counts, edges, bars = plt.hist(probs, weights=np.ones_like(probs) / len(probs), bins=10)
plt.bar_label(bars, fmt='%.e')

plt.show()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
ConfusionMatrixDisplay.from_predictions(true[model], np.argmax(prob[model], axis=-1), normalize='pred', display_labels=class_labels)
plt.show()

