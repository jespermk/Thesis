import random

import numpy as np

from models.NN import radialNN, radial_conv2d, radial_aff, radial_model, sigma_T, weighted_ensemble
import torch as t
from tqdm import tqdm
from utils import pred_mean_from_ensemble_model, mse, batch_from_set, shape_to_param_n, sample_from_ensemble
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# seed
t.manual_seed(72)
random.seed(72)

""""
dim_in, dim_out = 5, 3
s = t.tensor([[dim_in, 10], [10, dim_out]])
dim = shape_to_param_n(s)
"""

# Data
"""
nn = NN(s)
data_p = t.rand(dim) / t.sqrt(t.tensor(1/(10) ) )
data_n = t.tensor(200000)
x_data = t.randn(data_n, dim_in) / 2
y_data = nn.evl(x_data,data_p)
"""
data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



# Training set up
device = 'cpu'
iteration_n = t.tensor(100)
batch_size = 16
loader = DataLoader(data, batch_size=16, shuffle=True)

# Radial

"""
# model

m = radialNN(device)
m = m.to(device)
optimizer = t.optim.Adam(m.parameters(), amsgrad=True)

e_singular = []



for i in tqdm(range(iteration_n)):

    batch_x, batch_y = next(iter(loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    optimizer.zero_grad()

    y_pred, kl_qp = m( batch_x )

    nll = NLL(y_pred.reshape([16 * 100, 10]), batch_y.repeat_interleave(100)).reshape([16, 100])
    nll = t.mean(nll)
    loss = nll + kl_qp / (3375 * 16)
    loss.backward()
    optimizer.step()



# Plots

fig, ax = plt.subplots()
plt.plot(e_singular)
ax.set_ylabel('- log likelihood')
ax.set_xlabel('iterations')
plt.savefig('nll_InverseEntropy.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(loss_singular)
ax.set_ylabel('loss')
ax.set_xlabel('iterations')
plt.savefig('loss_InverseEntropy.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(grad_singular)
ax.set_ylabel('Gradient Norm')
ax.set_xlabel('iterations')
plt.savefig('grad_singular_InverseEntropy.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(sigma_singular)
ax.set_ylabel('Sigma Norm')
ax.set_xlabel('iterations')
plt.savefig('sigma_singular_norm_InverseEntropy.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(grad_var)
ax.set_ylabel('Gradient Variance')
ax.set_xlabel('iterations')
plt.savefig('grad_singular_var_InverseEntropy.png', dpi=300)
plt.show()
"""
# Radial

m = radial_model(device)
m = m.to(device)
optimizer = t.optim.Adam(m.parameters(), amsgrad=True)

# Radial ensemble

ensemble_n = t.tensor(3)

ensemble_m = t.nn.ModuleList( [radial_model(device) for i in range(ensemble_n)] )
weighted_ensemble_m = weighted_ensemble(ensemble_m)
ensemble_optimizer = t.optim.Adam(ensemble_m.parameters(), amsgrad=True)
ensemble_m, weighted_ensemble_m = ensemble_m.to(device), weighted_ensemble_m.to(device)
weighted_ensemble_optimizer = t.optim.Adam(weighted_ensemble_m.parameters(), amsgrad=True)
NLL = t.nn.NLLLoss(reduction='none')
NLL_error = t.nn.NLLLoss(reduction='mean')

# For plots

error_ensemble = []
error_weighted_ensemble = []
error_singular = []

for i in tqdm(t.arange(iteration_n)):

    batch_x, batch_y = next(iter(loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    index = t.fmod(i, ensemble_n).long()

    # Singular

    optimizer.zero_grad()

    y_pred, kl_qp = m( batch_x, kl=True )

    nll = NLL(y_pred.reshape([16 * 100, 10]), batch_y.repeat_interleave(100)).reshape([16, 100])
    nll = t.mean(nll)
    loss = nll + kl_qp / (3375 * 16)
    loss.backward()
    optimizer.step()

    # Ensemble
    ensemble_optimizer.zero_grad()

    y_pred, kl_qp = ensemble_m[index](batch_x, kl=True)
    nll = NLL(y_pred.reshape([16 * 100, 10]), batch_y.repeat_interleave(100)).reshape([16, 100])
    nll = t.mean(nll)
    loss = nll + kl_qp / (3375 * 16)

    loss.backward()
    ensemble_optimizer.step()


    # Weighted Ensemble
    weighted_ensemble_optimizer.zero_grad()

    y_pred = weighted_ensemble_m(batch_x, ensemble_m).mean(1)
    loss = NLL_error(y_pred, batch_y)

    loss.backward()
    weighted_ensemble_optimizer.step()



    # Data

    batch_x, batch_y = next(iter(loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    y_pred = m(batch_x).mean(1)
    y_pred_ensemble = t.stack([ensemble_m[i](batch_x) for i in t.arange(ensemble_n)]).mean([0,2])
    y_pred_weighted_ensemble = weighted_ensemble_m(batch_x, ensemble_m).mean(1)

    error_weighted_ensemble += [ NLL_error(y_pred_weighted_ensemble, batch_y) ]
    error_ensemble += [ NLL_error(y_pred_ensemble.mean(1), batch_y) ]
    error_singular += [ NLL_error(y_pred, batch_y) ]

#Save data
t.save(error_weighted_ensemble, 'error_weighted_ensemble.pt')
t.save(error_ensemble, 'error_ensemble.pt')
t.save(error_singular, 'error_singular.pt')



fig, ax = plt.subplots()
plt.plot(error_weighted_ensemble, label='weighted ensemble radial')
plt.plot(error_ensemble, label='ensemble radial')
plt.plot(error_singular, label='singular radial')
ax.set_ylabel('predictive Error')
ax.set_xlabel('iterations')
ax.legend()
plt.savefig('error.png', dpi=300)
plt.show()


plt.plot(e_ensemble)
plt.savefig('error_ensemble.png', dpi=300)

plt.plot(loss_ensemble)
plt.savefig('loss_ensemble.png', dpi=300)
plt.plot()

fig, axs = plt.subplots(2, 1)

axs[0].plot(e_singular, label='Singular Radial')
axs[0].plot(e_ensemble, label='Ensemble Radial')
axs[0].set_xlabel('Batch Iteration')
axs[0].set_ylabel('Mean Square Error')
axs[0].legend()

axs[1].plot(loss_singular, label='Singular Radial')
axs[1].plot(loss_ensemble, label='Ensamble Radial')
axs[1].set_xlabel('Batch Iteration')
axs[1].set_ylabel('Loss')
axs[1].legend()

plt.tight_layout()

plt.savefig('ErrorAndLoss_ensemble_vs_singular.png', dpi=350)
plt.show()
"""