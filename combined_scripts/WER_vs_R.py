import numpy as np
from models.NN import image_model, weighted_ensemble
from VI_models.simple_models import radial
import torch as t
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Subset


# seed
t.manual_seed(72)

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
    train=True,
    download=True,
    transform=ToTensor()
)

test_set = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

"""
# train 1000
train_size = 1000
train_index = t.arange(train_size)
test_index = t.arange(train_size,data_size)
sub1000_train, sub1000_test = Subset(data, train_index), Subset(data, test_index)


# train 200
train_size= 200
train_index = t.arange(train_size)
test_index = t.arange(train_size, data_size)
sub200_train, sub200_test = Subset(data, train_index), Subset(data, test_index)

# train 50
train_size= 50
train_index = t.arange(train_size)
test_index = t.arange(train_size, data_size)
sub50_train, sub50_test = Subset(data, train_index), Subset(data, test_index)

data_sets = [ [sub1000_train, sub1000_test],
              [sub200_train, sub200_test],
              [sub50_train, sub50_test]    ]

sessions = ['1000samples', '200samples', '50samples']
"""
# train 60000
train_size = 60000
train_index = t.arange(train_size)
sub60000_train = Subset(data, train_index)

# train 30000
train_size = 30000
train_index = t.arange(train_size)
sub30000_train = Subset(data, train_index)

# train 5000
train_size = 5000
train_index = t.arange(train_size)
sub5000_train = Subset(data, train_index)

# train 1000
train_size= 1000
train_index = t.arange(train_size)
sub1000_train = Subset(data, train_index)

# train 256
train_size= 256
train_index = t.arange(train_size)
sub256_train = Subset(data, train_index)



data_sets = [
              sub256_train,
              sub1000_train,
              sub5000_train,
              sub30000_train,
              sub60000_train
   ]

sessions = [ '256samples', '1000samples', '5000samples','30000samples','60000samples']
dataset_sizes = [256, 1000, 5000, 30000, 60000]

# Training set up
device = 'cuda' # -> cuda
iteration_n = t.tensor(10) # -> 100000
batch_size = 64 # -> 64

ensemble_n = t.tensor(4)

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


for s in range(len(sessions)):
    loader_train = DataLoader(data_sets[s], batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    dataset_size = dataset_sizes[s]
    # Radial

    m = image_model(radial, device)
    m = m.to(device)
    optimizer = t.optim.Adam(m.parameters(), amsgrad=True)

    # Radial ensemble / Weighted radial ensemble

    ensemble_m = t.nn.ModuleList( [image_model(radial, device) for i in range(ensemble_n)] )
    weighted_ensemble_m = weighted_ensemble(ensemble_m)
    ensemble_optimizer = t.optim.Adam(ensemble_m.parameters(), amsgrad=True)
    ensemble_m, weighted_ensemble_m = ensemble_m.to(device), weighted_ensemble_m.to(device)
    weighted_ensemble_optimizer = t.optim.Adam(weighted_ensemble_m.parameters(), amsgrad=True)
    NLL = t.nn.NLLLoss(reduction='none')
    NLL_ME = t.nn.NLLLoss(reduction='mean')

    # For plots

    error_ensemble = []
    error_weighted_ensemble = []
    error_singular = []

    for i in t.arange(iteration_n):

        batch_x, batch_y = next(iter(loader_train))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        index = t.fmod(i, ensemble_n).long()

        # Singular
        optimizer.zero_grad()

        y_pred, kl_qp = m( batch_x, kl=True )
        nll = NLL(y_pred.reshape([batch_size * 100, 10]), batch_y.repeat_interleave(100)).reshape([batch_size, 100])
        nll = t.mean(nll)
        loss = nll + kl_qp / (dataset_size * batch_size)

        loss.backward()
        optimizer.step()

        # Ensemble
        ensemble_optimizer.zero_grad()

        y_pred, kl_qp = ensemble_m[index](batch_x, kl=True)
        nll = NLL(y_pred.reshape([batch_size * 100, 10]), batch_y.repeat_interleave(100)).reshape([batch_size, 100])
        nll = t.mean(nll)
        loss = nll + kl_qp / (dataset_size * batch_size)

        loss.backward()
        ensemble_optimizer.step()


        # Weighted Ensemble
        weighted_ensemble_optimizer.zero_grad()

        y_pred = weighted_ensemble_m(batch_x, ensemble_m).mean(1)
        loss = NLL_ME(y_pred, batch_y)

        loss.backward()
        weighted_ensemble_optimizer.step()


        # Data


        batch_x, batch_y = next(iter(loader_test))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)


        y_pred = m(batch_x).detach().mean(1)
        y_pred_ensemble = t.stack([ensemble_m[i](batch_x) for i in t.arange(ensemble_n)]).detach().mean([0,2])
        y_pred_weighted_ensemble = weighted_ensemble_m(batch_x, ensemble_m).detach().mean(1)

        error_weighted_ensemble += [ NLL_ME(y_pred_weighted_ensemble, batch_y).cpu() ]
        error_ensemble += [ NLL_ME(y_pred_ensemble, batch_y).cpu() ]
        error_singular += [ NLL_ME(y_pred, batch_y).cpu() ]

    error_weighted_ensemble = t.stack(error_weighted_ensemble).numpy()
    error_ensemble = t.stack(error_ensemble).numpy()
    error_singular = t.stack(error_singular).numpy()



    #Save data
    np.save('error_weighted_ensemble_%s' %sessions[s], error_weighted_ensemble)
    np.save('error_ensemble_%s' %sessions[s], error_ensemble)
    np.save('error_singular_%s' %sessions[s], error_singular)


    """
    fig, ax = plt.subplots()
    plt.plot(error_weighted_ensemble, label='weighted ensemble radial %s' %sessions[s])
    plt.plot(error_ensemble, label='ensemble radial %s' %sessions[s])
    plt.plot(error_singular, label='singular radial %s' %sessions[s])
    ax.set_ylabel('predictive Error')
    ax.set_xlabel('iterations')
    ax.legend()
    plt.savefig('error_%s.png' %sessions[s], dpi=300)

    plt.clf()
    plt.cla()
    """



# do not include

"""

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