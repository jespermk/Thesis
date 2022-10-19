import numpy as np
from models.NN import image_model
import torch as t
from VI_models.simple_models import MFG, radial
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# seed
t.manual_seed(72)

data_test = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

data_train = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


# Training set up
device = 'cuda' # -> cuda
iteration_n = t.tensor(10) # -> 100000
batch_size = 128 # -> 256
dataset_size = 60000

loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)

# Radial
m_radial = image_model(radial, device)
m_radial = m_radial.to(device)
optimizer_radial = t.optim.Adam(m_radial.parameters(), amsgrad=True)

# MFG
m_normal = image_model(MFG, device)
m_normal = m_normal.to(device)
optimizer_normal = t.optim.Adam(m_normal.parameters(), amsgrad=True)

# objective
NLL = t.nn.NLLLoss(reduction='none')
NLL_error = t.nn.NLLLoss(reduction='mean')

grad_hist_radial = []
grad_hist_normal = []

grads_radial_std = []
grads_normal_std = []

grads_radial_norm = []
grads_normal_norm = []

error_radial = []
error_normal = []


for i in t.arange(iteration_n):

    batch_x, batch_y = next(iter(loader_train))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # radial
    optimizer_radial.zero_grad()

    y_pred, kl_qp = m_radial( batch_x, kl=True )
    print(y_pred.size())
    nll = NLL(y_pred.reshape([batch_size * 100, 10]), batch_y.repeat_interleave(100)).reshape([batch_size, 100])
    nll = t.mean(nll)
    loss = nll + kl_qp / (dataset_size * batch_size)

    loss.backward()
    optimizer_radial.step()

    # MFG
    optimizer_normal.zero_grad()

    y_pred, kl_qp = m_normal( batch_x, kl=True )

    nll = NLL(y_pred.reshape([batch_size * 100, 10]), batch_y.repeat_interleave(100)).reshape([batch_size, 100])
    nll = t.mean(nll)
    loss = nll + kl_qp / (dataset_size * batch_size)

    loss.backward()
    optimizer_normal.step()


    # collecting data for plots

    batch_x, batch_y = next(iter(loader_test))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # radial
    error_radial += [NLL_error(m_radial(batch_x).mean(1), batch_y).detach().cpu()]

    if i > 100:
        grads_radial = t.cat([p.grad for p in m_radial.parameters()], dim=0).detach().cpu()

        grad_hist_radial += [grads_radial]

        grad_hist_radial = grad_hist_radial[-100:]

        grads_radial_std += [t.stack(grad_hist_radial, dim=0).std(1).norm()]

    else:
        grads_radial = t.cat([p.grad for p in m_radial.parameters()], dim=0).detach().cpu()

        grad_hist_radial += [grads_radial]

        grads_radial_std += [t.stack(grad_hist_radial, dim=0).std(1).norm()]

    # normal
    error_normal += [NLL_error(m_normal(batch_x).mean(1), batch_y).detach().cpu()]

    if i > 100:
        grads_normal = t.cat([p.grad for p in m_normal.parameters()], dim=0).detach().cpu()

        grad_hist_normal += [grads_normal]

        grad_hist_normal = grad_hist_normal[-100:]

        grads_normal_std += [t.stack(grad_hist_normal, dim=0).std(1).norm()]

    else:
        grads_normal = t.cat([p.grad for p in m_normal.parameters()], dim=0).detach().cpu()

        grad_hist_normal += [grads_normal]

        grads_normal_std += [t.stack(grad_hist_normal, dim=0).std(1).norm()]


error_radial = t.stack(error_radial).numpy()
grads_radial_std = t.stack(grads_radial_std).numpy()

error_normal = t.stack(error_normal).numpy()
grads_normal_std = t.stack(grads_normal_std).numpy

np.save('grads_radial_std', grads_radial_std)
np.save('grads_normal_std', grads_normal_std)
np.save('error_radial', error_radial)
np.save('error_normal', error_normal)

#plots
"""
fig, ax = plt.subplots()
plt.plot(error_radial, label='radial error')
ax.set_ylabel('Mean error')
ax.set_xlabel('iterations')
ax.legend()
plt.savefig('radial_error_.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(error_radial, label='radial error')
plt.plot(error_normal, label='MFG error')
ax.set_ylabel('Mean error')
ax.set_xlabel('iterations')
ax.legend()
plt.savefig('MFGvsRadial_error.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(grads_radial_std, label='radial gradiant variance')
plt.plot(grads_normal_std, label='MFG gradiant variance')
ax.set_ylabel('Gradiant Variance')
ax.set_xlabel('iterations')
ax.legend()
plt.savefig('MFGvsRadial_grad_var.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(grads_radial_norm, label='radial gradiant norm')
plt.plot(grads_normal_norm, label='MFG gradiant norm')
ax.set_ylabel('Gradient Norm')
ax.set_xlabel('iterations')
ax.legend()
plt.savefig('MFGvsRadial_grad_norm.png', dpi=300)
plt.show()

# log
fig, ax = plt.subplots()
plt.plot(np.log(error_radial), label='radial error')
plt.plot(np.log(error_normal), label='MFG error')
ax.set_ylabel('Log Mean error')
ax.set_xlabel('iterations')
ax.legend()
plt.savefig('MFGvsRadial_log_error.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(np.log(grads_radial_std), label='radial gradiant variance')
plt.plot(np.log(grads_normal_std), label='MFG gradiant variance')
ax.set_ylabel('Log Gradiant Variance')
ax.set_xlabel('iterations')
ax.legend()
plt.savefig('MFGvsRadial_grad_log_var.png', dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.plot(np.log(grads_radial_norm), label='radial gradiant norm')
plt.plot(np.log(grads_normal_norm), label='MFG gradiant norm')
ax.set_ylabel('Log Gradient Norm')
ax.set_xlabel('iterations')
ax.legend()
plt.savefig('MFGvsRadial_grad_log_norm.png', dpi=300)
plt.show()
"""