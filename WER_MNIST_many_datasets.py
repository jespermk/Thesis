import numpy as np
from models.NN import radial_VCL_image_model, ensemble_model
from VI_models.simple_models import radial, MFG
import torch as t
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F

# Seed
t.manual_seed(72)

# MNIST
data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_set = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Data set experiments

# Train 1000
train_size = 1000
train_index = t.arange(train_size)
sub1000_train = Subset(data, train_index)

# Train 500
train_size = 500
train_index = t.arange(train_size)
sub500_train = Subset(data, train_index)

# Train 250
train_size = 250
train_index = t.arange(train_size)
sub250_train = Subset(data, train_index)

# Train 125
train_size = 125
train_index = t.arange(train_size)
sub125_train = Subset(data, train_index)


data_sets = [
    sub1000_train,
    sub500_train,
    sub250_train,
    sub125_train
]

sessions = ['1000samples', '500samples', '250samples', '125samples']
dataset_sizes = [1000, 500, 250, 125]

# Training set up
if t.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
iteration_n = t.tensor(60000)  # -> 60000
batch_size = 64  # -> 64
ensemble_n = t.tensor(4)

# Objective
NLL = t.nn.NLLLoss(reduction='none')
NLL_ME = t.nn.NLLLoss(reduction='mean')

for s in range(len(sessions)):

    # Data loaders and data size
    loader_train = DataLoader(data_sets[s], batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    dataset_size = dataset_sizes[s]

    # Radial ensemble
    radials = [radial_VCL_image_model(radial, device=device, other_prior=MFG) for i in range(ensemble_n)]
    em = ensemble_model(radials, device, weighted=True)
    ensemble_optimizer = [t.optim.Adam(em.ensemble[i].parameters(), amsgrad=True) for i in range(ensemble_n)]
    weights_optimizer = t.optim.Adam(em.parameters(), amsgrad=True)

    # Data bins
    error_weighted_ensemble = []

    for i in t.arange(iteration_n):

        batch_x, batch_y = next(iter(loader_train))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        index = t.fmod(i, ensemble_n).long()

        # Ensemble optimization
        em.ensemble[index].to(device)
        ensemble_optimizer[index].zero_grad()

        y_pred, kl_qp = em.ensemble[index](batch_x, train=True)
        y_pred = F.log_softmax(y_pred, dim=-1)
        nll_vector = NLL(
            y_pred.reshape([batch_size * 100, 10]),
            batch_y.repeat_interleave(100)
        ).reshape([batch_size, 100]).mean(1)
        nll_normalized = nll_vector.mean(0)
        loss = nll_normalized + kl_qp / dataset_size

        loss.backward()
        ensemble_optimizer[index].step()

        # Cleaning up GPU
        em.ensemble[index].to('cpu')
        t.cuda.empty_cache()

        # Weights optimization
        em.to(device)
        weights_optimizer.zero_grad()

        y_pred = F.log_softmax(em(batch_x), dim=-1)
        loss = NLL_ME(y_pred, batch_y)

        loss.backward()
        weights_optimizer.step()

        # Clean up GPU
        em.to('cpu')
        t.cuda.empty_cache()

        # Collecting data for plots

        batch_x, batch_y = next(iter(loader_test))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Collecting prediction form radial models

        # Binning data
        em.to(device)
        error_weighted_ensemble += [NLL_ME(F.log_softmax(em(batch_x), dim=-1), batch_y).detach().cpu()]
        em.to('cpu')

    # Convert to numpy
    error_weighted_ensemble = t.stack(error_weighted_ensemble).numpy()

    # Save data
    np.save('generated_data/error_weighted_ensemble_%s' % sessions[s], error_weighted_ensemble)

    # Save model
    t.save(em, 'trained_models/weighted_ensemble_m_%s' % sessions[s])
