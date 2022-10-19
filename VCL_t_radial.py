import numpy as np
from models.NN import image_model, vi_image_model, sigma_T, radial_VCL_image_model
import torch as t
from VI_models.simple_models import t_radial, MFG
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

t.manual_seed(72)

# FASHIONMNIST
transform = transforms.Compose([transforms.ToTensor()])

data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test_set = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

for d, _ in DataLoader(data, batch_size=60000, shuffle=False):
    data_mu, data_sigma = d.mean(0), d.std(0)

# Training set up
if t.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

epoch_n = t.tensor(60)  # -> 60000
batch_size = 1000  # -> 64
dataset_size = 60000/5

# Objective
NLL = t.nn.BCEWithLogitsLoss(reduction='none')
NLL_error = t.nn.BCELoss(reduction='mean')

layer_kind = [
    'aff',
    'aff',
    'aff',
    'aff',
    'aff',
    'aff']


for i in range(5):

    if i == 0:
        radialNN = None

    # set up next task
    shape = [
        [784, 200],
        [200, 200],
        [200, 200],
        [200, 200],
        [200, 200],
        [200, i+1]
    ]
    radialNN = radial_VCL_image_model(t_radial,
                                      layer_kind=layer_kind,
                                      shape=shape,
                                      device=device,
                                      prior=radialNN,
                                      rho=-6)
    radialNN.to(device)
    optimizer_radialNN = t.optim.Adam(radialNN.parameters(), amsgrad=True)

    # will give [0,1], [2,3] ...
    index_1 = (data.train_labels == 2*i).nonzero()
    index_2 = (data.train_labels == 2*i + 1).nonzero()
    index = t.cat([index_1, index_2]).squeeze(1)
    task_set = Subset(data, index)
    loader_train = DataLoader(task_set, batch_size=batch_size, shuffle=True)

    for e in range(epoch_n):

        for batch_x, batch_y in loader_train:

            batch_x, batch_y = ((batch_x - data_mu) / data_sigma).to(device), batch_y.to(device)

            batch_x = batch_x.flatten(start_dim=1)  # So it fits our dense NN
            batch_y = (batch_y == 2*i + 1).float().unsqueeze(1)  # [0,1], [2,3], ... -> [0,1]

            # radial optimization
            optimizer_radialNN.zero_grad()

            x, kl_qp = radialNN(batch_x, train=True)

            nll_vector = NLL(
                x[:, :, i],  # only use the label for the ith task
                batch_y.repeat_interleave(100, dim=1)
            ).mean(1)
            nll_normalized = nll_vector.mean(0)
            loss = nll_normalized + kl_qp / dataset_size

            loss.backward()
            optimizer_radialNN.step()

    radialNN.to('cpu')


correct = []
nll = []
radialNN.to(device)
for i in range(5):
    # will give [0,1], [2,3] ...
    index_1 = (test_set.train_labels == 2 * i).nonzero()
    index_2 = (test_set.train_labels == 2 * i + 1).nonzero()
    index = t.cat([index_1, index_2]).squeeze(1)
    task_set = Subset(data, index)
    loader_test = DataLoader(test_set, batch_size=1, shuffle=True)

    correct_ = []
    nll_ = []
    for batch_x, batch_y in loader_test:
        batch_x, batch_y = ((batch_x - data_mu) / data_sigma).to(device), batch_y.to(device)

        batch_x = batch_x.flatten(start_dim=1)  # So it fits our dense NN
        batch_y = (batch_y == 2 * i + 1).float()  # [0,1], [2,3], ... -> [0,1]

        with t.no_grad():
            p = t.sigmoid(
                radialNN(
                    batch_x
                         )[:, i]
            )
        correct_ += [(t.round(p) == batch_y).float()]
        nll_ += [NLL_error(p, batch_y).unsqueeze(0)]
    correct += [t.cat(correct_).mean().cpu()]
    nll += [t.cat(nll_).mean().cpu()]

correct = t.stack(correct).numpy()
nll = t.stack(nll).numpy()


radial_vcl = np.array([correct, nll])

np.save('generated_data/t_radial_vcl', radial_vcl)









