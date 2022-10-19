import numpy as np
from models.NN import CLImageDN
import torch as t
from VI_models.simple_models import radial, MFG, exponential_radial, t_radial
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

######################
prior = 't'
vi_model = t_radial
name = 't'
#######################

t.manual_seed(72)

# FASHIONMNIST
transform = transforms.Compose([transforms.ToTensor()])
data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test = datasets.FashionMNIST(
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

dataset_size = 60000/5
result_hyp_params = []
# Objective
NLL = t.nn.BCEWithLogitsLoss(reduction='none')
NLL_error = t.nn.BCELoss(reduction='mean')

result = np.load('generated_data/VCL_' + name + '.npy')
print(result[:, 0])
argmin = np.argmin(result[:, 0])
mse, batch_size, epoch_n, vi_s, lr, std = result[argmin]


batch_size = int(batch_size)
epoch_n = int(epoch_n)
vi_s = int(vi_s)
M = CLImageDN(vi_model=vi_model,
              device=device,
              std=std,
              vi_samples_n=vi_s).to(device)

optimizer = t.optim.SGD(M.parameters(), momentum=0.9, lr=lr)

for i in range(5):

    if i > 0:
        M.con_learn(vi_model=radial,
                    expansion=1,
                    prior=prior)

    # will give [0,1], [2,3] ...
    index_1 = (data.train_labels == 2*i).nonzero()
    index_2 = (data.train_labels == 2*i + 1).nonzero()
    index = t.cat([index_1, index_2]).squeeze(1)
    task_set = Subset(data, index)
    test_size = 1200
    t.manual_seed(42)
    train, _ = random_split(task_set, [12000 - 1200, 1200])
    loader_train = DataLoader(task_set, batch_size=batch_size, shuffle=True, drop_last=True)

    for e in range(epoch_n):

        for batch_x, batch_y in tqdm(loader_train):

            batch_x, batch_y = ((batch_x - data_mu) / data_sigma).to(device), batch_y.to(device)

            batch_x = batch_x.flatten(start_dim=1)  # So it fits our dense NN
            batch_y = (batch_y == 2 * i + 1).float().unsqueeze(1)  # [0,1], [2,3], ... -> [0,1]

            # radial optimization
            optimizer.zero_grad()

            x, kl_qp = M(batch_x, train=True)

            nll_vector = NLL(
                x[:, :, i],  # only use the label for the ith task
                batch_y.repeat_interleave(M.vi_samples_n, dim=1)
            ).mean(1)
            nll_normalized = nll_vector.mean(0)
            loss = nll_normalized + kl_qp / (len(loader_train) * batch_size)

            loss.backward()
            optimizer.step()

correct = []
nll = []
M.vi_samples_n = 100

for i in range(5):
    # will give [0,1], [2,3] ...
    index_1 = (test.train_labels == 2 * i).nonzero()
    index_2 = (test.train_labels == 2 * i + 1).nonzero()
    index = t.cat([index_1, index_2]).squeeze(1)
    task_set = Subset(test, index)
    loader_test = DataLoader(task_set, batch_size=200, shuffle=True)

    correct_ = []
    nll_ = []
    with t.no_grad():
        for batch_x, batch_y in loader_test:
            batch_x, batch_y = ((batch_x - data_mu) / data_sigma).to(device), batch_y.to(device)

            batch_x = batch_x.flatten(start_dim=1)  # So it fits our dense NN
            batch_y = (batch_y == 2 * i + 1).double()  # [0,1], [2,3], ... -> [0,1]

            p = t.sigmoid(M(batch_x)[:, :, i]).mean(dim=1)

            correct_ += [(t.round(p).int() == batch_y).float()]
        correct += [t.cat(correct_).mean().cpu()]

correct = t.stack(correct).numpy()

np.save('generated_data/VCL_correct_%s' % name, correct)






