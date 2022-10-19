import numpy as np
from models.NN import radial_VCL_image_model, ensemble_model
from VI_models.simple_models import radial, MFG
import torch as t
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_probability as tfp
from torch.utils.data import Subset

# Seed
t.manual_seed(72)

# FASHIONMNIST
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
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

index = t.randperm(60000)[:6000]
data = Subset(data, index)

# Training set up
if t.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

epoch_n = t.tensor(35)  # -> 120000
batch_size = 100  # -> 64
ensemble_n = t.tensor(4)  # -> 64

# Objective
NLL = t.nn.NLLLoss(reduction='none')
NLL_error = t.nn.NLLLoss(reduction='mean')

# Data loaders and data size
loader_train = DataLoader(data, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(test_set, batch_size=100, shuffle=True)
dataset_size = 60000

# Weighted radial ensemble
model = 'WER_FM_less'
radials = [radial_VCL_image_model(radial, device=device, other_prior=MFG) for i in range(ensemble_n)]
em = ensemble_model(radials, device, weighted=True)
ensemble_optimizer = [t.optim.Adam(em.ensemble[i].parameters(), amsgrad=True) for i in range(ensemble_n)]
weights_optimizer = t.optim.Adam(em.parameters(), amsgrad=True)


# Data bins
error_normal = []
correct_rate = []
ece = []

for e in t.arange(epoch_n):

    for index in t.arange(ensemble_n):
        for batch_x, batch_y in loader_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

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

            # Clean up GPU
            em.ensemble[index].to('cpu')
            t.cuda.empty_cache()

    for batch_x, batch_y in loader_train:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Weights optimization
        em.to(device)
        weights_optimizer.zero_grad()

        y_pred = F.log_softmax(em(batch_x), dim=-1)
        loss = NLL_error(y_pred, batch_y)

        loss.backward()
        weights_optimizer.step()

        # Clean up GPU
        em.to('cpu')
        t.cuda.empty_cache()

    # Collecting data for plots

    # Binning data
    em.to(device)

    logits_ = []
    true_ = []
    error_normal_ = []
    correct_rate_ = []

    for batch_x, batch_y in loader_test:
        with t.no_grad():

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = em(batch_x)

            pred = F.log_softmax(logits, dim=-1)

            error_normal_ += [NLL_error(pred, batch_y).detach().cpu()]
            correct_rate_ += [(pred.argmax(dim=-1) == batch_y).float().mean().cpu()]

            logits_ += [logits]
            true_ += [batch_y]

    ece += [tfp.stats.expected_calibration_error(10,
                                                 logits=tf.convert_to_tensor(
                                                     t.stack(logits_).flatten(end_dim=-2).cpu().numpy()),
                                                 labels_true=tf.convert_to_tensor(
                                                     t.cat(true_).flatten(end_dim=-1).cpu().numpy())
                                                 ).numpy()]

    error_normal += [t.stack(error_normal_).mean().cpu()]
    correct_rate += [t.stack(correct_rate_).mean().cpu()]

    em.to('cpu')

# Convert to numpy
error_normal = t.stack(error_normal).numpy()
correct_rate = t.stack(correct_rate).numpy()
ece = np.stack(ece)

# Save data
np.save('generated_data/error_%s' % model, error_normal)
np.save('generated_data/correct_rate_%s' % model, correct_rate)
np.save('generated_data/ece_%s' % model, ece)

# Save model
t.save(em, 'trained_models/m_%s' % model)
