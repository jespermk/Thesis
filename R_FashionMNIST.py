import numpy as np
from models.NN import radial_VCL_image_model, sigma_T
import torch as t
from VI_models.simple_models import MFG, radial
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import tensorflow as tf
import tensorflow_probability as tfp

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

# Training set up
if t.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
epoch_n = t.tensor(150)  # -> 120000
batch_size = 100  # -> 64
dataset_size = 60000

# Normal
model = 'radial_FM'
m_normal = radial_VCL_image_model(radial, device=device, other_prior=MFG)
m_normal.to(device)
optimizer_normal = t.optim.Adam(m_normal.parameters(), amsgrad=True)

# Objective
NLL = t.nn.NLLLoss(reduction='none')
NLL_error = t.nn.NLLLoss(reduction='mean')

# Data loaders and data size
loader_train = DataLoader(data, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(test_set, batch_size=100, shuffle=True)


# Data bins
grad_hist_normal = []
grads_normal_std = []
error_normal = []
correct_rate = []
loss_normal = []
ece = []

norm_sigma_normal = []
norm_mu_normal = []

mean_sigma_normal = []
mean_mu_normal = []

norm_mu_grad_normal = []
norm_sigma_grad_normal = []

for i in range(epoch_n):

    for batch_x, batch_y in loader_train:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Normal optimization
        optimizer_normal.zero_grad()

        y_pred, kl_qp = m_normal(batch_x, train=True)
        y_pred = F.log_softmax(y_pred, dim=-1)
        nll_vector = NLL(
            y_pred.reshape([batch_size * 100, 10]),
            batch_y.repeat_interleave(100)
        ).reshape([batch_size, 100]).mean(1)
        nll_normalized = nll_vector.mean(0)
        loss = nll_normalized + kl_qp / dataset_size

        loss.backward()
        optimizer_normal.step()

        # Collecting data for plots

        loss_normal += [loss.detach().cpu()]

        # Collecting 'parameters'
        rho_ws = m_normal.rho_w
        rho_bs = m_normal.rho_b

        mu_ws = m_normal.mu_w
        mu_bs = m_normal.mu_b

        mu = t.cat([mu_ws, mu_bs]).detach()
        sigma = sigma_T(t.cat([rho_ws, rho_bs]).detach())

        mu_grad = t.cat([mu_ws.grad, mu_bs.grad]).detach()
        rho_grad = t.cat([rho_ws.grad, rho_bs.grad]).detach()

        if len(grad_hist_normal) >= 100:
            grads_normal = t.cat([mu_grad, rho_grad], dim=0).cpu()

            grad_hist_normal += [grads_normal]

            grad_hist_normal = grad_hist_normal[-100:]

            # Binning standard deviation of grads
            grads_normal_std += [t.stack(grad_hist_normal, dim=0).std(1).norm().cpu()]

        else:
            grads_normal = t.cat([mu_grad, rho_grad], dim=0).cpu()

            grad_hist_normal += [grads_normal]

            # Binning standard deviation of grads
            grads_normal_std += [t.stack(grad_hist_normal, dim=0).std(1).norm().cpu()]

        # Binning norm mu and sigma
        norm_mu_normal += [mu.norm().cpu()]
        norm_sigma_normal += [sigma.norm().cpu()]

        # Binning mean mu and sigma
        mean_mu_normal += [mu.mean().cpu()]
        mean_sigma_normal += [sigma.mean().cpu()]

        # Binning grads for mu and sigma
        norm_mu_grad_normal += [mu_grad.norm().cpu()]
        norm_sigma_grad_normal += [rho_grad.norm().cpu()]

    # Binning error and loss

    logits_ = []
    true_ = []
    error_normal_ = []
    correct_rate_ = []

    for batch_x, batch_y in loader_test:
        with t.no_grad():

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = m_normal(batch_x)

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

# Convert to numpy
error_normal = t.stack(error_normal).numpy()
correct_rate = t.stack(correct_rate).numpy()
ece = np.stack(ece)
loss_normal = t.stack(loss_normal).numpy()

norm_sigma_normal = t.stack(norm_sigma_normal).numpy()
norm_mu_normal = t.stack(norm_mu_normal).numpy()

mean_sigma_normal = t.stack(mean_sigma_normal).numpy()
mean_mu_normal = t.stack(mean_mu_normal).numpy()

norm_mu_grad_normal = t.stack(norm_mu_grad_normal).numpy()
norm_sigma_grad_normal = t.stack(norm_sigma_grad_normal).numpy()

grads_normal_std = t.stack(grads_normal_std).numpy()

# Save as numpy
np.save('generated_data/error_%s' % model, error_normal)
np.save('generated_data/correct_rate_%s' % model, correct_rate)
np.save('generated_data/ece_%s' % model, ece)
np.save('generated_data/loss_%s' % model, loss_normal)

np.save('generated_data/norm_sigma_%s' % model, norm_sigma_normal)
np.save('generated_data/norm_mu_%s' % model, norm_mu_normal)

np.save('generated_data/mean_sigma_%s' % model, mean_sigma_normal)
np.save('generated_data/mean_mu_%s' % model, mean_mu_normal)

np.save('generated_data/norm_mu_grad_%s' % model, norm_mu_grad_normal)
np.save('generated_data/norm_sigma_grad_%s' % model, norm_sigma_grad_normal)

np.save('generated_data/grads_%s_std' % model, grads_normal_std)

# Save model
t.save(m_normal, 'trained_models/m_%s' % model)
