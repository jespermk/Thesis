import numpy as np
from models.NN import radial_VCL_image_model
import torch as t
from VI_models.simple_models import radial, t_radial, exponential_radial, MFG

t.manual_seed(0)


def noisy_function(x, trans):
    x = x ** (0.4)

    x = x - t.min(x)

    x = x / t.max(x)

    b = t.randn(x.size()) / 10

    return x + trans, t.sin(3 * (x + trans)) + b


# for cvl regression

# radial model
# Training set up
if t.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

layer_kind = [
    'aff',
    'aff',
    'aff',
    'aff',
    'aff',
    'aff'
]

nll = t.nn.GaussianNLLLoss(reduction='none')
it_n = t.tensor(40000)  # <- 35000
model = radial
prior = None

gen_data_x_ = []
gen_data_y_ = []

for i in range(4):

    if i == 0:
        radialNN = None

    shape = [
        [1, 200],
        [200, 200],
        [200, 200],
        [200, 200],
        [200, 200],
        [200, 1+i]
    ]

    radialNN = radial_VCL_image_model(model,
                                      layer_kind=layer_kind,
                                      shape=shape,
                                      device=device,
                                      prior=radialNN)

    radialNN.to(device)
    optimizer_radialNN = t.optim.Adam(radialNN.parameters(), amsgrad=True)

    # generate data

    gen = t.rand(5000)
    x_data, y_data = noisy_function(gen, 2*i)

    gen_data_x_ += [x_data]
    gen_data_y_ += [y_data]

    gen_data_x = t.stack(gen_data_x_).numpy()
    gen_data_y = t.stack(gen_data_y_).numpy()

    x = x_data.unsqueeze(1)
    y = y_data.unsqueeze(1)

    dataset_size = x.size(dim=0)
    var = t.ones(dataset_size, device=device)

    for it in range(it_n):
        # training

        optimizer_radialNN.zero_grad()

        y_pred, kl_qp = radialNN(
            x.to(device),
            train=True)

        y_pred = y_pred[:, :, i]
        y_ = y.to(device).repeat_interleave(100, dim=1)

        nll_vector = nll(
            y_pred,
            y_,
            var / 2
        ).mean(1)

        nll_normalized = nll_vector.mean(0)
        loss = nll_normalized + (kl_qp / dataset_size)

        loss.backward()
        optimizer_radialNN.step()


    t.save(radialNN, 'trained_models/2d_reg_R_m_%i' % i)

    np.save('generated_data/2d_reg_R_gen_data_x_%i' % i, gen_data_x)
    np.save('generated_data/2d_reg_R_gen_data_y_%i' % i, gen_data_y)

    pred_mean_ = []
    pred_std_ = []
    pred_interval_ = []

    for j in range(i+1):

        x_ = t.arange(2*j-0.5, 2*(j+1)-0.5, 0.01, device=device).unsqueeze(1)

        with t.no_grad():
            y_, _ = radialNN(x_, train=True)

        y_ = y_[:,:,j]

        pred_mean_ += [y_.mean(dim=1).cpu()]
        pred_std_ += [y_.std(dim=1).cpu()]
        pred_interval_ += [x_.cpu()]



    pred_mean = t.stack(pred_mean_).numpy()
    pred_std = t.stack(pred_std_).numpy()
    pred_interval = t.stack(pred_interval_).numpy()

    np.save('generated_data/2d_reg_R_pred_mean_%i' % i, pred_mean)
    np.save('generated_data/2d_reg_R_pred_std_%i' % i, pred_std)
    np.save('generated_data/2d_reg_R_pred_interval_%i' % i, pred_interval)
    radialNN.to('cpu')







