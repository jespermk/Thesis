import torch as t
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch.nn.functional as F



if t.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

test_set = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

nlll_eval = t.nn.NLLLoss()

module_location = [

    'trained_models/m_normal',
    'trained_models/m_radial',
    'trained_models/m_t_radial',
    'trained_models/m_exp_radial',

    'trained_models/m_normal_FM',
    'trained_models/m_radial_FM',
    'trained_models/m_t_radial_FM',
    'trained_models/m_exp_radial_FM',

]
models = [t.load(module_location[i]) for i in range(len(module_location))]

correct = []
nlll = []
ece = []
model_stat = []
for m in range(len(models)):

    t.manual_seed(72)
    loader_test = DataLoader(test_set, batch_size=1, shuffle=True)

    pred_ = []
    true_ = []
    nlll_ = []
    correct_ = []

    models[m].to(device)
    for i in range(10000):
        batch_x, batch_y = next(iter(loader_test))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        with t.no_grad():
            p, target = F.log_softmax(models[m](batch_x), dim=-1), batch_y

        correct_ += [(p.argmax(dim=-1) == target).float()]
        nlll_ += [nlll_eval(p, target).unsqueeze(0)]

        pred_ += [p.squeeze(0)]
        true_ += [target]
    models[m].to('cpu')

    correct += [t.cat(correct_).mean().cpu().numpy()]
    nlll += [t.cat(nlll_).mean().cpu().numpy()]
    ece += [tfp.stats.expected_calibration_error(10,
                                                 logits=tf.convert_to_tensor(t.stack(pred_).cpu().numpy()),
                                                 labels_true=tf.convert_to_tensor(t.cat(true_).cpu().numpy())
                                                 ).numpy()]

model_stat = np.array([correct, nlll, ece])
np.save('generated_data/model_stat', model_stat)

"""
normal_nlll = t.stack(normal_nlll).mean().cpu().numpy()
radial_nlll = t.stack(radial_nlll).mean().cpu().numpy()
ensemble_nlll = t.stack(ensemble_nlll).mean().cpu().numpy()
weighted_ensemble_nlll = t.stack(weighted_ensemble_nlll).mean().cpu().numpy()

nllls = np.array([normal_nlll, radial_nlll, ensemble_nlll, weighted_ensemble_nlll])
print(nllls)

np.save('generated_data/nllls_6e2samples', nllls)


normal_ = tf.convert_to_tensor(t.stack(normal_).cpu().numpy())
radial_ = tf.convert_to_tensor(t.stack(radial_).cpu().numpy())
ensemble_ = tf.convert_to_tensor(t.stack(ensemble_).cpu().numpy())
weighted_ensemble_ = tf.convert_to_tensor(t.stack(weighted_ensemble_).cpu().numpy())
true_ = tf.convert_to_tensor(t.cat(true_).cpu().numpy())


ECE_normal = tfp.stats.expected_calibration_error(15, normal_, true_).numpy()
ECE_radial = tfp.stats.expected_calibration_error(15, radial_, true_).numpy()
ECE_ensemble = tfp.stats.expected_calibration_error(15, ensemble_, true_).numpy()
ECE_weighted_ensemble = tfp.stats.expected_calibration_error(15, weighted_ensemble_, true_).numpy()

ECEs = np.array([ECE_normal, ECE_radial, ECE_ensemble, ECE_weighted_ensemble])

np.save('generated_data/ECEs_6e2samples', ECEs)
"""




