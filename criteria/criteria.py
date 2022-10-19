import torch as t
import numpy as np
from models.NN import ResNet, sigma_T, Block, Bottleneck, ImageDN
from VI_models.simple_models import MFG, radial
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.calibration import calibration_curve
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError as mulclasscalerr

t.manual_seed(42)

class Test(t.nn.Module):
    def __init__(self, param=None, model=None, data=None, vi_model=None, use_test_set=False):
        super(Test, self).__init__()

        if t.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if model == 'DDN':
            self.M = ImageDN(
                vi_model=vi_model,  # <-
                device=self.device,
                std=1,
                vi_samples_n=1,
            )

        else:
            if model == 'ResNet18':
                repeats = [2, 2, 2, 2]
                PreAc = False
                archi = Block
            else:
                repeats = [3, 4, 6, 3]
                archi = Bottleneck
                if model == 'PreResNet50':
                    PreAc = True
                else:
                    PreAc = False

            self.M = ResNet(
                vi_model=vi_model,
                device=self.device,
                vi_samples_n=1,
                c_in=3,
                out_dim=10,
                std=10,
                resblock=archi,
                repeats=repeats,
                PreAc=PreAc
            )

        if data in {'FashionMNIST', 'MNIST'}:

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])

            if data == 'MNIST':
                if use_test_set:
                    test_set = datasets.MNIST(
                        root="data",
                        train=False,
                        download=True,
                        transform=transform
                    )
                else:

                    t.manual_seed(42)

                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (0.5,))])

                    # MNIST
                    data = datasets.MNIST(
                        root="data",
                        train=True,
                        download=True,
                        transform=transform
                    )
                    _, test_set = random_split(data, [55000, 5000])

            elif data == 'FashionMNIST':
                if use_test_set:
                    test_set = datasets.FashionMNIST(
                        root="data",
                        train=False,
                        download=True,
                        transform=transform
                    )
                else:

                    t.manual_seed(42)

                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (0.5,))])

                    # MNIST
                    data = datasets.FashionMNIST(
                        root="data",
                        train=True,
                        download=True,
                        transform=transform
                    )
                    _, test_set = random_split(data, [55000, 5000])
        else:
            t.manual_seed(42)

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            data = datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)

            test, eval = random_split(data, [8000, 2000])

            test_set = test if use_test_set else eval

        self.loader = DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False)
        self.eval = False
        self.accuracy = None
        self.grad_std = None
        self.ece = None
        self.M.load_state_dict(t.load(param))
        self.M.to(self.device)

    def run_eval(self, tests):

        if t.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        if 'lpd' in tests:
            LPD = []

        if 'accuracy' in tests:
            accuracy_ = []

        if 'calibration_plot' or 'ece' in tests:
            self.pred = []
            self.corr = []

        with t.no_grad():
            for batch_x, batch_y in self.loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = self.M(batch_x)

                if self.M.vi_model != 'non_bayes':
                    logits = [logits]
                    for _ in range(10):
                        logits += [self.M(batch_x)]
                    logits = t.concat(logits, dim=1)
                    pred = F.softmax(logits, dim=-1).mean(dim=1)
                else:
                    pred = F.softmax(logits, dim=-1)

                if 'lpd' in tests:
                    LPD += [t.log(t.tensor(
                        [lp[y] for lp, y in zip(pred, batch_y)]
                    )).mean().cpu()]

                if 'accuracy' in tests:
                    accuracy_ += [(pred.argmax(dim=-1) == batch_y).float().mean().cpu()]

                if 'calibration_plot' or 'ece' in tests:
                    self.pred += [pred.cpu().numpy()]
                    self.corr += [batch_y.cpu().numpy()]

        if 'accuracy' in tests:
            self.accuracy = t.stack(accuracy_).mean().cpu()

        if 'lpd' in tests:
            self.LPD = t.stack(LPD).mean().cpu()

        if 'calibration_plot' or 'ece' in tests:
            self.pred = np.concatenate(self.pred)
            self.corr = np.concatenate(self.corr)

        self.eval = True



    def get_calibration(self, tests):

        if not self.eval:
            self.run_eval(tests)

        if 'calibration_plot':
            self.calibration_x, self.calibration_y = calibration_curve(
                [float(y == e) for y in self.corr for e in range(10)],
                [p for pred in self.pred for p in pred],
                n_bins=10
            )

        if 'ece' in tests:
            ece = mulclasscalerr(10, n_bins=10)
            self.ece = ece(
                t.from_numpy(self.pred),
                t.from_numpy(self.corr),
            )

    def get_grad_var(self):  # implement in NN

        # Objective
        NLL = t.nn.NLLLoss(reduction='none')
        optimizer_normal = t.optim.SGD(self.M.parameters(), momentum=0.9, lr=0.01)

        if t.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        grad_std = []
        for batch_x, batch_y in self.loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            grad_std_per_batch = []
            for i in range(10):

                # Normal optimization
                optimizer_normal.zero_grad()

                y_pred, kl_qp = self.M(batch_x, train=True)
                y_pred = F.log_softmax(y_pred, dim=-1)
                nll_vector = NLL(
                    y_pred.reshape([batch_y.shape[0], 10]),
                    batch_y.repeat_interleave(1)
                ).reshape([batch_y.shape[0], 1]).mean(1)
                nll_normalized = nll_vector.mean(0)
                loss = nll_normalized + kl_qp / len(self.loader) * 128

                loss.backward()

                grad_std_per_batch += [t.concat([param.grad.view(-1) for param in self.M.parameters()]).detach().cpu()]

            grad_std += [t.stack(grad_std_per_batch).std(dim=0).norm()]
        self.grad_std = t.stack(grad_std).mean()


    def run_stats(self, tests):

        test_list = []

        if 'accuracy' in tests:
            if not self.eval:
                self.run_eval(tests)
            test_list += [self.accuracy.numpy()]

        if 'lpd' in tests:
            if not self.eval:
                self.run_eval(tests)
            test_list += [self.LPD.numpy()]

        if 'ece' in tests:
            if not self.eval:
                self.run_eval(tests)
            self.get_calibration(tests)
            test_list += [self.ece.numpy()]

        if 'grad_std' in tests:
            self.get_grad_var()
            test_list += [self.grad_std.cpu().numpy()]

        if 'calibration_plot' in tests:
            if not self.eval:
                self.run_eval(tests)
            if 'ece' not in tests:
                self.get_calibration(tests)
            test_list += [[self.calibration_x, self.calibration_y]]

        if 'corruption' in tests:
            self.corruption()
            test_list += [self.corrupt_nlls.cpu().numpy()]
            test_list += [[self.corrupt_prob_ind.cpu().numpy(), self.corrupt_images.cpu().numpy()]]


        return test_list

    def corruption(self, sigmas=t.logspace(start=-1.5, end=0.5, steps=10), individual_image_nr=42):  # check the paper - bayesian probobalistic approach something ...
        NLL = t.nn.NLLLoss(reduction='mean')
        nll_per_sigma = []
        if individual_image_nr is not None:
            prob_per_sigma_ind = []
            image_per_sigma = []
        if t.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        for sigma in sigmas:
            nll_ = []
            with t.no_grad():
                for i, (batch_x, batch_y) in enumerate(self.loader):
                    batch_x, batch_y = self.corrupt(batch_x, sigma=sigma).to(device), batch_y.to(device)
                    logits = self.M(batch_x)

                    if self.M.vi_model != 'non_bayes':
                        logits = [logits]
                        for _ in range(10):
                            logits += [self.M(batch_x)]
                        logits = t.concat(logits, dim=1)

                        pred = F.log_softmax(logits, dim=-1).mean(dim=1)
                    else:
                        pred = F.log_softmax(logits, dim=-1)

                    nll_ += [NLL(pred, batch_y).cpu()]

                    if individual_image_nr is not None and individual_image_nr // 128 == i:
                        idx = individual_image_nr % 128
                        if self.M.vi_model != 'non_bayes':
                            prob_ind = F.softmax(logits, dim=-1).mean(dim=1).cpu()[idx, batch_y[idx]]
                        else:
                            prob_ind = F.softmax(logits, dim=-1).cpu()[idx, batch_y[idx]]
                        image = batch_x[idx]

            nll_per_sigma += [t.stack(nll_).mean()]

            if individual_image_nr is not None:
                prob_per_sigma_ind += [prob_ind]
                image_per_sigma += [image]

        self.corrupt_nlls = t.stack(nll_per_sigma)


        if individual_image_nr is not None:
            self.corrupt_prob_ind = t.stack(prob_per_sigma_ind)
            self.corrupt_images = t.stack(image_per_sigma)

    def corrupt(self, x, sigma=1):
        t.manual_seed(42)

        emp_sigma = x.std(dim=(-1, -2)).mean()

        return x + t.randn_like(x) * sigma * emp_sigma
