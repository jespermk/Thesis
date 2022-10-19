import matplotlib.pyplot as plt
import numpy as np


def ma(x, w):
    """
    Moving average
    """
    return np.convolve(x, np.ones(w)/w, mode='valid')


def rstd(x, w):
    """
    moving std
    """
    return np.stack(
        [np.std(x[0+i:w+i]) for i in range(int(np.size(x)-w+1))]
    )


def plot(
        data_location,
        names,
        yx_label,
        title,
        save_loc,
        MA=True,
        log=True,
        std_bar=True,
        w=400):

    data = [np.load(data_location[i]) for i in range(len(data_location))]
    fig, ax = plt.subplots()
    for d in range(len(data)):

        if MA:
            ma_data = ma(data[d], w)

            plt.plot(
                ma_data,
                label=names[d],
                linewidth=0.5
            )
            if std_bar:
                std = rstd(data[d], w)
                ax.fill_between(np.arange(np.size(ma_data)),
                                (ma_data - std),
                                (ma_data + std),
                                color='b',
                                alpha=.3)
        else:
            plt.plot(
                data[d],
                label=names[d],
                linewidth=0.5
            )
    ax.set_ylabel(yx_label[0])
    ax.set_xlabel(yx_label[1])
    plt.title(title)
    if log:
        ax.set_yscale('log')
    ax.legend()
    fig.savefig(save_loc, dpi=1000)


S = ['1000samples', '500samples', '250samples', '125samples']
"""
for s in range(len(S)):

    # Error for the various data set sizes
    plot(
        ['generated_data/error_normal_%s.npy' % S[s],
         'generated_data/error_radial_%s.npy' % S[s],
         'generated_data/error_ensemble_%s.npy' % S[s],
         'generated_data/error_weighted_ensemble_%s.npy' % S[s],
         'generated_data/error_t_radial_%s.npy' % S[s],
         'generated_data/error_exp_radial_%s.npy' % S[s]]
        ,
        ['Normal',
         'Radial',
         'Ensemble',
         'Weighted ensemble',
         'T radial',
         'Exp radial']
        ,
        ['Mean Error',
         'Iterations']
        ,
        'Error with %s.npy' % S[s]
        ,
        'plots/error_%s_new' % S[s]
    )

    # Loss for the various data set sizes
    plot(
        ['generated_data/loss_radial_%s.npy' % S[s],
         'generated_data/loss_radial_%s.npy' % S[s],
         'generated_data/mean_loss_ensemble_%s.npy' % S[s],
         'generated_data/loss_t_radial_%s.npy' % S[s],
         'generated_data/loss_exp_radial_%s.npy' % S[s]]
        ,
        ['Normal',
         'radial',
         'Ensemble',
         'T radial',
         'Exp radial']
        ,
        ['Loss',
         'Iterations']
        ,
        'log loss with %s.npy' % S[s]
        ,
        'plots/loss_new%s' % S[s]
    )


# Mu gradients
plot(
    ['generated_data/norm_mu_grad_radial.npy',
     'generated_data/norm_mu_grad_normal.npy',
     'generated_data/norm_mu_grad_t_radial.npy',
     'generated_data/norm_mu_grad_exp_radial.npy']
    ,
    ['Radial',
     'Normal',
     'T radial',
     'Exp radial']
    ,
    ['Norm of the mu gradient',
     'Iterations']
    ,
    'Norm of mu gradients'
    ,
    'plots/norm_mu_grad'
)

# Sigma gradients
plot(
    ['generated_data/norm_sigma_grad_radial.npy',
     'generated_data/norm_sigma_grad_normal.npy',
     'generated_data/norm_sigma_grad_t_radial.npy',
     'generated_data/norm_sigma_grad_exp_radial.npy']
    ,
    ['Radial',
     'Normal',
     'T radial',
     'Exp radial']
    ,
    ['Norm of sigma gradient',
     'Iterations']
    ,
    'Norm of sigma gradients'
    ,
    'plots/norm_sigma_grad'
)

# Sigma gradients radials
plot(
    ['generated_data/norm_sigma_grad_radial.npy',
     'generated_data/norm_sigma_grad_t_radial.npy',
     'generated_data/norm_sigma_grad_exp_radial.npy']
    ,
    ['Radial',
     'T radial',
     'Exp radial']
    ,
    ['Norm of sigma gradient',
     'Iterations']
    ,
    'Norm of sigma gradients'
    ,
    'plots/norm_sigma_grad_radials'
    ,
    log=False
)

# Mu
plot(
    ['generated_data/norm_mu_radial.npy',
     'generated_data/norm_mu_normal.npy',
     'generated_data/norm_mu_t_radial.npy',
     'generated_data/norm_mu_exp_radial.npy']
    ,
    ['Radial',
     'Normal',
     'T radial',
     'Exp radial']
    ,
    ['Norm of mu',
     'Iterations']
    ,
    'Norm of mu'
    ,
    'plots/norm_mu'
)

# Sigma
plot(
    ['generated_data/norm_sigma_radial.npy',
     'generated_data/norm_sigma_normal.npy',
     'generated_data/norm_sigma_t_radial.npy',
     'generated_data/norm_sigma_exp_radial.npy']
    ,
    ['Radial',
     'Normal',
     'T radial',
     'Exp radial']
    ,
    ['Norm of sigma',
     'Iterations']
    ,
    'Norm of sigma'
    ,
    'plots/norm_sigma'
    ,
    log=False
    ,
    MA=False
)
# Norm of standard deviation of gradients
plot(
    ['generated_data/grads_radial_std.npy',
     'generated_data/grads_normal_std.npy',
     'generated_data/grads_t_radial_std.npy',
     'generated_data/grads_exp_radial_std.npy']
    ,
    ['Radial',
     'Normal',
     'T radial',
     'Exp radial']
    ,
    ['Norm of sigma',
     'Iterations']
    ,
    'Norm of std of gradients'
    ,
    'plots/norm_grad_std'
)

# Norm of standard deviation of gradients
plot(
    ['generated_data/grads_radial_std.npy',
     'generated_data/grads_t_radial_std.npy',
     'generated_data/grads_exp_radial_std.npy']
    ,
    ['Radial',
     'T radial',
     'Exp radial']
    ,
    ['Norm of sigma',
     'Iterations']
    ,
    'Norm of std of gradients'
    ,
    'plots/norm_grad_std_radials'
)

# all the data for radial
plot(
    ['generated_data/norm_mu_grad_radial.npy',
     'generated_data/norm_sigma_grad_radial.npy',
     'generated_data/norm_mu_radial.npy',
     'generated_data/norm_sigma_radial.npy',
     'generated_data/grads_radial_std.npy',
     'generated_data/loss_radial.npy',
     'generated_data/error_radial.npy']
    ,
    ['Mu gradiant norm',
     'Sigma gradient norm',
     'Mu norm',
     'Sigma norm',
     'Norm of std of gradients',
     'Loss',
     'Error']
    ,
    ['Norm',
     'Iterations']
    ,
    'Radial'
    ,
    'plots/radial'
)

# all the data for normal
plot(
    ['generated_data/norm_mu_grad_normal.npy',
     'generated_data/norm_sigma_grad_normal.npy',
     'generated_data/norm_mu_normal.npy',
     'generated_data/norm_sigma_normal.npy',
     'generated_data/grads_normal_std.npy',
     'generated_data/loss_normal.npy',
     'generated_data/error_normal.npy']
    ,
    ['Mu gradiant norm',
     'Sigma gradient norm',
     'Mu norm',
     'Sigma norm',
     'Norm of std of gradients',
     'Loss',
     'Error']
    ,
    ['Norm',
     'Iterations']
    ,
    'Normal'
    ,
    'plots/normal'
)

# all the data for t radial
plot(
    ['generated_data/norm_mu_grad_t_radial.npy',
     'generated_data/norm_sigma_grad_t_radial.npy',
     'generated_data/norm_mu_t_radial.npy',
     'generated_data/norm_sigma_t_radial.npy',
     'generated_data/grads_t_radial_std.npy',
     'generated_data/loss_t_radial.npy',
     'generated_data/error_t_radial.npy']
    ,
    ['Mu gradiant norm',
     'Sigma gradient norm',
     'Mu norm',
     'Sigma norm',
     'Norm of std of gradients',
     'Loss',
     'Error']
    ,
    ['Norm',
     'Iterations']
    ,
    'T radial'
    ,
    'plots/t_radial'
)

# all the data for exp radial
plot(
    ['generated_data/norm_mu_grad_exp_radial.npy',
     'generated_data/norm_sigma_grad_exp_radial.npy',
     'generated_data/norm_mu_exp_radial.npy',
     'generated_data/norm_sigma_exp_radial.npy',
     'generated_data/grads_exp_radial_std.npy',
     'generated_data/loss_exp_radial.npy',
     'generated_data/error_exp_radial.npy']
    ,
    ['Mu gradiant norm',
     'Sigma gradient norm',
     'Mu norm',
     'Sigma norm',
     'Norm of std of gradients',
     'Loss',
     'Error']
    ,
    ['Norm',
     'Iterations']
    ,
    'Radial'
    ,
    'plots/exp_radial'
)

# Error and loss
plot(
    ['generated_data/error_radial.npy',
     'generated_data/error_normal.npy',
     'generated_data/error_t_radial.npy',
     'generated_data/error_exp_radial.npy',
     'generated_data/loss_radial.npy',
     'generated_data/loss_normal.npy',
     'generated_data/loss_t_radial.npy',
     'generated_data/loss_exp_radial.npy']
    ,
    ['Radial error',
     'Normal error',
     'T radial error',
     'Exp radial error',
     'Radial loss',
     'Normal loss',
     'T radial loss',
     'Exp radial loss']
    ,
    ['Norm of sigma',
     'Iterations']
    ,
    'Norm of std of gradients'
    ,
    'plots/error_loss'
)

# Error
plot(
    ['generated_data/error_normal.npy',
     'generated_data/error_radial.npy',
     'generated_data/error_t_radial.npy',
     'generated_data/error_exp_radial.npy']
    ,
    ['Normal error',
     'Radial error',
     'T radial error',
     'Exp radial error']
    ,
    ['Norm of sigma',
     'Iterations']
    ,
    'Error of radials'
    ,
    'plots/error_new'
)


# Norm of standard deviation of gradients - radials
plot(
    ['generated_data/grads_radial_std.npy',
     'generated_data/grads_t_radial_std.npy',
     'generated_data/grads_exp_radial_std.npy']
    ,
    ['Radial',
     'T radial',
     'Exp radial']
    ,
    ['Norm of sigma',
     'Iterations']
    ,
    'Norm of std of gradients'
    ,
    'plots/norm_grad_std_radials'
)

# Sigma - radial
plot(
    ['generated_data/norm_sigma_radial.npy',
     'generated_data/norm_sigma_t_radial.npy',
     'generated_data/norm_sigma_exp_radial.npy']
    ,
    ['Radial',
     'T radial',
     'Exp radial']
    ,
    ['Norm of sigma',
     'Iterations']
    ,
    'Norm of sigma'
    ,
    'plots/norm_sigma_radials'

)

S = ['Mnist', 'FashionMnist']

stats = np.load('generated_data/model_stat.npy')

correct_rate, nlll, ece = stats[0], stats[1], stats[2]


begin = 0
end = 0
for s in range(len(S)):
    end += 4
    fig, axs = plt.subplots(1, 3, figsize=(14,4))

    axs[0].bar(['N', 'R', 'TR', 'ExpR'], 1 - correct_rate[begin:end])
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Failure rate')
    axs[0].set_xlabel('Models')
    axs[1].bar(['N', 'R', 'TR', 'ExpR'], nlll[begin:end])
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Nlll')
    axs[1].set_xlabel('Models')
    axs[2].bar(['N', 'R', 'TR', 'ExpR'], ece[begin:end])
    axs[2].set_yscale('log')
    axs[2].set_ylabel('ECE')
    axs[2].set_xlabel('Models')
    fig.suptitle('Model stats for %s radials' %S[s])

    fig.tight_layout()
    fig.savefig('plots/new_model_stat_%s' %S[s], dpi=400)
    begin += 4
"""
N = np.load('generated_data/normal_vcl.npy')
R = np.load('generated_data/radial_vcl.npy')
tR = np.load('generated_data/t_radial_vcl.npy')
expR = np.load('generated_data/exp_radial_vcl.npy')


names = ['N', 'R', 'expR', 'tR']
x_axis_names = ('T1', 'T2', 'T3', 'T4', 'T5')
X = np.arange(5)

data = [N[0],
        R[0],
        expR[0],
        tR[0]]
fig, ax = plt.subplots()
ax.bar(X + 0.00, data[0], width=0.2)
ax.bar(X + 0.20, data[1], width=0.2)
ax.bar(X + 0.40, data[2], width=0.2)
ax.bar(X + 0.60, data[3], width=0.2)
ax.legend(labels=names)
ax.set_ylabel('Failure rate')
ax.set_xlabel('Tasks')
ax.plot(np.arange(0,4.75,0.01),np.ones_like(np.arange(0,4.75,0.01)), ls='--', c='black')
ax.plot(np.arange(0,4.75,0.01),np.ones_like(np.arange(0,4.75,0.01))*.5, ls='--', c='grey')
fig.savefig('plots/cvl_stat_correct', dpi=400)
plt.show()


data = [N[1],
        R[1],
        expR[1],
        tR[1]]
fig, ax = plt.subplots()
ax.bar(X + 0.00, data[0], width = 0.2)
ax.bar(X + 0.20, data[1], width = 0.2)
ax.bar(X + 0.40, data[2], width = 0.2)
ax.bar(X + 0.60, data[3], width = 0.2)
ax.legend(labels=names)
ax.set_ylabel('Nll')
ax.set_xlabel('Tasks')
fig.savefig('plots/cvl_stat_nll', dpi=400)


