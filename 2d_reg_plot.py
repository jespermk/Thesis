import matplotlib.pyplot as plt
import numpy as np

n_tasks = 3

mean = np.load('generated_data/2d_reg_R_pred_mean_%i.npy' % n_tasks)
std = np.load('generated_data/2d_reg_R_pred_std_%i.npy' % n_tasks)
interval = np.load('generated_data/2d_reg_R_pred_interval_%i.npy' % n_tasks)

gen_data_x = np.load('generated_data/2d_reg_R_gen_data_x_%i.npy' % n_tasks)
gen_data_y = np.load('generated_data/2d_reg_R_gen_data_y_%i.npy' % n_tasks)
names = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']



fig, ax = plt.subplots()
[ax.plot(interval[t], mean[t], label=names[t]) for t in range(n_tasks+1)]
[ax.scatter(gen_data_x[t], gen_data_y[t], color='y', alpha=0.02) for t in range(n_tasks+1)]
[ax.fill_between(np.ndarray.flatten(interval[t]),
                            (mean[t] - std[t]),
                            (mean[t] + std[t]),
                            color='b',
                            alpha=.1) for t in range(n_tasks+1)]

ax.legend()
fig.savefig('plots/2d_reg_vcl_5_task', dpi=400)
plt.show()
