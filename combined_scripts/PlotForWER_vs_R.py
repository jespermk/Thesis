import numpy as np
import matplotlib as plt

sessions = [ '200samples', '1000samples', '5000samples','30000samples','60000samples']

for s in range(len(sessions)):

    error_weighted_ensemble = np.load('error_weighted_ensemble_%s' % sessions[s])
    error_ensemble = np.load('error_ensemble_%s' % sessions[s])
    error_singular = np.load('error_singular_%s' % sessions[s])


    fig, ax = plt.subplots()

    plt.plot(error_weighted_ensemble, label='weighted ensemble radial %s' %sessions[s])
    plt.plot(error_ensemble, label='ensemble radial %s' %sessions[s])
    plt.plot(error_singular, label='singular radial %s' %sessions[s])
    ax.set_ylabel('predictive Error')
    ax.set_xlabel('iterations')
    ax.legend()
    plt.savefig('error_%s.png' %sessions[s], dpi=300)

    plt.clf()
    plt.cla()