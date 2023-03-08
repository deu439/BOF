import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from flow_vis import flow_to_color
import timeit

from contrib.aux import sp_plot
from contrib import vb, parameter_sets

# User parameters ====
bmode_path = "/home/deu/STRAUS/%s/%s/frame%d.mat"
flow_path = "/home/deu/STRAUS/%s/%s/flow%d.mat"
est_path = "/home/deu/STRAUS/%s/%s/est%d.mat"
n_frames = 34
seq = 'ladprox'
scan = 'long_axis'
method = vb
params = parameter_sets.vb_c_longaxis
sp_samples = 200
# End of user parameters ====

overall_aee = 0
overall_auc = 0
overall_auec = 0

fig, ax = plt.subplots(3, 2)
plt.ion()

for j in range(n_frames-1):
    # Load envelopes ====
    F = loadmat(bmode_path % (seq, scan, j), squeeze_me=True)['slc']
    G = loadmat(bmode_path % (seq, scan, j + 1), squeeze_me=True)['slc']
    M, N = F.shape

    # Load ground truth, create a mask
    gt = loadmat(flow_path % (seq, scan, j), squeeze_me=True)
    w = np.empty((M, N, 2))
    w[:, :, 0] = gt['u']
    w[:, :, 1] = gt['v']
    mask = np.isnan(w[:, :, 0])
    w[mask, :] = 0

    w0 = np.zeros((M, N, 2), dtype=np.float32)
    start = timeit.default_timer()
    w1, s1 = method.run(np.float32(F), np.float32(G), w0, params)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    w1[mask, :] = 0
    s1[mask, :] = 0

    # Store the estimates
    savemat(est_path % (seq, scan, j), {'u': w1[:, :, 0], 'v': w1[:, :, 1], 'su': s1[:, :, 0], 'sv': s1[:, :, 1]})

    # Calculate end-point error
    dw = w1 - w
    ee = np.sqrt(dw[:, :, 0]**2 + dw[:, :, 1]**2)
    aee = np.mean(ee, where=~mask)
    print('Frame %i, EE: %.4f' % (j, aee))
    overall_aee += aee

    # Plot the result
    log_ee = 20 * np.log10(ee / 1.0 + 1e-9)
    log_ee[log_ee < -90] = 0
    masked_ee = np.ma.array(ee, mask=mask)
    masked_s1 = np.ma.array(np.sum(np.log(s1 + 1e-9), axis=2), mask=mask)

    # Calculate sparsification plot
    en = np.log(s1[~mask, 0]) + np.log(s1[~mask, 1])
    est = sp_plot(ee[~mask], en, sp_samples)
    orc = sp_plot(ee[~mask], ee[~mask], sp_samples)
    erc = (est - orc) / est[0]
    auc = np.sum(est[~np.isnan(est)]) / (est[0] * sp_samples)
    overall_auc += auc
    auec = np.sum(erc[~np.isnan(erc)]) / sp_samples
    overall_auec += auec
    print('Frame %i, EE: %.3f, AUC: %.3f, AUEC: %.3f' % (j, aee, auc, auec))

    # Figure 1
    ax[0,0].imshow(F, cmap='gray')
    x = np.linspace(0, 1, sp_samples)
    ax[0,1].clear()
    ax[0,1].plot(x, est / est[0])
    ax[0,1].plot(x, orc / orc[0])
    ax[0,1].legend(['Estimate', 'Oracle'])
    ax[1,0].imshow(flow_to_color(w / np.max(w)))
    ax[1,1].imshow(flow_to_color(w1 / np.max(w)))
    ax[2,0].clear()
    ax[2,0].imshow(masked_ee)
    ax[2,1].clear()
    ax[2,1].imshow(masked_s1)
    plt.show(block=False)
    plt.pause(1.0)


print('Overall AEE: ', overall_aee / (n_frames-1))
print('Overall AUC: ', overall_auc / (n_frames-1))
print('Overall AUEC: ', overall_auec / (n_frames-1))
