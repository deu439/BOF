#   Copyright [2021] [Jan Dorazil]
#  #
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import multiprocessing
from functools import partial
from scipy.io import loadmat

from contrib.aux import sp_plot
from contrib import vb, parameter_sets

# Some user defined params ====
bmode_path = "/home/deu/STRAUS/%s/%s/frame%d.mat"
flow_path = "/home/deu/STRAUS/%s/%s/flow%d.mat"
n_frames = 34
scan = 'long_axis'
n_processes = 4
sp_samples = 200
method = vb
params = parameter_sets.vb_c_longaxis
# End of user defined params ====


def evaluate(seq, scan):
    # Start the processing in parallel
    objective_partial = partial(objectives, seq=seq, scan=scan)
    with multiprocessing.Pool(processes=n_processes) as pool:
        score = pool.map(objective_partial, np.arange(0, n_frames-1))

    score = np.array(score)

    # Return the score array
    return np.mean(score, axis=0)


def objectives(j, seq, scan):
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
    try:
        w1, s1 = method.run(np.float32(F), np.float32(G), w0, params)
    except Exception:
        return np.inf

    # Calculate end-point error
    dw = w1 - w
    ee = np.sqrt(dw[:, :, 0]**2 + dw[:, :, 1]**2)
    aee = np.mean(ee, where=~mask)

    # Calculate sparsification plot
    en = np.log(s1[~mask, 0]) + np.log(s1[~mask, 1])
    sp = sp_plot(ee[~mask], en, sp_samples)
    auc = np.sum(sp[~np.isnan(sp)]) / (sp[0] * sp_samples)

    # Calculate -2 log p(w|F, G, theta)
    dw = w1 - w
    ll = np.sum(np.log(s1) + dw*dw/s1, axis=2)
    ll = np.sum(ll, where=~mask)

    return aee, auc, ll


for seq in ['normal', 'ladprox', 'laddist', 'lcx', 'rca']:
    scores = evaluate(seq, scan)
    print('%s) AEE: %.3f, AUC: %.3f, LL: %e' % (seq, scores[0], scores[1], scores[2]))
