#  Copyright [2023] [Jan Dorazil]
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
from scipy.interpolate import interp1d


# Generate a sparsification plot
def sp_plot(error, entropy, sp_samples=200):
    n = error.size     # Number of pixels with known ground-truth
    m = 512

    # Convert to uint8
    entropy = entropy - np.min(entropy)
    entropy = entropy / np.max(entropy)
    entropy = np.uint16(entropy * (m-1))

    # Calculate the sparsification plots on a natural grid
    num = np.empty(m+1, dtype=int)
    aepe = np.empty(m+1, dtype=float)
    for k in range(m+1):
        th = m - 1 - k
        ind = (entropy <= th)
        num[k] = np.sum(1-ind)
        if num[k] < n:
            aepe[k] = np.mean(error[ind])
        else:
            aepe[k] = np.nan

    # Now resample on a uniform grid
    num, ind = np.unique(num, return_index=True)
    f = interp1d(num, aepe[ind], kind='linear')
    return f(np.linspace(0, n, sp_samples))