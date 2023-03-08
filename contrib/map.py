#  Copyright [2021] [Jan Dorazil]
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
from scipy.ndimage import map_coordinates, convolve1d, gaussian_filter1d
from skimage.transform import resize
import sor.Sor as Sor

from penalty_functions import penalty_c


def reduce(x):
    return np.sum(x, axis=2)


def warp_image(G, w):
    M, N = G.shape[0:2]

    # Create a warped grid
    Y, X = np.mgrid[0:M, 0:N]
    X = X + w[:, :, 0]
    Y = Y + w[:, :, 1]

    # Warp
    G_w = map_coordinates(G, [Y, X], order=1, mode='constant', cval=np.nan)

    return G_w


def image_derivatives(F, G, w_bar, h, params):
    # Warp the second image
    Gw = warp_image(G, w_bar)

    # Identify out-of-border pixels and remove NaNs
    dis = np.isnan(Gw)
    Gw[dis] = 0

    # Temporal average of F and G for spatial derivatives
    FGw = 0.5 * F + 0.5 * Gw

    # Presmooth only for derivatives
    if params['sigma'] > 1e-6:
        FGw = gaussian_filter1d(FGw, params['sigma'], axis=0, mode='reflect')
        FGw = gaussian_filter1d(FGw, params['sigma'], axis=1, mode='reflect')

    # Calculate derivatives
    Gx = convolve1d(FGw, h, axis=1, mode='nearest')
    Gy = convolve1d(FGw, h, axis=0, mode='nearest')
    D = Gw - F

    # Normalize gradients
    theta = 1.0
    if params['gradnorm']:
        theta = Gx * Gx + Gy * Gy + 0.01

    # Set derivatives of out-of-border pixels to zero
    D[dis] = 0
    Gx[dis] = 0
    Gy[dis] = 0

    D2 = D*D / theta
    DGx = D * Gx / theta
    DGy = D * Gy / theta
    Gx2 = Gx * Gx / theta
    GxGy = Gy * Gx / theta
    Gy2 = Gy * Gy / theta

    return D2, DGx, DGy, Gx2, GxGy, Gy2


def setup_sor(w0, DGx, DGy, Gx2, GxGy, Gy2, Lamb, Lama_horiz, Lama_vert, params):
    # Contribution of color constancy term
    alpha = params['alpha']
    A11 = alpha * Lamb * Gx2
    A12 = alpha * Lamb * GxGy
    A22 = alpha * Lamb * Gy2
    b1 = -alpha * Lamb * DGx
    b2 = -alpha * Lamb * DGy

    # Contribution of smoothness term
    # horizontal
    beta = params['beta']
    horiz = np.empty_like(Lama_horiz)
    horiz[:, 0:-1] = beta * Lama_horiz[:, 0:-1]
    horiz[:, -1] = 0

    # Right-hand-side contribution
    u0 = w0[:, :, 0]
    v0 = w0[:, :, 1]
    u0x = np.empty_like(u0)
    u0x[:, 0:-1] = u0[:, 1:] - u0[:, 0:-1]
    u0x[:, -1] = 0
    v0x = np.empty_like(v0)
    v0x[:, 0:-1] = v0[:, 1:] - v0[:, 0:-1]
    v0x[:, -1] = 0

    tmp = horiz * u0x
    b1 += tmp
    b1[:, 1:] -= tmp[:, 0:-1]
    tmp = horiz * v0x
    b2 += tmp
    b2[:, 1:] -= tmp[:, 0:-1]

    # vertical
    vert = np.empty_like(Lama_vert)
    vert[0:-1, :] = beta * Lama_vert[0:-1, :]
    vert[-1, :] = 0

    # Right-hand-side contribution
    u0y = np.empty_like(u0)
    u0y[0:-1, :] = u0[1:, :] - u0[0:-1, :]
    u0y[-1, :] = 0
    v0y = np.empty_like(v0)
    v0y[0:-1, :] = v0[1:, :] - v0[0:-1, :]
    v0y[-1, :] = 0

    tmp = vert * u0y
    b1 += tmp
    b1[1:, :] -= tmp[0:-1, :]
    tmp = vert * v0y
    b2 += tmp
    b2[1:, :] -= tmp[0:-1, :]

    return A11, A12, A22, b1, b2, horiz, vert


def e_step(dw, w0, DGx, DGy, Gx2, GxGy, Gy2, Lamb, Lama_horiz, Lama_vert, params):
    M, N = DGx.shape[0:2]

    # Setup the matrices for SOR
    A11, A12, A22, b1, b2, horiz, vert = setup_sor(w0, DGx, DGy, Gx2, GxGy, Gy2, Lamb, Lama_horiz, Lama_vert, params)

    # Run SOR
    du, dv = Sor.solve(np.float32(dw[:, :, 0]), np.float32(dw[:, :, 1]), np.float32(A11), np.float32(A12),
                     np.float32(A22), np.float32(b1), np.float32(b2), np.float32(horiz), np.float32(vert),
                     params['niter_solver'], params['omega_sor'])
    dw = np.dstack((du, dv))

    return dw


def m_step(dw, w0, D2, DGx, DGy, Gx2, GxGy, Gy2, params):
    du = dw[:, :, 0]
    dv = dw[:, :, 1]
    du2 = du * du
    dv2 = dv * dv
    duv = du * dv

    # Color constancy weights
    tmp = D2 + 2 * DGx * du + 2 * DGy * dv + Gx2 * du2 + 2 * GxGy*duv + Gy2 * dv2
    phi = params['phi']
    Lamb = phi(tmp, params)

    # Contribution of the posterior mean
    # horizontal
    u = w0[:, :, 0] + dw[:, :, 0]
    v = w0[:, :, 1] + dw[:, :, 1]
    ux = np.empty_like(u)
    ux[:, 0:-1] = u[:, 1:] - u[:, 0:-1]
    ux[:, -1] = 0
    uy = np.empty_like(u)
    uy[0:-1, :] = u[1:, :] - u[0:-1, :]
    uy[-1, :] = 0
    # vertical
    vx = np.empty_like(v)
    vx[:, 0:-1] = v[:, 1:] - v[:, 0:-1]
    vx[:, -1] = 0
    vy = np.empty_like(v)
    vy[0:-1, :] = v[1:, :] - v[0:-1, :]
    vy[-1, :] = 0

    # horizontal
    tmp = ux*ux + vx*vx
    psi = params['psi']
    Lama_horiz = psi(tmp, params)

    # vertical
    tmp = uy*uy + vy*vy
    Lama_vert = psi(tmp, params)

    return Lamb, Lama_horiz, Lama_vert


def get_sigma(w0, DGx, DGy, Gx2, GxGy, Gy2, Lamb, Lama_horiz, Lama_vert, params):
    M, N = DGx.shape[0:2]

    # Setup the matrices for SOR
    A11, A12, A22, b1, b2, horiz, vert = setup_sor(w0, DGx, DGy, Gx2, GxGy, Gy2, Lamb, Lama_horiz, Lama_vert, params)

    # Calculate posterior variance
    tmp = horiz + vert
    tmp[:, 1:] += horiz[:, 0:-1]
    tmp[1:, :] += vert[0:-1, :]
    J11 = A11 + tmp
    J22 = A22 + tmp
    s = np.empty((M, N, 2))
    s[:, :, 0] = 1 / J11
    s[:, :, 1] = 1 / J22

    return s


def default_params(params):
    default = {
        'phi': penalty_c,
        'alpha': 1.0,
        'psi': penalty_c,
        'beta': 1.0,
        'eps': 1e-5,
        'sigma': 0.0,
        'nwarp': 5,
        'niter': 5,
        'solver': 'sor',
        'niter_solver': 30,
        'omega_sor': 1.9,
        'eta': 0.001,
        'gradnorm': False
    }

    if params is None:
        params = dict()

    for key in default.keys():
        if key not in params.keys():
            params[key] = default[key]

    return params


def run(F, G, w0, params=None):
    assert(F.dtype == np.float32 and G.dtype == np.float32)
    M, N = F.shape

    # Setup default params
    params = default_params(params)

    # Data term convolution kernel
    h = np.array([-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12], dtype=np.float32)

    # Calculate min & max scale
    min_size = np.min(np.array(F.shape))
    eta = params['eta']
    max_scale = max(0, int(np.log(15/min_size) / np.log(eta)))

    # Iterate coarse to fine
    w = np.nan
    dw = np.nan
    for scale in range(max_scale, -1, -1):
        coarse_shape = np.ceil(np.array(F.shape) * (eta ** (scale+1))).astype(int)
        shape = np.ceil(np.array(F.shape) * (eta ** scale)).astype(int)
        if all(shape == coarse_shape):
            continue

        # Downsample images
        F_coarse = resize(F, shape, order=1)
        G_coarse = resize(G, shape, order=1)

        # Downsample initial flow
        if scale == max_scale:
            down = shape / F.shape
            w = resize(w0, (shape[0], shape[1], 2), order=1)
            w[:, :, 0] *= down[0]
            w[:, :, 1] *= down[1]

        # Upsample coarse flow
        if scale < max_scale:
            up = shape / coarse_shape
            w = resize(w, (shape[0], shape[1], 2), order=1)
            w[:, :, 0] *= up[0]
            w[:, :, 1] *= up[1]

        # Run niter iterations
        for l in range(params['nwarp']):
            D2, DGx, DGy, Gx2, GxGy, Gy2 = image_derivatives(F_coarse, G_coarse, w, h, params)

            dw = np.zeros((shape[0], shape[1], 2))
            for k in range(params['niter']):
                #print('Scale', scale, 'iteration', l+1, 'subiteration', k+1)
                Lamb, Lama_horiz, Lama_vert = m_step(dw, w, D2, DGx, DGy, Gx2, GxGy, Gy2, params)
                dw = e_step(dw, w, DGx, DGy, Gx2, GxGy, Gy2, Lamb, Lama_horiz, Lama_vert, params)

            w = w + dw

    s = get_sigma(w, DGx, DGy, Gx2, GxGy, Gy2, Lamb, Lama_horiz, Lama_vert, params)
    return w, s