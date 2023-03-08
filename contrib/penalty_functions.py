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


def penalty_c(x, params):
    """
    derivative of the Charbonnier penalty function
    :param x: argument (M x N array)
    :param params: parameter set (dictionary)
    :return: penalty (M x N array)
    """
    eps = params['eps']
    p = 1
    e = (p - 2) / 2
    return p * ((x+eps)**e) / 2


def penalty_cd2(x, params):
    """
    derivative of the CD2 penalty function
    :param x: argument (M x N array)
    :param params: parameter set (dictionary)
    :return: penalty (M x N array)
    """
    b = params['b']

    # lim_{x->0} gp(x) = 1/(b^2)
    out = np.empty_like(x)
    ind = x <= 0
    out[ind] = 1/(b*b)

    # Otherwise
    sqrx = np.sqrt(x[~ind])
    tmp = np.exp(2*sqrx/b)
    out[~ind] = (tmp - 1) / (b*sqrx*(1 + tmp))

    return out


def penalty_ms(x, params):
    """
    derivative of the MS penalty function
    :param x: argument (M x N array)
    :param params: parameter set (dictionary)
    :return: penalty (M x N array)
    """
    b = params['b']
    rho = params['rho']

    # lim_{x->0} gp(x) = 1/(b^2)
    out = np.empty_like(x)
    ind = x <= 0
    out[ind] = -(2-rho)/(2*b*b*(rho - 1))

    # Otherwise
    tmp = np.exp(2*np.sqrt(x[~ind])/b)
    sqrx = np.sqrt(x[~ind])
    out[~ind] = -(tmp*tmp + 2*rho*tmp + 2*tmp + 1) * (tmp - 1) / ((4*rho*tmp-2*tmp-tmp*tmp-1)*b*sqrx*(tmp+1))

    return out
