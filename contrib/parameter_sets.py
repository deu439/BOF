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

from contrib.penalty_functions import penalty_c, penalty_cd2, penalty_ms

# All parameters were obtained by Bayesian optimization on the ladprox sequence with known ground-truth. Parameters of
# the MAP methods are tuned for minimum endpoint-error and parameters of the VB methods for maximum CL. Note that the
# endpoint-error can only be used to identify the optimal ratio alpha/beta (in case of the MAP methods). That is why
# alpha=1 for this scenario. Gradient normalization 'gradnorm' is commonly used in classical OF estimation methods
# to improve convergence. Unfortunately it cannot be used with domain-specific likelihood functions, since it affects
# the image statistics.

# MAP (C) short-axis
map_c_shortaxis = {'phi': penalty_c, 'alpha': 1, 'psi': penalty_c, 'beta': 1.3509, 'gradnorm': True}
# MAP (C) long-axis
map_c_longaxis = {'phi': penalty_c, 'alpha': 1, 'psi': penalty_c, 'beta': 1.2352, 'gradnorm': True}
# MAP (CD2) short-axis
map_cd2_shortaxis = {'phi': penalty_cd2, 'alpha': 1, 'psi': penalty_c, 'beta': 0.2883, 'gradnorm': False}
# MAP (CD2) long-axis
map_cd2_longaxis = {'phi': penalty_cd2, 'alpha': 1, 'psi': penalty_c, 'beta': 0.2240, 'gradnorm': False}
# MAP (MS) short-axis
map_ms_shortaxis = {'phi': penalty_ms, 'alpha': 1, 'psi': penalty_c, 'beta': 2.0800, 'gradnorm': False}
# MAP (MS) long-axis
map_ms_longaxis = {'phi': penalty_ms, 'alpha': 1, 'psi': penalty_c, 'beta': 1.8455, 'gradnorm': False}


# VB (C) short-axis
vb_c_shortaxis = {'phi': penalty_c, 'alpha': 0.1137, 'psi': penalty_c, 'beta': 3.2917, 'gradnorm': True}
# VB (C) long-axis
vb_c_longaxis = {'phi': penalty_c, 'alpha': 0.4229, 'psi': penalty_c, 'beta': 4.4985, 'gradnorm': True}
# VB (CD2) short-axis
vb_cd2_shortaxis = {'phi': penalty_cd2, 'alpha': 0.5513, 'psi': penalty_c, 'beta': 3.1023, 'gradnorm': False}
# VB (CD2) long-axis
vb_cd2_longaxis = {'phi': penalty_cd2, 'alpha': 0.6113, 'psi': penalty_c, 'beta': 3.3662, 'gradnorm': False}
# VB (MS) short-axis
vb_ms_shortaxis = {'phi': penalty_ms, 'alpha': 0.0923, 'psi': penalty_c, 'beta': 3.2876, 'gradnorm': False}
# VB (MS) long-axis
vb_ms_longaxis = {'phi': penalty_ms, 'alpha': 0.1757, 'psi': penalty_c, 'beta': 3.4936, 'gradnorm': False}
