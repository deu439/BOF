# BOF
This repository contains source code supplementing the ICASSP 2023 paper "Bayesian Methods For Optical Flow Estimation Using a Variational Approximation, With Application to Ultrasound"
![Figure_1](https://user-images.githubusercontent.com/6999875/223785652-68433f42-aa7f-4e68-b13c-f6cd596bf684.png)
## Structure
- `contrib`directory contains python implementations of the VB and MAP methods.
- `sor`directory contains an extension module written in c which implements the [Successive over-relaxation](https://en.wikipedia.org/wiki/Successive_over-relaxation) solver. The c code was mostly taken over from [epicflow](https://github.com/suhangpro/epicflow/tree/master).
- `STRAUS.zip` file contains 2D data extracted from the [STRAUS dataset](https://team.inria.fr/epione/en/data/straus/) in the MATLAB's .mat format. This includes the B-mode images (with values from -65 to 0 dB) and the ground-truth optical flow (displacement vectors in pixels).
- `demo.py`runs the VB method and visualizes results.
- `evaluation.py`should replicate the results presented in the paper.

## Instructions
```bash
# Install the required packages
REPO=$(pwd)		# The root directory of the cloned repository
pip install -r requirements.txt

# Build the c extension
mkdir $REPO/sor/build
cd $REPO/sor/build
cmake ..
make install	# This will build and install the module into $REPO/sor

# Run the demo/evaluation
cd $REPO
unzip STRAUS.zip
python demo.py
python evaluation.py
```