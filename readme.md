<img alt="PyPI" src="https://img.shields.io/pypi/v/matchernet"> <img alt="APM" src="https://img.shields.io/apm/l/matchernet">


<img alt="MN logo" src="MatcherNetLogo.png" width="200" height="200" >

# What is MatcherNet?

## What is it?
MatcherNet is a probabilistic state-space model for dynamic system identification and control. With MatcherNet, you can easily design a dynamic world model of high-dimensional / multi-modal / multi-scale states and observations for robotics, image processing, sensor networks, and their hybrid cases. MatcherNet may provide you a solution better than a large state space model, a deep neural network of end-to-end structure, etc.

<img alt="MatcherNet" src="MatcherNet.png" height="300"> 
                                                                                                  
## MatcherNet as a state space model
<img alt="MatcherNet" src="ssm.png" height="200">
MatcherNet is a state space model that includes Extended Kalman-filter (EKF), a non-linear extension of Kalman-filter, as a special case. The EKF works with a pair of given observation model p( y(t) | x(t) ) and dynamics model p( x(t+Delta t) | x(t) ), and it calculates the posterior of the current state q( x(t) ) = p( x(t) | y(1),...,y(t) ) in a sequential manner. In MatcherNet, the algorithm is written in a message passing among the bundles (the observation and the state space model) and the matcher (the prediction model), see the figure above.

<img alt="MatcherNet" src="decomposition.png" height="200">
For MatcherNet, the user may divide the observed variable  y  and the state variable  x  into multiple parts,  like  y = (y^(1), y^(2), y^(3) ),  x = ( x^(1), x^(2), x^(3), x^(4) ), respectively. Then, the state space is approximated with the hierarchically decomposed model, in which all parts of dynamics/observation model are low-dimensional, easy to learn, and re-usable models.

## MatcherNet as a model predictive controller
<img alt="MatcherNet" src="mn_controllers2.png" height="200">

MatcherNet can emit control signal u in an online manner as model predictive control (MPC). A typical network structure (left pannel in the Fig. above) can implement well known controllers, such as PID, iLQR (iLQG), whereas a simpler structure (right pannel) can implement so called "active inference". In any cases, you can provide a control goal as a prior probability of the state variable, and the controller calculates the control signal that minimizes the current and future distance to the prior. 

## MatcherNet and multi-thread computing
<img alt="MatcherNet" src="mn_parallel.png" height="300">

MatcherNet efficiently works with multi-thread computing. Modular division of state-space model lower the dimensionality of each state variable, and multiple modules run in parallel in a multi-core computing environment. 

## Acknowledgment
MatcherNet was developped with support by the New Energy and Industrial Technology Development Organization (NEDO), Japan,
and by Post-K application development for exploratory challenges from the MEXT, Japan.

# Document
See the [matchernet documentation](https://shigeyukioba.github.io/matchernet/)

# How to install?
The alpha version has been tested and released below:
https://test.pypi.org/project/matchernet-py-001/

You can install the alpha version with the following command:
```bash
pip install -i https://test.pypi.org/simple/ matchernet-py-001
```

# How to try demos?
See demo files under the `/demos` directory.
```bash
Python3 demos/demo_ekf.py
```

See also jupyter notebooks under the `/examples` directory.

# How to set up in detail?
## For MacOSX
### Install Python 3.7.*
It is tested on 3.7.2 and 3.7.3.
3.8.* is not supported yet.
### Set up virtual environment
```bash
mkdir ~/virt_env
cd ~/virt_env
virtualenv -p Python3 mn
workon mn
```
### Install libraries
```bash
pip3 install brica2==0.5
pip3 install matplotlib==3.1.1
pip3 install numpy==1.17.3 
pip3 install autograd==1.3
pip3 install -i https://test.pypi.org/simple/ matchernet-py-001
```
### Modify PYTHONPATH
```bash
export PYTHONPATH="/path/to/dir:$PYTHONPATH"
```
You need this if you are to modify files in the matchernet original packages.

### Run a demo
```bash
Python3 demos/demo_ekf.py
```

## For ARM server
Setting up virtual environment at the ARM server with many cores.

- First create a directory for the environments
```bash
$ mkdir ~/virt_env
```

- Then load the module

```bash
$ module load loadonly/python_wrapper
```

- It is a good idea to add this line to your ~/.bash_rc.

- Then create a virtual environment. 
- You create a python3 environment called mynewenv. The environment will be immediately activated with

```bash
$ mkvirtualenv -p /usr/bin/python3 mynewenv
```

- You may directly install the GPU version of tensorflow with

```bash
$ pip install tensorflow-gpu tensorboard
```

- You can deactivate the current environment with

```bash
$ deactivate mynewenv
```

- You can activate it with

```bash
$ workon mynewenv
```

Activate the virtual environment

```bash
$ module load loadonly/python_wrapper
$ source ~/virt_env/mynewenv/bin/activate
```

Installing BriCA2 parallel (beta)

```bash
$ git clone https://github.com/BriCA/BriCA2
$ git checkout wip/python
$ cd BriCA2
```

activate the virtual environment

```bash
$ pip install .
```

# For more info

## For Developer

By releasing the test, you will be using the `pip install`, but if you use this, the development status will not be reflected until the version goes up.
If you want to use the latest branch that you have modified or developed at hand, or you want to use the latest branch, raise the priority of the one that is currently in development.

If you do it in a file, for example:

```python
import sys
import os
sys.path.insert(1, os.getcwd())
# When searching a library, the current directory has the next highest priority after the executable file path
```

If you want to do it from outside the file, for example, change the PYTHONPATH setting. for example,
```bash
export PYTHONPATH="/path/to/dir:$PYTHONPATH"
```
If you want to make the settings permanent, write the settings in `.bashrc` etc.

