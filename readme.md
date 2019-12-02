<img alt="PyPI" src="https://img.shields.io/pypi/v/matchernet"> <img alt="APM" src="https://img.shields.io/apm/l/matchernet">

# What is MatcherNet?

MatcherNet is a modular and hierarchical state-space modeling platform for dynamic system identification and control. If you are working with a world model of high-dimensional / multi-modal / multi-scale states and observations for robotics, image processing, sensor networks, and their hybrid cases, MatcherNet may provide you a better solution.

MatcherNet includes Extended Kalman-filter (EKF), a non-linear extension of Kalman-filter, as a special case. Design a pair of observation model p( y_t | x_t ) and dynamics model p( x_t+1 | x_t ), and EKF calculates the posterior of the current state p( x_t | y_1:t ). For MatcherNet, divide the state variable into multiple parts x_t = (x_1t, x_2t, ..., x_kt), design dynamics models for each part, and then, MatcherNet can manage them in a parallel and principled manner.

MatcherNet includes system controllers, such as PID, iLQR (iLQG). Provide a control goal as a prior probability of the state variable, and the controller calculates the control signal that minimizes the current and future distance to the prior. 

MatcherNet efficiently works with multi-thread computing. Modular division of state-space model lower the dimensionality of each state variable, and multiple modules run in parallel in a multi-core computing environment. 


# How to install?


# How to try demos?




### How to use at ARM server?
Setting up virtual environment at the ARM server

- First create a directory for the environments

$ mkdir ~/virt_env

- Then load the module

$ module load loadonly/python_wrapper

- It is a good idea to add this line to your ~/.bash_rc.

- Then create a virtual environment. 
- You create a python3 environment called mynewenv. The environment will be immediately activated with

$ mkvirtualenv -p /usr/bin/python3 mynewenv

- You may directly install the GPU version of tensorflow with

$ pip install tensorflow-gpu tensorboard

- You can deactivate the current environment with

$ deactivate mynewenv

- You can activate it with

$ workon mynewenv


Activate the virtual environment

$ module load loadonly/python_wrapper
$ source ~/virt_env/mynewenv/bin/activate

Installing BriCA2 parallel (beta)

$ git clone https://github.com/BriCA/BriCA2
$ git checkout wip/python
$ cd BriCA2

activate the virtual environment
$ pip install .

# For more info

## For Developer

By releasing the test, you will be using the `pip install`, but if you use this, the development status will not be reflected until the version goes up.
If you want to use the latest branch that you have modified or developed at hand, or you want to use the latest branch, raise the priority of the one that is currently in development.

If you do it in a file, for example:

```
import sys
import os
sys.path.insert(1, os.getcwd())
# When searching a library, the current directory has the next highest priority after the executable file path
```

If you want to do it from outside the file, for example, change the PYTHONPATH setting.

