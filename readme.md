# What is MatcherNet?

MatcherNet is a modular and hierarchical state space modeling platform for dynamic system indentification and control. If you have a world of high-dimensional / multi-modal / multi-scale states and observations, MatcherNet provides you a simple and robust modeling process.

MatcherNet includes, Extended Kalman-filter (EKF), a non-linear extension of Kalman-filter, as a special case. If you design a pair of observation model p( y_t | x_t ) and dynamics model p( x_t+1 | x_t ), EKF calculates the posterior of the current state  p( x_t | y_1:t ). MatcherNet can divides the state variable into multiple parts x_t = (x_1t, x_2t, ..., x_kt) and manages the dynamics model for each parts. space model into multiple modules, and you can easily design each module and efficiently estimate the parameters.

MatcherNet can be used with PID, LQR (LQG), iLQR (iLQG) controlers. 

MatcherNet is consistent with the predictive coding framework, or free energy principle (FEP), of brain computing.

MatcherNet efficiently work with multi-thread computing. 


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
