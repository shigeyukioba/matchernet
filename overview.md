# An overview of MatcherNet
## What is MatcherNet?
MatcherNet is a probabilistic state-space model for dynamic system identification and control. You can design a dynamic model of the world especially when it has high-dimensional / multi-modal / multi-scale states by combining simple building blocks.

## MatcherNet as a model of animal mind
We can regard the MatcherNet as a model of the recognition process of animal's mind in a dynamic environment. Consider an animal who senses the environment and recognizes something in its mind. We formally take this process as an on-line update of the recognition state  x  describing the internal state of the mind. The recognition state  x  is updated at each observation of the sense channel  y. Here, the observation can consists of multi-modal channels; namely, visual inputs from the eyes, sound signals from the ears, somato sensor from each part of the musculoskeltal system etc. 

Predictive coding (PC) model, and its extended version, free-energy principle (FEP), provide principled ways to derive the recognition process. In PC/FEP, the recognition state  x  is considered to include the information to predict the observation  y  and updated in order to predict the observation better. Thus, under this principle, the objective function of the update is nothing other thant the likelihood function  p( y | x ). In FEP, specifically, we consider the posterior recognition state  q(x)  rather than  x.


A MatcherNet model consists of two types of module, Bundle and Matcher. A bundle module has a state variable and updates the current state by applying its own dynamics model. A matcher has a link between a set of (typically two) bundles, monitoring the states, calculates the distortion (or mutual prediction error), and feeds back the state update signals that decreases the distortion. In total, the current state of the all the bundles are recognizing the state of the external world from the set of on-line observations.

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
