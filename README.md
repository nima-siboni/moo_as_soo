# RL for Adaptive Multi-Objective Sequential Decision-Making
Welcome to the RL for Adaptive Multi-Objective Sequential Decision-Making repository! This project aims to tackle
one of the challenges of multi-objective optimization in the context of decision-making problems by leveraging the power of single objective optimization
(SOO) reinforcement learning (RL) techniques.

# Introduction
In conventional optimization methods, dealing with multiple objectives often involves combining
these objectives into a single objective function; The single objective function is commonly constructed as a weighted
average of different objectives, where the weights are chosen by the domain experts, and the weights represent the
condition under which the optimal behavior is looked for. This approach is widely used in
industry as it is easy to implement and the optimization problem is reduced to a SOO which can be solved with
well-established methods.

A disadvantage of casting the MOO into a SOO is that for any new set of weights one needs to re-run the whole optimization procedure. In many cases, this is not feasible as the
optimization procedure is computationally expensive, e.g. "online" decision-making setups where the weights change and
immediate adaption of the objective is required. In this work, we address this challenge by leveraging the power of
 single objective reinforcement learning.  A simple example of changing weights is the multi-objective navigation where
you want to find the optimal behavior for reaching your destination (i) as quick as possible and (ii) as fuel efficient
as possible. The weights for combining these two objectives could change depending on your situation. For example, in
case of an emergency you might want to reach your destination as quick as possible and fuel consumption is not a concern.
On the other hand, if you are on a long trip, you might want to save fuel and the time efficiency is not the greater concern.
In this example, the weights for combining the two objectives are different in the two situations. In these cases the
optimization should be re-run for the new circumstance (i.e. new weights). This is not feasible in many cases as the
optimization procedure is computationally expensive, specially in realistic applications. In this work, we address this
challenge by leveraging the power of single objective reinforcement learning.

More precisely, we train an agent to behave optimally under any set of weights. This way, the agent can be used to address
the MOO problem without the need to re-run the optimization procedure for different weights, although training itself
becomes more challenging.

We achieve this by exposing the agent to different conditions (reflected in the weights) during the training while at
the same time we inform the agent about the weights under which it is operating. This way the agent has the chance of
learning the optimal behavior for any set of weights. More details and results are shown in the "approach" and "demo"
sections.


# Our approach:
As outlined above, our approach to address the adaptiveness required for changing the weights of the objectives is to
train an agent to behave optimally under any set of weights. More precisely, this is done by three modifications in the
environment:

- **Augmenting the Observation:** we modify the environment by passing the weights used for casting to the agent
as a part of the environment's observation. This is the first step to give the agent has the chance of learning the
optimal behavior in different circumstances. The first step can be done with a minimal wrapper which exposes the values
of the weights in the observations. A sample wrapper is provided in [env_utils.py](./env_utils.py)
- **Covering the Relevant Paremeter Range:** during the training (more precisely during interactions with the environment,
where transition samples are collected) we need to create different circumstances for the agent. This can be done by
creating instances of the environment which are different from each other in the values of those weights. This can also
be done in a wrapper which overwrites the `reset` method of the environment, if this feature is not already implemented in
the environment. For the test environment we used in this repository this is already implemented in the environment itself.
- **Augmenting the Reward:** we modify the environment by passing the weights used for casting to the agent.
This can also be done in a wrapper which overwrites the `step` method of the environment, if this feature is not already
implemented in the environment. For the test environment we used in this repository this is already implemented in
the environment itself.

# Characteristics of our approach
Characteristics of the approach presented in this repository includes:

**Adaptive Optimization**: The agent can dynamically
adjust its actions based on the observed weights, even during one episode. This could be very interesting
for business where the weights of different objectives can change when the process is already started. The approach here
allows for **online reaction to the changes in the objective functions**. An example is shown in the "Demo" section.
One should emphasize that the computational cost of the evaluating the performance of a trained agent under changing
conditions can be order of magnitudes smaller than re-running the optimization procedure for the new weights.

**Flexibility in Objective Trade-offs**: By formulating multi-objective optimization as a reinforcement learning
problem, this approach allows for flexible trade-offs between objectives.
The agent can learn to adapt its decision-making based on the specific weights assigned to each objective, capturing the
complexities and trade-offs inherent in multi-objective scenarios.

**Pareto Front Exploration in Sequential Decision-Making Settings**: As the computational cost of changing the weights of the objective is much smaller than of
the similar approach with conventional optimization techniques, the trained RL agent can be used to explore the Pareto front,
identifying non-dominated solutions that represent optimal trade-offs between objectives. This enables decision-makers
to analyze and visualize a diverse set of solutions, empowering them to make informed decisions based on their
preferences.

# Installation and Usage
First create a python virtual environment (for example with `conda`) and activate it.
```bash
conda create -n moo_as_soo python=3.10
conda activate moo_as_soo
```
```bash
git clone
cd moo_as_soo
pip install -e .
```
# Demo
For demonstration purposes, we used the simple environment of [UnevenMaze](https://github.com/nima-siboni/uneven_maze).
In this environment, an agent shall find the path to a destination while minimizing:
- **1st objective** the number of steps, and
- **2nd objective** the energy cost for the steps. The maze has an uneven terrain which leads step costs depending on the elevation changes.

This environment offers a test bed for our approach; in particular, the height profile is designed such that
the shortest path (the path with the least number of steps) is not necessarily the path with the least energy cost! This
offers an interesting situation where the agent should find different solutions (i.e. different compromises) depending on
the weights of different objectives.

An agent is trained with different step costs and height costs (which are effectively the weights of these two contributions in the combined objective function).

In the following, we demonstrate the trajectory taken by the agent depending on the unit energy cost for taking a step; This
emulate the fuel cost if our agent was a vehicle! In particular, we consider three cases:

- a case where the fuel is for free,
- a case where the fuel is as high as possible (as high as agent has seen during the training), and most interestingly,
- a case where at the beginning of the episode the fuel is as expensive as it gets, and after a number of steps
the fuel becomes cheaper and its price goes to zero.

## Zero cost fuel
The agent goes as quick as possible (effectively optimizing only 1st objective, i.e. the number of steps) toward the goal on a straight line.

This is the optimal behavior in this setup.
![](images/step_10.png)

## Very expensive fuel
Here, we set the cost of the fuel as high as the agent has seen during the training. This should led to agent avoiding the high slopes and going straightly towards
the destination, but rather taking a detour around the high slop areas. The trained agent, at least qualitatively, shows this behavior as depicted in the figure below.
![](images/step_40.png)

## Fuel becomes free: an online reaction to change of the objective
Here, we demonstrate one of the strength points of this approach, i.e. the adaptivity built into the agent concerning the change of different
objective weights during the episode. An example of such a situation is where the agent starts acting when the fuel has one price, but the
cost of the fuel suddenly changes to a different price and the agent should online adapt its strategy. This is the test carried out here.

In order to show the effects visibily, we change the fuel price from the highest possible value to zero in one step (at step 10)!


![](images/step_21.png)

Interesting, the agent starts going on low slope path when the fuel is expensive but as soon as the price
changes to zero it changes the course and goes as quick as possible to the destination (which goes over the high slopes).
