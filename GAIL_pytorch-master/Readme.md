# GAIL

This project is implement under two classical control problem: *Cartpole* and *Pendulum*, which represent discrete and continuous case respectively.

* First collect the expert trajectories by the PPO algorithm.
* Then utilize these expert trajectories to imitate them with GAIL.
* The paper use TRPO to optimize the policy net, however I use **PPO** with **GAE** here.