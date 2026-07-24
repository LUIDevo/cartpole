# Cartpole - an inverted pendulum balancing cart simulation trained through ML

Cartpole is an RL neural network that is trained to balance a pendulum along a singular axis from a motor command. The goal is to scale it to deal with a double or even triple pendulum through simulations, and then via Test Time Training be used in real life.

The network has only 3 layers currently, and is managed by PPO and Adam optimiser.

It takes about 150 steps before the model fully converges and is able to consistently balance the pole forever. 

Outside of the model, we also used 2 types of simulations - Godot, and math using Python

Math was used to quickly train the model without having to deal with pinging delays, as pinging through parallel cores to multiple godot instances drastically increased training times.

Then after we get our weights, we load them and watch the simulation in Godot using 'watch.py'. Inside the simulation, the user is able to attemp to throw the robot off by moving the cart, and the cart is able to rebalance the pole in real time.

Plans for the future:
- [ ] Add guassian noise during training to mimic sensor data
- [ ] Action delay to mimic what a real life pipeline would look like
- [ ] TTT
- [ ] If the above are insufficient for real life, implement wider randomization ranges for training

Then to prepare for double pendulum
- [x] Make a larger model
- [x] Redo the simulation pipeline to deal with a double pendulum (especially godot)

Then to build the real thing
- [ ] Design and build a physical double pendulum balancing robot
- [ ] Train it using ML, using everything above (sim-to-real via TTT / domain randomization)

Current single pendulum balancing demo in simulation
![Cartpole balancing demo](media/demo.gif)
