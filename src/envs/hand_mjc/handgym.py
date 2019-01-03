import gym

env = gym.make("HandManipulateEgg-v0")
env_uw = env.unwrapped
sim = env_uw.sim
state = sim.get_state()

pass