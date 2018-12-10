from envs.centipede.centipede import Centipede8

env = Centipede8(GUI=True)
for i in range(1000):
    env.step(env.random_action())


