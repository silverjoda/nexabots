from envs.centipede.centipede import Centipede

env = Centipede(8, gui=True)
for i in range(10000):
    env.step(env.random_action())


