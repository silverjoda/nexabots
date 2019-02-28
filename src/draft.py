import copy
from functools import partial

from pytorch_es import EvolutionModule
import src.policies as policies
from src.my_utils import to_tensor

def get_reward(model, env, weights, render=False):

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    total_rew = 0

    s = env.reset()
    done = False
    while not done:
        act = model(to_tensor(s, True))
        s, r, done, _ = env.step(act[0].detach().numpy())
        total_rew += r

    return total_rew


from src.envs.hexapod_flat_pd_mjc import hexapod_pd
env = hexapod_pd.Hexapod()
model = policies.NN_PG(env)

# EvolutionModule runs the population in a ThreadPool, so
# if you need to inject other arguments, you can do that
# using the partial tool
partial_func = partial(get_reward, model, env)
mother_parameters = list(model.parameters())

es = EvolutionModule(
    mother_parameters, partial_func, population_size=50,
    sigma=0.1, learning_rate=0.001,
    reward_goal=200, consecutive_goal_stopping=20,
    threadcount=10, cuda=False, render_test=False
)


final_weights = es.run(1000, print_step=10)
reward = partial_func(final_weights, render=True)

