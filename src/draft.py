import cma
import numpy as np
import time


es = cma.CMAEvolutionStrategy(np.random.randn(1000) * 0.1, 0.5)
while not es.stop():
    X = es.ask()
    # X = [(1 - a) * x for x in X]
    es.tell(X, [np.square(x).sum() for x in X])
    # es.mean *= 1 - a

    print(es.mean.min(), es.mean.max())
    es.logger.add()
    es.disp()

#es.logger.plot()
#cma.plot()
#time.sleep(5)

