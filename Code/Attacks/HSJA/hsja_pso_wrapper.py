import csv

import numpy as np
from keras.models import load_model

from Attacks.HSJA.hsja_position import HSJAPosition
from Attacks.HSJA.hsja_swarm import HSJASwarm

if __name__ == '__main__':
    np.random.seed(42)
    n_particles = 5
    model = load_model('../../MNIST/models/mnist', compile=False)
    inits = []
    for n in range(n_particles):
        init_n_evals = np.random.randint(0, 200)
        max_n_evals = np.random.randint(200, 600)
        gamma = np.random.random()
        step_size_decrease = np.random.uniform(1.01, 5)
        inits.append(HSJAPosition(init_n_evals=init_n_evals, max_n_evals=max_n_evals, gamma=gamma,
                                  step_size_decrease=step_size_decrease))

    swarm = HSJASwarm(n_particles, inits, model)
    for _ in range(3):
        swarm.optimize()
        p, _ = swarm.get_best_particle()
        with open('hsja_pso.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([swarm.total_queries, swarm.best_fitness, p.init_n_evals, p.max_n_evals, p.step_size_decrease, p.gamma])
