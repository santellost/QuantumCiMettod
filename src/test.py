# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:58:30 2024

@author: stefa
"""

from genetics import genetic, random_walk
import visualization as vis
import utils

import pandas as pd
from deap.tools import Logbook
from qiskit.quantum_info import Statevector, random_statevector

from IPython.display import display


def simple_test(state: Statevector):
    num_qubits = 4
    initial = Statevector.from_label('0' * num_qubits)
    
    ngen = 300
    best, genetic_logbook = genetic(state, ngen=ngen)
    _, random_logbook = random_walk(state, ngen=ngen)
    
    vis.plot_logbook(genetic_logbook, Random=random_logbook)
    if num_qubits < 7:
        vis.compare_histograms(best, state)
    print('Evolved is equivalent to desired:', state.equiv(initial.evolve(best)))
    
    display(best.draw('mpl'))
    display(initial.evolve(best).draw('latex'))


def iterate_test(state: Statevector, num_iters: int = 5, **kwargs) -> Logbook:
    logbooks = []
    ngen = kwargs.pop('ngen', 500)
    max_depth = kwargs.pop('max_depth', 15)
    min_depth = kwargs.pop('min_depth', 2)
    for i in range(num_iters):
        _, logbook = genetic(state, ngen, min_depth=min_depth, max_depth=max_depth, **kwargs)
        logbooks.append(logbook)
    random = {'Random': [random_walk(state, ngen, min_depth=min_depth, max_depth=max_depth)[1] for _ in range(num_iters)]}
    vis.plot_logbook(*logbooks, **random)
    return logbooks
    
    
def test_fixed_qubits(*states: Statevector, num_iters: int = 5, plot_final: bool = True, **kwargs):
    logbooks = []
    randoms = []
    ngen = kwargs.pop('ngen', 500)
    max_depth = kwargs.pop('max_depth', 15)
    min_depth = kwargs.pop('min_depth', 2)
    for state in states:
        logbooks.extend(iterate_test(state, num_iters, ngen=ngen, min_depth=min_depth, max_depth=max_depth, **kwargs))
    if plot_final and len(states) > 1:
        randoms.extend([random_walk(state, ngen, ngen=ngen, min_depth=min_depth, max_depth=max_depth, **kwargs)[1] for _ in range(num_iters*len(states))])
        vis.plot_logbook(*logbooks, **{'Random': randoms})


def grid_search(state: Statevector, num_iters: int = 5, **kwargs):
    cxpbs = [0.75, 1]
    mutpbs = [0.01, 0.1, 0.3, 0.5]
    depths = [15, 20, 25]
    tourn_ratios = [0.02, 0.05, 0.1]
    data = pd.DataFrame()
    for cxpb in cxpbs:
        for mutpb in mutpbs:
            for max_depth in depths:
                for tourn_ratio in tourn_ratios:
                    logbooks = [genetic(state, cxpb=cxpb, mutpb=mutpb, max_depth=max_depth, tourn_ratio=tourn_ratio, **kwargs)[1] for _ in range(num_iters)]
                    fitnesses = []
                    for i, logbook in enumerate(logbooks):
                        generations = zip(logbook.select('gen'), logbook.select('min'))
                        fitnesses.extend([(i, gen, fitness) for gen, fitness in generations])
                    temp = pd.DataFrame({
                        'Generations': [gen for _, gen, _ in fitnesses],
                        'Fitness': [fitness for _, _, fitness in fitnesses],
                        'Iteration': [iteration for iteration, _, _ in fitnesses],
                        'Crossing-over probability': [cxpb] * len(fitnesses),
                        'Mutation probability': [mutpb] * len(fitnesses),
                        'Max depth': [max_depth] * len(fitnesses),
                        'Tournament ratio': [tourn_ratio] * len(fitnesses)
                        })
                    data = pd.concat([data, temp])
    data = utils.update_file('data.csv', data)
    vis.plot_grid_search(data)
                    

if __name__ == '__main__':
    paper = Statevector([ -0.139-0.117j, -0.03-0.437j, 0.155+0.311j,
                           -0.341+0.404j, 0+0j, 0+0j, -0.057+0.012j, 0.011-0.021j, 0.09-0.107j, 0.335-0.023, -0.239+0.119j,
                           -0.31-0.262j, 0+0j, 0+0j, 0.027+0.007j, 0.007+0.027j])
    #simple_test(paper)
    #test_fixed_qubits(paper, num_iters=10, ngen=500)
    grid_search(paper, ngen=500, num_iters=15)