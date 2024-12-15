# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:58:30 2024

@author: stefa
"""

from genetics import genetic, random_walk
import visualization as vis

import numpy as np
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


def iterate_test(state: Statevector, num_iters: int = 5, ngen: int = 100) -> Logbook:
    random = {'Random': [random_walk(state, ngen)[1] for _ in range(num_iters)]}
    logbooks = []
    for i in range(num_iters):
        _, logbook = genetic(state, ngen)
        logbooks.append(logbook)
    vis.plot_logbook(*logbooks, **random)
    return logbooks
    
    
def test_fixed_qubits(*states: Statevector, num_iters: int = 5, ngen: int = 100, plot_final: bool = True):
    logbooks = []
    randoms = []
    for state in states:
        randoms.extend([random_walk(state, ngen)[1] for _ in range(num_iters)])
        logbooks.extend(iterate_test(state, num_iters, ngen))
    if plot_final and len(states) > 1:
        vis.plot_logbook(*logbooks, **{'Random': randoms})


if __name__ == '__main__':
    paper = Statevector([ -0.139-0.117j, -0.03-0.437j, 0.155+0.311j,
                           -0.341+0.404j, 0+0j, 0+0j, -0.057+0.012j, 0.011-0.021j, 0.09-0.107j, 0.335-0.023, -0.239+0.119j,
                           -0.31-0.262j, 0+0j, 0+0j, 0.027+0.007j, 0.007+0.027j])
    #simple_test(paper)
    test_fixed_qubits(paper, num_iters=10, ngen=500)