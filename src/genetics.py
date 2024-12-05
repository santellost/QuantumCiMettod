# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:56:45 2024

@author: stefa
"""

import utils
from individual import Individual

import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from deap import base, tools, algorithms
import visualization as vis


def l_distance(circuit_builder: callable, desired: Statevector, qc: Individual, 
             order: int | type(np.inf) = 2, ignorePhase: bool = True, rtol: float = 1e-5, atol: float = 1e-8) -> tuple[float]:
    '''
    Evaluate the quantum circuit by computing the l-order distance between the 
    desired output vector and the computed vector

    Parameters
    ----------
    circuit_builder : callable
        Function used to build the qiskit circuit
    desired : Statevector
        Desired output vector.
    qc : QuantumCircuit
        Quantum circuit built by the genetic algorithm.
    order : int | type(np.inf), optional
        Order of the norm. The default is 2
    ignorePhase : bool, optional
        If true ignore global phase difference. The default is True.
    rtol : TYPE, optional
        Relative tollerance to use during computation. The default is 1e-5.
    atol : TYPE, optional
        Absolute tollerance to use during computation. The default is 1e-8.

    Returns
    -------
    float
        Euclidean distance between the two vectors.

    '''
    psi = Statevector.from_label('0' * qc.num_qubits)
    psi = psi.evolve(circuit_builder(qc))
    desired = desired.copy()    
    if ignorePhase:
        # Get phase of first non-zero entry of psi and out
        # and multiply all entries by the conjugate
        for elt in psi:
            if abs(elt) > atol:
                angle = np.angle(elt)
                psi = np.exp(-1j * angle) * psi
                break
        for elt in desired:
            if abs(elt) > atol:
                angle = np.angle(elt)
                desired = np.exp(-1j * angle) * desired
                break
    return np.linalg.norm(psi - desired, ord=order), 


def insert_mutation(qc: Individual) -> tuple[Individual]:
    '''
    Insert or append a random gate to the circuit

    Parameters
    ----------
    qc : creator.Individual
        Quantum circuit to mutate.
    inspb : float
        Probability of the mutation to happen.

    Returns
    -------
    qc : tuple[creator.Individual]
        Mutated quantum circuit.

    '''
    gate = utils.random_gate(qc.num_qubits)
    queue = list(range(len(qc) + 1))
    random.shuffle(queue)
    while queue:
        index = queue.pop()
        qubits = list(range(qc.num_qubits))
        if index == len(qc):
            selected = random.sample(qubits, gate.num_qubits)
            qc.append([(gate, selected)])
            return qc,
        
        for gate, used_qubits in qc[index]:
            qubits = list(filter(lambda x: not x in used_qubits, qubits))
        
        if len(qubits) < gate.num_qubits:
            continue
        
        selected = random.sample(qubits, gate.num_qubits)
        qc[index].append((gate, selected))
        return qc,
    raise ValueError('Could not append the selected gate')


def delete_mutation(qc: Individual) -> tuple[Individual]:
    '''
    Delete a random gate from the circuit

    Parameters
    ----------
    qc : Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : tuple[Individual]
        Mutated quantum circuit.

    '''
    for i in random.sample(range(len(qc)), len(qc)):
        if len(qc[i]) > 0:
            j = random.randrange(len(qc[i]))
            del qc[i][j]
            return qc,
    return qc,


def gate_flip(qc: Individual) -> tuple[Individual]:
    '''
    Replace a gate with a new random gate

    Parameters
    ----------
    qc : Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : tuple[Individual]
        Mutated quantum circuit.

    '''
    for i in random.sample(range(len(qc)), len(qc)):
        if len(qc[i]) > 0:
            j = random.randrange(len(qc[i]))
            del qc[i][j]
            
            qubits = list(range(qc.num_qubits))
            for gate, used_qubits in qc[i]:
                qubits = list(filter(lambda x: not x in used_qubits, qubits))
                            
            gate = utils.random_gate(len(qubits))
            qc[i].append((gate, random.sample(qubits, gate.num_qubits)))
            return qc,
    return qc,


def swap_columns(qc: Individual) -> tuple[Individual]:
    '''
    Swap two random columns

    Parameters
    ----------
    qc : Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : Individual
        Mutated quantum circuit.

    '''
    i, j = random.sample(range(len(qc)), 2)
    qc[i], qc[j] = qc[j], qc[i]
    return qc,


def swap_qubits(qc: Individual) -> tuple[Individual]:
    '''
    Swap two qubits lines

    Parameters
    ----------
    qc : Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : Individual
        Mutated quantum circuit.

    '''
    i, j = random.sample(range(qc.num_qubits), 2)
    for k in range(len(qc)):
        for l in range(len(qc[k])):
            t = []
            for m in range(len(qc[k][l][1])):
                if qc[k][l][1][m] == i:
                    t.append(j)
                elif qc[k][l][1][m] == j:
                    t.append(i)
                else:
                    t.append(qc[k][l][1][m])
            t = tuple(t)
            qc[k][l] = (qc[k][l][0], t)
    return qc, 


def debloat_mutation(qc: Individual) -> tuple[Individual]:
    i = random.randrange(len(qc))
    j = random.randrange(i, len(qc))
    for _ in range(j-i):
        del qc[i]
    return qc,


def mutate(qc: Individual, inspb: float = 0.3, delpb: float = 0.7,
           flppb: float = 0.4, colpb: float = 0.3, qbtpb: float = 0.1,
           dblpb: float = 0.2) -> tuple[Individual]:
    '''
    Mutate the individual by:
        Inserting a new gate
        Deleting a gate
        Flip a gate with a different gate

    Parameters
    ----------
    qc : creator.Individual
        Quantum circuit to mutate.

    Returns
    -------
    tuple[creator.Individual]
        Mutated quantum circuit.

    '''
    # TODO add other mutations
    if random.random() < inspb:
        insert_mutation(qc)
    if random.random() < delpb:
        delete_mutation(qc)
    if random.random() < flppb:
        gate_flip(qc)
    if random.random() < colpb:
        swap_columns(qc)
    if random.random() < qbtpb:
        swap_qubits(qc)
    if random.random() < dblpb:
        debloat_mutation(qc)
    return qc,


def genetic(desired: Statevector, npop=50, cxpb=0.75, mutpb=0.2, ngen=50):
    toolbox = base.Toolbox()
    toolbox.register('individual', Individual.from_random_gates, num_qubits=desired.num_qubits, max_depth=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("circuit_builder", Individual.build_circuit)
    toolbox.register('evaluate', l_distance, toolbox.circuit_builder, desired)
    
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    best = tools.HallOfFame(1)
    _, logbook = algorithms.eaSimple(toolbox.population(npop), toolbox, cxpb, mutpb, ngen, stats, best)
    return toolbox.circuit_builder(best[0]), logbook


def random_walk(desired: Statevector, ngen: int = 50, max_depth: int = 5) -> tuple[QuantumCircuit, tools.Logbook]:
    '''
    Simplest evolutionary algorithm: generates a random quantum circuit each generation

    Parameters
    ----------
    desired : Statevector
        Desired statevector.
    ngen : int, optional
        Number of generation to evolve. The default is 50.
    max_depth : int, optional
        Max depth of the circuit. The default is 5.

    Returns
    -------
    best : QuantumCircuit
        Best quantum circuit from all generations.
    logbook : tools.Logbook
        Logbook with 'gen' and 'fitness' columns.

    '''
    logbook = tools.Logbook()
    best = Individual.from_random_gates(desired.num_qubits, max_depth)
    best.fitness.values = l_distance(Individual.build_circuit, desired, best)
    logbook.record(gen=0, fitness=best.fitness.values)
    for i in range(1, ngen+1):
        current = Individual.from_random_gates(desired.num_qubits, max_depth)
        current.fitness.values = l_distance(Individual.build_circuit, desired, current)
        logbook.record(gen=i, fitness=current.fitness.values)
        if best == None or best.fitness.values < current.fitness.values:
            best = current
    return best, logbook


if __name__ == '__main__':
    num_qubits = 2
    initial = Statevector.from_label('0' * num_qubits)
    desired = Statevector(np.sqrt([1/2, 0, 0, 1/2]))
    
    ngen = 60
    best, genetic_logbook = genetic(desired, npop=100, ngen=ngen)
    _, random_logbook = random_walk(desired, ngen)
    
    vis.plot_logbook(genetic_logbook, Random=random_logbook)
    vis.compare_histograms(best, desired)
    print('Evolved is equivalent to deisred:', desired.equiv(initial.evolve(best)))
    
    display(best.draw('mpl'))
    display(initial.evolve(best).draw('latex'))
    
    