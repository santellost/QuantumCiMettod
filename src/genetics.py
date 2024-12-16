# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:56:45 2024

@author: stefa
"""

import utils
from individual import Individual

import random
import numpy as np
import multiprocessing
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from deap import base, tools, algorithms


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


def swap_layers(qc: Individual) -> tuple[Individual]:
    '''
    Swap two random layers

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


def paramters_mutation(qc: Individual) -> tuple[Individual]:
    '''
    Mutate a random parameter of a random gate

    Parameters
    ----------
    qc : Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : Individual
        Mutated quantum circuit.

    '''
    for i in random.sample(range(len(qc)), len(qc)):
        for j in random.sample(range(len(qc[i])), len(qc[i])):
            if len(qc[i][j][0].params) > 0:
                k = random.choice(range(len(qc[i][j][0].params)))
                try:
                    std = qc.fitness.values[0]
                except IndexError:
                    std = 1
                qc[i][j][0].params[k] = random.gauss(qc[i][j][0].params[k], std)
                return qc,


def insert_mutation(qc: Individual) -> tuple[Individual]:
    '''
    Tries to add a random gate in a random layer, if it can't and it's depth 
    is less than max_depth create a new layer with the new gate

    Parameters
    ----------
    qc : Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : Individual
        Mutated quantum circuit.

    '''
    layers = random.sample(range(len(qc)), len(qc))
    for i in layers:
        qubits = list(range(qc.num_qubits))
        for gate, used_qubits in qc[i]:
            qubits = list(filter(lambda x: not x in used_qubits, qubits))
        
        if len(qubits) > 0:
            gate = utils.random_gate(len(qubits))
            qc[i].append((gate, random.sample(qubits, gate.num_qubits)))
            return qc,
    
    # Tries to create a new layer
    if len(qc) < qc.max_depth:
        gate = utils.random_gate(qc.num_qubits)
        qubits = random.sample(range(qc.num_qubits), gate.num_qubits)
        qc.append([(gate, qubits)])
    return qc,


def delete_mutation(qc: Individual) -> tuple[Individual]:
    '''
    Tries to delete a single gate without leaving an empty layer

    Parameters
    ----------
    qc : Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : Individual
        Mutated quantum circuit.

    '''
    layers = random.sample(range(len(qc)), len(qc))
    for i in layers:
        if len(qc[i]) > 1:
            del qc[i][random.randrange(len(qc[i]))]
            return qc,
    
    # Tries to delete a layer
    if len(qc) > qc.min_depth:
        i = random.randrange(len(qc))
        del qc[i]
    return qc,


def debloat_mutation(qc: Individual) -> tuple[Individual]:
    '''
    Tries to delete a many layers to reduce bloating

    Parameters
    ----------
    qc : Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : Individual
        Mutated quantum circuit.

    '''
    if len(qc) <= qc.min_depth:
        return qc,
    
    i = random.randrange(qc.min_depth-1, len(qc)-1)
    j = random.randrange(i+1, len(qc))
    qc[i:] = qc[j:]
    
    return qc,


def mutate(qc: Individual, insert: float = 0.1, delete: float = 0.1,
           flip: float = 0.1, layers: float = 0.1, qubits: float = 0.1,
           debolat: float = 0.1, params: float = 0.1) -> tuple[Individual]:
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
    if random.random() < insert:
        insert_mutation(qc)
    if random.random() < delete:
        delete_mutation(qc)
    if random.random() < flip:
        gate_flip(qc)
    if random.random() < layers:
        swap_layers(qc)
    if random.random() < qubits:
        swap_qubits(qc)
    if random.random() < delete:
        debloat_mutation(qc)
    if random.random() < params:
        paramters_mutation(qc)
    return qc,

def genetic(desired: Statevector, ngen: int = 500, npop: int = 300,
            min_depth: int = 2, max_depth: int = 15, cxpb: float = 1,
            mutpb: float = 0.5, tourn_ratio: float = 0.05):
    tourn_size = int(tourn_ratio*npop)

    toolbox = base.Toolbox()
    toolbox.register('individual', Individual.from_random_gates, num_qubits=desired.num_qubits, min_depth=min_depth, max_depth=max_depth)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=tourn_size)
        toolbox.register("circuit_builder", Individual.build_circuit)
        toolbox.register('evaluate', l_distance, toolbox.circuit_builder, desired)
        
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        
        best = tools.HallOfFame(1)
        _, logbook = algorithms.eaSimple(toolbox.population(npop), toolbox, cxpb, mutpb, ngen, stats, best)
    return toolbox.circuit_builder(best[0]), logbook


def random_walk(desired: Statevector, ngen: int = 50, min_depth: int = 2, max_depth: int = 40) -> tuple[QuantumCircuit, tools.Logbook]:
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
    best = Individual.from_random_gates(desired.num_qubits, min_depth, max_depth)
    best.fitness.values = l_distance(Individual.build_circuit, desired, best)
    logbook.record(gen=0, fitness=best.fitness.values[0])
    for i in range(1, ngen+1):
        current = Individual.from_random_gates(desired.num_qubits, min_depth, max_depth)
        current.fitness.values = l_distance(Individual.build_circuit, desired, current)
        logbook.record(gen=i, fitness=current.fitness.values[0])
        if best == None or best.fitness.values < current.fitness.values:
            best = current
    return best, logbook