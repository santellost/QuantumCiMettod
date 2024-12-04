# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:56:45 2024

@author: stefa
"""

import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import library
from qiskit.circuit import Gate
from qiskit.circuit.singleton import SingletonGate
from qiskit.quantum_info import Statevector
from deap import base, creator, tools, algorithms
import visualization as vis


NO_PARAMS = [g() for g in library.__dict__.values() if isinstance(g, type) and issubclass(g, SingletonGate)]
# TODO find a method to use parameter gates


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, num_qubits=int)


def init_ind(icls, generator, num_qubits, max_depth):
    '''
    Initialize individual
    '''
    ind = icls(generator(num_qubits, max_depth))
    ind.num_qubits = num_qubits
    ind.max_depth = max_depth
    return ind


def random_gate(max_qubits: int) -> Gate:
    '''
    Randomly extract an unparametrized gate

    Parameters
    ----------
    max_quibts : int
        Inclusive max number of qubits.

    Returns
    -------
    Gate
        Random gate.

    '''
    return random.choice(list(filter(lambda g: g.num_qubits <= max_qubits and g.name != 'id', NO_PARAMS)))


def random_gates(num_qubits: int, max_depth: int) -> list[list[tuple[Gate, list[int]]]]:
    '''
    Creates a random circuit based on random gates

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    max_depth : int
        Max depth of the circuit.

    Returns
    -------
    qc : list[list[tuple[Gate, list[int]]]]
        The first index decides the layer of the circuit, the second the number of the gate.

    '''
    qc = []
    for i in range(max_depth):
        layer = []
        qubits = list(range(num_qubits))
        random.shuffle(qubits)
        while len(qubits) > 0 and random.random() < 0.5:
            gate = random_gate(len(qubits))
            selected = qubits[:gate.num_qubits]
            layer.append((gate, selected))
            qubits = qubits[gate.num_qubits:]
        qc.append(layer)
    return qc


def build_from_list(qc: creator.Individual) -> QuantumCircuit:
    '''
    Builds the quantum circuit as defined in the list

    Parameters
    ----------
    qc : list
        Quantum circuit evolved as list.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    _qc : QuantumCircuit
        Qiskit quantum circuit.

    '''
    _qc = QuantumCircuit(qc.num_qubits)
    for layer in qc:
        for gate in layer:
            _qc.append(gate[0], gate[1])
    return _qc


def l_distance(circuit_builder: callable, desired: Statevector, qc: creator.Individual, 
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


def cosine_similarity(circuit_builder: callable, desired: Statevector, qc: creator.Individual,
                      ignorePhase: bool = True, rtol: float = 1e-5, atol: float = 1e-8) -> tuple[float]:
    '''
    Evaluate the quantum circuit by computing the cosine similarity between the 
    desired output vector and the computed vector

    Parameters
    ----------
    circuit_builder : callable
        Function used to build the qiskit circuit
    desired : Statevector
        Desired output vector.
    qc : QuantumCircuit
        Quantum circuit built by the genetic algorithm.
    ignorePhase : bool, optional
        If true ignore global phase difference. The default is True.
    rtol : TYPE, optional
        Relative tollerance to use during computation. The default is 1e-5.
    atol : TYPE, optional
        Absolute tollerance to use during computation. The default is 1e-8.

    Returns
    -------
    tuple[float]
        Cosine similarity.

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
    return np.abs((psi - desired).inner(psi- desired)/(np.linalg.norm(psi) * np.linalg.norm(desired))), 


def insert_mutation(qc: creator.Individual) -> tuple[creator.Individual]:
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
    gate = random_gate(qc.num_qubits)
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


def delete_mutation(qc: creator.Individual) -> tuple[creator.Individual]:
    '''
    Delete a random gate from the circuit

    Parameters
    ----------
    qc : creator.Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : tuple[creator.Individual]
        Mutated quantum circuit.

    '''
    for i in random.sample(range(len(qc)), len(qc)):
        if len(qc[i]) > 0:
            j = random.randrange(len(qc[i]))
            del qc[i][j]
            return qc,
    return qc,


def gate_flip(qc: creator.Individual) -> tuple[creator.Individual]:
    '''
    Replace a gate with a new random gate

    Parameters
    ----------
    qc : creator.Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : tuple[creator.Individual]
        Mutated quantum circuit.

    '''
    for i in random.sample(range(len(qc)), len(qc)):
        if len(qc[i]) > 0:
            j = random.randrange(len(qc[i]))
            del qc[i][j]
            
            qubits = list(range(qc.num_qubits))
            for gate, used_qubits in qc[i]:
                qubits = list(filter(lambda x: not x in used_qubits, qubits))
                            
            gate = random_gate(len(qubits))
            qc[i].append((gate, random.sample(qubits, gate.num_qubits)))
            return qc,
    return qc,


def swap_columns(qc: creator.Individual) -> tuple[creator.Individual]:
    '''
    Swap two random columns

    Parameters
    ----------
    qc : creator.Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : creator.Individual
        Mutated quantum circuit.

    '''
    i, j = random.sample(range(len(qc)), 2)
    qc[i], qc[j] = qc[j], qc[i]
    return qc,


def swap_qubits(qc: creator.Individual) -> tuple[creator.Individual]:
    '''
    Swap two qubits lines

    Parameters
    ----------
    qc : creator.Individual
        Quantum circuit to mutate.

    Returns
    -------
    qc : creator.Individual
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


def mutate(qc: creator.Individual, inspb: float = 0.3, delpb: float = 0.5,
           flppb: float = 0.4, colpb: float = 0.3, qbtpb: float = 0.1) -> tuple[creator.Individual]:
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
    return qc,


def genetic(desired: Statevector, npop=50, cxpb=0.75, mutpb=0.2, ngen=50):
    toolbox = base.Toolbox()
    toolbox.register('individual', init_ind, creator.Individual, random_gates, num_qubits=desired.num_qubits, max_depth=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=6)
    toolbox.register("circuit_builder", build_from_list)
    toolbox.register('evaluate', l_distance, toolbox.circuit_builder, desired, order=np.inf)
    
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
    best = init_ind(creator.Individual, random_gates, desired.num_qubits, max_depth)
    best.fitness.values = l_distance(build_from_list, desired, best)
    logbook.record(gen=0, fitness=best.fitness.values)
    for i in range(1, ngen+1):
        current = init_ind(creator.Individual, random_gates, desired.num_qubits, max_depth)
        current.fitness.values = l_distance(build_from_list, desired, current)
        logbook.record(gen=i, fitness=current.fitness.values)
        if best == None or best.fitness.values < current.fitness.values:
            best = current
    return best, logbook


if __name__ == '__main__':
    u = Statevector.from_label('0000' )
    psi = Statevector(np.sqrt([1/4, 1/8, 0, 0, 1/2, 1/8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    
    ngen = 30
    best, genetic_logbook = genetic(psi, npop=200, ngen=ngen)
    _, random_logbook = random_walk(psi, ngen)
    
    vis.plot_logbook(genetic_logbook, Random=random_logbook)
    vis.compare_histograms(best, psi)
    
    display(best.draw('mpl'))
    display(u.evolve(best).draw('latex'))
    
    