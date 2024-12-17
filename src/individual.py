# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:02:45 2024

@author: stefa
"""

import utils
import random
import numpy.random
from deap import creator, base
from qiskit import QuantumCircuit


try:
    del creator.FitnessMin
except Exception:
    pass

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

class Individual(list):
    def __init__(self, num_qubits: int, min_depth: int, max_depth: int, iterable: iter):
        super().__init__(iterable)
        self.fitness = creator.FitnessMin()
        self.num_qubits = num_qubits
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    
    def from_random_gates(num_qubits: int, min_depth: int, max_depth: int):
        '''
        Generates a random circuit from random gates

        Parameters
        ----------
        num_qubits : int
            Total number of qubits.
        min_depth : int
            Inclusive minimum depth of the circuit.
        max_depth : int
            Inclusive maximum depth of the circuit.

        Returns
        -------
        Individual
            Quantum circuit as a list.

        '''
        qc = []
        depth = random.randints(min_depth, max_depth)
        for _ in range(depth):
            layer = []
            # Random number of random qubits used as an input of the layer's gates
            qubits = random.sample(range(num_qubits), random.randint(1, num_qubits))
            while len(qubits) > 0:
                gate = utils.random_gate(len(qubits))
                selected = qubits[:gate.num_qubits]
                layer.append((gate, selected))
                qubits = qubits[gate.num_qubits:]
            qc.append(layer)
        return Individual(num_qubits, min_depth, max_depth, qc)
    
    
    def build_circuit(self) -> QuantumCircuit:
        '''
        Builds the quantum circuit as defined in the list

        Returns
        -------
        qc : QuantumCircuit
            Qiskit quantum circuit.

        '''
        qc = QuantumCircuit(self.num_qubits)
        for layer in self:
            for gate in layer:
                qc.append(gate[0], gate[1])
        return qc