# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:02:45 2024

@author: stefa
"""

import utils
import random
from deap import creator, base
from qiskit import QuantumCircuit


try:
    del creator.FitnessMin
except Exception:
    pass

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

class Individual(list):
    def __init__(self, num_qubits: int, max_depth: int, iterable: iter):
        super().__init__(iterable)
        self.fitness = creator.FitnessMin()
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        
    
    def from_random_gates(num_qubits: int, max_depth: int):
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
        qc : Individual
            Random quantum circuit.

        '''
        qc = []
        for i in range(max_depth):            
            layer = []
            qubits = random.sample(range(num_qubits), random.randint(1, num_qubits))
            while len(qubits) > 0:
                gate = utils.random_gate(len(qubits))
                selected = qubits[:gate.num_qubits]
                layer.append((gate, selected))
                qubits = qubits[gate.num_qubits:]
            qc.append(layer)
        return Individual(num_qubits, max_depth, qc)
    
    
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