# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:03:29 2024

@author: stefa
"""

import random
from qiskit.circuit import library, Gate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate


NO_PARAMS = [g() for g in library.__dict__.values() if isinstance(g, type) and (issubclass(g, SingletonGate) or issubclass(g, SingletonControlledGate))]
# TODO find a method to use parameter gates


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


def get_complete_base(num_qubits: int):
    return [format(x, f'0{num_qubits}b') for x in range(2**num_qubits)]
