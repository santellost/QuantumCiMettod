# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:03:29 2024

@author: stefa
"""

import random
import numpy as np
import pandas as pd
from qiskit.circuit import library, Gate


gates_map = library.get_standard_gate_name_mapping()
gates_map = dict(filter(lambda gate: not gate[0] in ['id', 'measure', 'delay', 'reset', 'global_phase'], gates_map.items()))


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
    gate = random.choice(list(filter(lambda g: g.num_qubits <= max_qubits, gates_map.values()))).copy()
    for index in range(len(gate.params)):
        # Random fixed Parameters from the interval [0, 2*pi]
        gate.params[index] = random.uniform(0, 2*np.pi)
    return gate


def get_complete_base(num_qubits: int):
    '''
    Generates all possible labels for a complete base

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Returns
    -------
    list
        list of all labels.

    '''
    return [format(x, f'0{num_qubits}b') for x in range(2**num_qubits)]


def update_file(filename: str, data: pd.DataFrame) -> pd.DataFrame:
    '''
    Appends data to filename and returns the updated data in the file

    Parameters
    ----------
    filename : str
        Path to the file.
    data : pd.DataFrame
        If the files already exist, be sure that the headers are the same.

    Returns
    -------
    data : pd.DataFrame
        The complete data after the update.

    '''
    try:
        with open(filename, 'x') as file:
            data.to_csv(file, index=False)
    except FileExistsError: 
        with open(filename, 'a') as file:
            data.to_csv(file, index=False, header=False)
    with open(filename, 'r') as file:
        data = pd.read_csv(file)
    return data