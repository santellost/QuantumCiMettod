# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:03:29 2024

@author: stefa
"""

def get_complete_base(num_qubits: int):
    return [format(x, f'0{num_qubits}b') for x in range(2**num_qubits)]
