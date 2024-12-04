# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:03:29 2024

@author: stefa
"""

import random
import numpy as np
import scipy as sc
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


def random_unitary(n: int):
    '''
    Generates a random unitary matrix

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    q : Matrix
        Unitary matrix.

    '''
    z = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
    q,r = sc.linalg.qr(z)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    q = np.multiply(q,ph,q)
    return q


def random_hermitian(n: int):
    '''
    Generates a random hermitian matrix

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    Matrix
        Unitary matrix.

    '''
    A = random_unitary(n)
    B = np.diag(np.random.randn(n))
    return A@B@A.conj().T


def random_hermitian_unitary(n: int):
    '''
    Generates a random hermitian and unitary matrix

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    Matrix
        Unitary matrix.

    '''
    A = random_unitary(n)
    B = np.diag([random.choice([1, -1]) for _ in range(n)])
    return A@B@A.conj().T
