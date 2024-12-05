# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:58:34 2024

@author: stefa
"""

import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from deap import tools
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


sns.set_theme()
sns.set_style("whitegrid")


def plot_logbook(logbook: tools.Logbook, **against: tools.Logbook):
    '''
    Plots the fitness inside the logbook againsts multiplt methods, usually a random walk

    Parameters
    ----------
    logbook : tools.Logbook
        Main logbook to plot.
    **against : tools.Logbook
        Additional logbooks to plot against the main.

    Returns
    -------
    None.

    '''
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    fig, ax1 = plt.subplots(layout='tight', dpi=300)
    line1 = ax1.plot(gen, fit_mins, "-", label="Minimum Fitness")
    line2 = ax1.plot(gen, fit_avgs, "-", label="Average Fitness")
    
    lines = []
    for label, log in against.items():
        a_gen = log.select('gen')
        a_fit = log.select('fitness')
        lines += ax1.plot(a_gen, a_fit, "-", label=label)
        
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    
    lns = line1 + line2 + lines
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')
    
    plt.show()
    
def compare_histograms(qc: QuantumCircuit, desired: Statevector):
    '''
    Plots both histograms obtained by the qunatum circuit and the desired statevector

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit from which the evolved statevector will be compared.
    desired : Statevector
        The desired statevector.
    shots : int, optional
        Number of measurement made for both histograms. The default is 100.

    Returns
    -------
    None.

    '''
    base_labels = utils.get_complete_base(qc.num_qubits)
    initial = Statevector.from_label(base_labels[0])
    evolved = initial.evolve(qc)
    
    ev_counts = evolved.probabilities_dict()
    ev_counts = [ev_counts.get(base, 0) for base in base_labels]
    ev_data = pd.DataFrame({'base': base_labels, 'from': ['Evolved'] * 2**qc.num_qubits, 'frequencies': ev_counts})
        
    de_counts = desired.probabilities_dict()
    de_counts = [de_counts.get(base, 0) for base in base_labels]
    de_data = pd.DataFrame({'base': base_labels, 'from': ['Desired'] * 2**qc.num_qubits, 'frequencies': de_counts})

    with sns.plotting_context('notebook', font_scale=1.2):
        data = pd.concat([ev_data, de_data])
        g = sns.catplot(data, x='base', y='frequencies', row='from', kind='bar', height=3, aspect=0.015*4**qc.num_qubits/2 + 2)
        g.set_axis_labels("Base", "Frequencies")
        g.set_titles('{row_name}')
        g.set(ylim=(0,1))
        
    plt.show()
