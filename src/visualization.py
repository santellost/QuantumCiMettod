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
    
    
def plot_fitness(data: pd.DataFrame):
    g = sns.relplot(data=data, x='Generations', y='Fitness', hue='Legend', kind='line', aspect=1.3)
    g.set(title=f'Fitness on {len(pd.unique(data["Iteration"]))} iterations')
    plt.show()

    
def plot_logbook(*logbooks: tools.Logbook, **against: tools.Logbook | list[tools.Logbook]):
    data = pd.DataFrame()
    for i, logbook in enumerate(logbooks):
        gens = logbook.select('gen')
        avgs = pd.DataFrame({
            'Generations': gens, 
            'Legend': ['Average'] * len(gens), 
            'Iteration': i,
            'Fitness': logbook.select('avg')
            })
        mins = pd.DataFrame({
            'Generations': gens,
            'Legend': ['Minimum'] * len(gens),
            'Iteration': i,
            'Fitness': logbook.select('min')
            })
        data = pd.concat([data, avgs, mins])
    for label, logs in against.items():
        if isinstance(logs, tools.Logbook):
            gens = logs.select('gen')
            temp = pd.DataFrame({
                'Generations': gens,
                'Legend': [label] * len(gens),
                'Iteration': 0,
                'Fitness': logs.select('fitness')
                })
            data = pd.concat([data, temp])
        else:
            for log in logs:
                gens = log.select('gen')
                temp = pd.DataFrame({
                    'Generations': gens,
                    'Legend': [label] * len(gens),
                    'Iteration': 0,
                    'Fitness': log.select('fitness')
                    })
                data = pd.concat([data, temp])
    plot_fitness(data)
        
    
def plot_grid_search(data: pd.DataFrame):
    palette = sns.color_palette("crest", as_cmap=True)
    g = sns.relplot(data, x='Generations', y='Fitness', kind='line', aspect=1.3,
                    hue='Mutation probability', style='Crossing-over probability', 
                    row='Max depth', col='Tournament ratio', 
                    palette=palette, hue_norm=(0.0, 1.0))
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
