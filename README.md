# QuantumCiMettod
# Membri:
Stefano Santello (team leader), Tommaso Fogarin, Manuel Martini
## Progetto:
Abbiamo implementato un algoritmo genetico tramite deap per evolvere circuiti quantistici basati sulla libreria qiskit.  
Nel file genetic.py è presente il codice dell'algoritmo genetico e della random walk come confronto.  
Il file individual.py contiene solo la class Individual, il file visualization.py contiene il codice per plottare i grafici e utils.py altre funzioni generiche.  
L'algoritmo genetico è quello semplice implementato da deap, noi abbiamo implementato l'individuo, le mutazioni e la fitness.  
Per il tuning dei parametri abbiamo scelto per ogni parametro dei valori che reputavamo sensati e ne abbiamo provato tutte le possibili combinazioni.  
Non abbiamo usato tecniche più avanzate perché avrebbe richiesto troppo tempo.  
## Per testare il codice:
Si può eseguire il file test.py che chiama in automatico la funzione simple_test.  
Alternativamente iterate_test ripete il test sullo stesso stato più volte ma stampa meno grafici.  
