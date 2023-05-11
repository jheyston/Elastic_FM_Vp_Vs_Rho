import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 # Use pd.read_pickle de la biblioteca de pds para leer los datos de pickle
dr = pd.read_pickle('descargaArchivosSGC.pkl')
trace = dr['Data']

print ('Leer datos \ n', trace.values[0])


plt.plot(trace.values[539])

plt.show()