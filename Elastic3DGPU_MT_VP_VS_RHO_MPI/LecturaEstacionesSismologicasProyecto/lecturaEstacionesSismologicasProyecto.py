

"""
Created on Mon Nov 21 17:00:00 2022
@author: Jheyston Serrano
Email: jheyston.serrano@e3t.uis.edu.co
"""


import pandas as pd
import numpy as np
from numpy import load, save
from math import pi, radians
from matplotlib.pyplot import subplots, figure, subplot, plot
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


latitud_1 = '''6° 15' 45.650" N 
6° 27' 35.277" N 
6° 34' 42.660" N
6° 50' 53.630" N
7° 8' 32.390" N
7° 13' 57.010" N
7° 15' 0.900" N
7° 26' 29.240" N
6° 35' 31.200" N
6° 46' 1.200" N
6° 58' 22.800" N
6° 59' 31.200" N
7° 1' 4.800" N
7° 4' 15.600" N
7° 7' 37.200" N
7° 8' 31.200" N
7° 20' 24.000" N
7° 21' 36.000" N
7° 21' 50.400" N
7° 22' 44.400" N
7° 26' 16.800" N
6° 10' 0.667" N
6° 14' 13.218" N
6° 21' 29.926" N
6° 22' 47.024" N
6° 25' 44.591" N
6° 35' 48.255" N
6° 38' 50.627" N
6° 45' 9.792" N
6° 53' 9.500" N
6° 53' 48.786" N
6° 54' 7.992" N
6° 54' 16.919" N
6° 56' 15.958" N
7° 1' 3.884" N
7° 3' 12.513" N
7° 4' 0.346" N
7° 4' 0.778" N
7° 5' 44.674" N
7° 8' 26.462" N
7° 10' 51.949" N
7° 18' 40.011" N
7° 25' 52.976" N'''

longitud_1 = '''73° 8' 45.540" W
72° 35' 49.621" W
73° 32' 47.270" W
73° 3' 40.840" W
73° 6' 58.760" W
73° 30' 52.430" W
72° 36' 30.870" W
73° 8' 48.710" W
73° 11' 2.400" W
73° 43' 30.000" W
73° 44' 38.400" W
73° 3' 50.400" W
73° 8' 2.400" W
73° 4' 22.800" W
73° 17' 24.000" W
73° 7' 8.400" W
72° 42' 0.000" W
73° 51' 7.200" W
72° 39' 54.000" W
72° 38' 13.200" W
73° 34' 15.600" W
73° 31' 28.318" W
72° 44' 29.814" W
73° 16' 47.848" W
73° 48' 23.390" W
72° 52' 17.209" W
73° 3' 53.067" W
73° 56' 56.066" W
72° 46' 30.742" W
73° 24' 24.607" W
73° 34' 31.292" W
72° 41' 4.524" W
73° 17' 16.475" W
73° 41' 44.488" W
73° 48' 38.620" W
72° 39' 40.062" W
73° 33' 39.522" W
73° 33' 38.976" W
73° 20' 20.700" W
73° 38' 55.075" W
73° 25' 38.557" W
72° 52' 13.856" W
73° 59' 48.026" W
'''

if __name__ == "__main__":
    latitud_1 = latitud_1.split("\n")
    longitud_1 = longitud_1.split("\n")
    latitud = []
    longitud = []
    refLat = 6
    refLon = -74
    EarthD = 12756.320
    dx, dy, dz = 1e3, 1e3, 1e3

    for i in range(len(latitud_1)):
        tempLat = latitud_1[i].split("N")[0].replace('\n', '').replace(' ', '').replace('\'', 'x').replace('°', 'x').replace('\"', 'x').split("x")[:-1]
        tempLon = longitud_1[i].split("W")[0].replace('\n', '').replace(' ', '').replace('\'', 'x').replace('°', 'x').replace('\"', 'x').split("x")[:-1]
        horasLat = float(tempLat[0])
        minLat = float(tempLat[1])
        segLat = float(tempLat[2])
        horasLon = float(tempLon[0])
        minLon = float(tempLon[1])
        segLon = float(tempLon[2])

        latitud.append((float(horasLat)+float(minLat)/60+float(segLat)/3600))
        longitud.append((-float(horasLon) - float(minLon) / 60 - float(segLon) / 3600))


    latitud = np.array(latitud)
    longitud = np.array(longitud)
    rec_pos_x = []
    rec_pos_y = []
    rec_pos_z = []

    for i in range(len(latitud)):
        # print(latitud[i], longitud[i])


        rec_pos_x.append( (((longitud[i] - refLon) * EarthD * pi * np.cos(radians(refLat)) / 360) / (dx / 1e3)).astype(int))
        rec_pos_y.append( (((latitud[i] - refLat) * EarthD * pi / 360) / (dx / 1e3)).astype(int))
        rec_pos_z.append(1.0)


    for i in range(len(rec_pos_x)):
        print("x:{:^5} y:{:^5} z:{:^5}".format(rec_pos_x[i], rec_pos_y[i], rec_pos_z[i]))


rec_pos_x = np.array(rec_pos_x)
rec_pos_y = np.array(rec_pos_y)
rec_pos_z = np.array(rec_pos_z)

save('npyFiles/rec_pos_x.npy', rec_pos_x)
save('npyFiles/rec_pos_y.npy', rec_pos_y)
save('npyFiles/rec_pos_z.npy', rec_pos_z)

fig = figure()
ax = plt.axes(projection='3d')
ax.scatter3D(rec_pos_x[:], rec_pos_y[:], -rec_pos_z[:],color='g', label='stations')
ax.legend(loc="upper right")
minLongitudBusqueda= np.min(rec_pos_x)
maxLongitudBusqueda= np.max(rec_pos_x)

minLatitudBusqueda= np.min(rec_pos_y)
maxLatitudBusqueda= np.max(rec_pos_y)

ax.set_xlim3d(minLongitudBusqueda, maxLongitudBusqueda)
ax.set_ylim3d(minLatitudBusqueda, maxLatitudBusqueda)
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.set_zlabel('Depth (Km)')

plt.show()


