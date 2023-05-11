"""
Created on Tue Sep 18 08:57:45 2021
@author: Jheyston Serrano
"""
''''1. Importing libraries'''
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import URL_MAPPINGS
from obspy import UTCDateTime, Stream
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, figure, subplot, plot
from mpl_toolkits import mplot3d
from math import log
from shapely.geometry import Point
import geopandas
from obspy import read
from tqdm import tqdm



def findEventLatLong(startTime, endTime, minMagnitud=0, maxMagnitud=10, minLatitud=6, maxLatitud=8, minLongitud=-74.2, maxLongitud=-72.8, minDepth=0, maxDepth=700, service="IRIS"):
    webservice = Client(service)  # http://bdrsnc.sgc.gov.co/
    tempcat = webservice.get_events(starttime=startTime, endtime=endTime,
                                    minlatitude=minLatitud, maxlatitude=maxLatitud, minlongitude=minLongitud, maxlongitude=maxLongitud,
                                    minmagnitude=minMagnitud, maxmagnitude=maxMagnitud , mindepth=minDepth, maxdepth=maxDepth)
    return tempcat, webservice


''' Mapea las pusibles llaves de URL
    for key in sorted(URL_MAPPINGS.keys()):
     print("{0:<11} {1}".format(key,  URL_MAPPINGS[key]))
'''

def graphTrace():
    global listaStrems, listComboBoxStation

    opcTrace = comboBox2.get()
    for j in range(len(listaStrems)):
            dataLista = listaStrems[j].stats.network+"."+listaStrems[j].stats.station+"."+listaStrems[j].stats.channel + " "+listaStrems[j].stats.starttime.ctime()
            if dataLista == opcTrace:
                listaStrems[j].plot()



def upload_file():
    global f

    try:
        f = filedialog.askopenfilename()
        f_split = f.split('/')
        textFieldRutaArchivo.delete(0, "end")
        textFieldRutaArchivo.insert(0, ".../"+f_split[-1])

    except:
        print("!!Error de lectura en el archivo " + f)

def graphStationsEvents():
    return None
    ############################################################

    # plt.show()
    # ############################################################
def consultaSGC():
    # print('cal_ini:{} \t'.format(cal_ini.get_date()), end='')
    # print('cal_end:{}'.format(cal_end.get_date()), end='')
    # print('')
    global  listaStrems, f, data

    if textFieldMinMagnitud.get() == '':
        messagebox.showerror(message="No ingreso la magnitud minima", title="Error")
        return None
    if textFieldMaxMagnitud.get() == '':
        messagebox.showerror(message="No ingreso la magnitud maxima", title="Error")
        return None
    if textFieldTimeRecordBefore.get() == '':
        messagebox.showerror(message="No ingreso tiempo de grabacion antes", title="Error")
        return None
    if textFieldTimeRecordAfter.get() == '':
        messagebox.showerror(message="No ingreso tiempo de grabacion despues", title="Error")
        return None
    if textFieldMinDepth.get() == '':
        messagebox.showerror(message="No ingreso minima profundidad de busqueda", title="Error")
        return None
    if textFieldMaxDepth.get() == '':
        messagebox.showerror(message="No ingreso maxima profundidad de busqueda", title="Error")
        return None

    timeRecordBefore = 60 * int(textFieldTimeRecordBefore.get())  # en segundos
    timeRecordAfter = 60 * int(textFieldTimeRecordAfter.get())  # en segundos
    minMagnitud = float(textFieldMinMagnitud.get())  # 5.0
    maxMagnitud = float(textFieldMaxMagnitud.get())  # 5.8
    canalBusqueda = comboBox.get()
    minDepthBusqueda = float(textFieldMinDepth.get())
    maxDepthBusqueda = float(textFieldMaxDepth.get())

    minLatitudBusqueda = float(textFieldMinLatitud.get())
    maxLatitudBusqueda = float(textFieldMaxLatitud.get())
    minLongitudBusqueda = float(textFieldMinLongitud.get())
    maxLongitudBusqueda = float(textFieldMaxLongitud.get())

    print("\nInformacion ingresada:")
    print("minMagnitud:{}".format(minMagnitud))
    print("maxMagnitud:{}".format(maxMagnitud))
    print("timeRecordBefore:{}".format(timeRecordBefore))
    print("timeRecordAfter:{}".format(timeRecordAfter))
    print("canalBusqueda:{}".format(canalBusqueda))
    print("minDepthBusqueda:{}".format(minDepthBusqueda))
    print("maxDepthBusqueda:{}".format(maxDepthBusqueda))
    print("minLatitudBusqueda:{}".format(minLatitudBusqueda))
    print("maxLatitudBusqueda:{}".format(maxLatitudBusqueda))
    print("minLongitudBusqueda:{}".format(minLongitudBusqueda))
    print("maxLongitudBusqueda:{}\n".format(maxLongitudBusqueda))

    try:
        # Lectura archivo excel
        data = pd.read_excel(f)
        data2 = data.copy()

        if 'HORA_UTC' in data:
            data2["FECHA"] = data["FECHA"] + " " + data["HORA_UTC"]
        else:
            data2["FECHA"] = data["FECHA"]

        data2 = data[
            (data['MAGNITUD'] >= minMagnitud) &
            (data['MAGNITUD'] <= maxMagnitud) &
            (data['PROFUNDIDAD'] >= minDepthBusqueda) &
            (data['PROFUNDIDAD'] <= maxDepthBusqueda) &
            (data['LATITUD'] >= minLatitudBusqueda) &
            (data['LATITUD'] <= maxLatitudBusqueda) &
            (data['LONGITUD'] >= minLongitudBusqueda) &
            (data['LONGITUD'] <= maxLongitudBusqueda)]

        print(data)
        print(data2)


        zdata = data2['PROFUNDIDAD'].to_numpy()
        xdata = data2['LONGITUD'].to_numpy()
        magdata = data2['MAGNITUD'].to_numpy()
        ydata = data2['LATITUD'].to_numpy()

        fig = figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xdata, ydata, -zdata, s=magdata**4, c=zdata, cmap='viridis')

        ax.set_xlim3d(minLongitudBusqueda, maxLongitudBusqueda)
        ax.set_ylim3d(minLatitudBusqueda, maxLatitudBusqueda)

        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.set_zlabel('Depth (Km)')

        print(data2['LATITUD'].describe())
        print(data2['LONGITUD'].describe())
        print(data2['PROFUNDIDAD'].describe())
        print(data2['MAGNITUD'].describe())

        fig, ax = subplots()

        subplot(2, 2, 1), data2['LATITUD'].hist(), plt.title('Hist Latitud')
        subplot(2, 2, 2), data2['LONGITUD'].hist(), plt.title('Hist Longitud')
        subplot(2, 2, 3), data2['PROFUNDIDAD'].hist(), plt.title('Hist Depth')
        subplot(2, 2, 4), data2['MAGNITUD'].hist(), plt.title('Hist Mag')

        # To ShapeFile
        dfToShape = pd.DataFrame(list(zip(data2['LATITUD'], data2['LONGITUD'], data2['PROFUNDIDAD'], data2['MAGNITUD'])),
                                 columns=['Latitud', 'Longitud', 'Profundidad','Mag'])
        dfToShape['geometry'] = dfToShape.apply(lambda x: Point((float(x.Longitud), float(x.Latitud), float(x.Profundidad))), axis=1)
        dfToShape = geopandas.GeoDataFrame(dfToShape, crs="EPSG:4326", geometry='geometry')
        dfToShape.to_file('shp/Sismicidad.shp',  driver='ESRI Shapefile')



        #plt.show() # Descomentar para obtener dataFrame


    except:
        messagebox.showerror(message="No se encontraron eventos", title="Error")
        return None



    listNetwork = []
    listStation = []
    listLocation = []
    listStartTime = []
    listEndTime = []
    listChannel = []
    listSampling_rate = []
    listDelta = []
    listNpts = []
    listEvenTime = []
    listXEventdata = []
    listZEventdata = []
    listYEventdata = []
    listMagEventdata = []
    listLatitudStation = []
    listLongitudStation = []
    listData= []
    listaStrems = []

    # Filtrar estaciones
    dataEstaciones = pd.read_excel('Estaciones/estaciones.xlsx')
    if (canalBusqueda == "BH?" or canalBusqueda == "HH?"):
        df_mask = dataEstaciones[
            (dataEstaciones['Localizador'] == 0) & # El localizador esta definido en 0 pero podria ser 10 para un acelerometro
            (dataEstaciones['Latitud'] >= minLatitudBusqueda) &
            (dataEstaciones['Latitud'] <= maxLatitudBusqueda) &
            (dataEstaciones['Longitud'] >= minLongitudBusqueda) &
            (dataEstaciones['Longitud'] <= maxLongitudBusqueda) &
            ((dataEstaciones['Canal'] == canalBusqueda[:-1] + "E") |
            (dataEstaciones['Canal'] == canalBusqueda[:-1] + "N") |
            (dataEstaciones['Canal'] == canalBusqueda[:-1] + "Z"))]

    else :
        df_mask = dataEstaciones[
                (dataEstaciones['Localizador'] == 0) & # El localizador esta definido en 0 pero podria ser 10 para un acelerometro
                (dataEstaciones['Latitud'] >= minLatitudBusqueda) &
                (dataEstaciones['Latitud'] <= maxLatitudBusqueda) &
                (dataEstaciones['Longitud'] >= minLongitudBusqueda) &
                (dataEstaciones['Longitud'] <= maxLongitudBusqueda) &
                (dataEstaciones['Canal'] == canalBusqueda )]

    # print(df_mask['Estacion'])
    strEstaciones = ""
    setEstaciones = set()

    for strDfEsta in df_mask['Estacion']:
        strEstaciones += strDfEsta+","
        setEstaciones.add(strDfEsta)

    strEstaciones = strEstaciones[:-1]
    str_val = ','.join(list(map(str, setEstaciones)))
    print("Estaciones encontradas en la zona seleccionada:")
    print(str_val)
    print("found %s event(s):" % len(data2))
    # print(data['FECHA'])
    print(data2.columns.values.tolist())



    for i in tqdm(data2.index):
    # for i in tqdm(range(2)):

        # print(data2['FECHA'][i])
        strTime = data2['FECHA'][i].replace(" ", "T")
        eventTime = UTCDateTime(strTime)
        startTime = eventTime - timeRecordBefore
        endTime = eventTime + timeRecordAfter

        zdata = data2['PROFUNDIDAD'][i]
        xdata = data2['LONGITUD'][i]
        magdata = data2['MAGNITUD'][i]
        ydata = data2['LATITUD'][i]

        # print (strTime)
        # print (str(startTime)[:-1])
        # print (str(endTime)[:-1])



        # LLamado al URL de la base de datos del SGC
        URL = "http://sismo.sgc.gov.co:8080/fdsnws/dataselect/1/query?starttime="+str(startTime)[:-1]+"&endtime="+str(endTime)[:-1]+"&network=CM&sta="+str_val+"&cha="+canalBusqueda+"&loc=00&format=miniseed&nodata=404"
        # URL = "http://sismo.sgc.gov.co:8080/fdsnws/dataselect/1/query?starttime="+str(startTime)[:-1]+"&endtime="+str(endTime)[:-1]+"&network=CM&sta=*&cha=HHZ&loc=*&format=miniseed&nodata=404"

        try:
            st = read(URL)
            # print(str_val)
            # print(st.__str__(extended=True))

            for stream in st:
                # print(stream.stats.station)
                listNetwork.append(stream.stats.network)
                listStation.append(stream.stats.station)

                df2 = df_mask[df_mask['Estacion'] == str(stream.stats.station)]

                for lat in df2['Latitud']:
                    latitudStation = lat

                for lon in df2['Longitud']:
                     longitudStation = lon

                listLatitudStation.append(latitudStation)
                listLongitudStation.append(longitudStation)

                listLocation.append(stream.stats.location)

                listChannel.append(stream.stats.channel)
                listStartTime.append(stream.stats.starttime)
                listEndTime.append(stream.stats.endtime)
                listSampling_rate.append(stream.stats.sampling_rate)
                listDelta.append(stream.stats.delta)
                listNpts.append(stream.stats.npts)
                listEvenTime.append(eventTime)
                listZEventdata.append(zdata)
                listXEventdata.append(xdata)
                listYEventdata.append(ydata)
                listMagEventdata.append(magdata)
                listData.append(stream.data)
                listaStrems.append(stream)
                # print(listaStrems)
                # print(listData)
                # print(stream.data)





        except:
            # print(st)
            print ("\nRegistro no encontrado\n")






    ########################################

    points_df = pd.DataFrame(list(zip(listNetwork, listStation, listLatitudStation, listLongitudStation,
                                      listLocation, listChannel,
                                      listStartTime, listEndTime,
                                      listSampling_rate, listDelta, listNpts,
                                      listEvenTime, listZEventdata, listXEventdata, listYEventdata, listMagEventdata, listData)),
                                     columns=['Network', 'Station', 'LatStation', 'LongStation', 'Location', 'Channel', 'Starttime', 'Endtime', 'Samplingrate', 'Delta', 'Npts', 'Eventime', 'DepthEvent', 'LongEvent', 'LatEvent', 'MagEvent', 'Data'])

    print(points_df['LatStation'].describe())


    comboBox2['values'] = ""
    listComboBoxStation = []  #
    for j in range(len(listaStrems)):
            if listaStrems[j] != 0:
                listComboBoxStation.append(listaStrems[j].stats.network + "." + listaStrems[j].stats.station + "."
                                           + listaStrems[j].stats.channel + " "
                                           + listaStrems[j].stats.starttime.ctime())
    comboBox2['values'] = listComboBoxStation

    print(points_df['LatEvent'].describe()) # Estadistica
    print(points_df['LongEvent'].describe())# Estadistica
    print(points_df['DepthEvent'].describe())# Estadistica
    print(points_df['MagEvent'].describe())# Estadistica

    plt.subplot(2, 2, 1), points_df['LatEvent'].hist(), plt.title('Hist Latitud'), plt.xlabel('Latitud'), plt.ylabel('Frecuencia')
    plt.subplot(2, 2, 2), points_df['LongEvent'].hist(), plt.title('Hist Longitud'), plt.xlabel('Longitud'), plt.ylabel('Frecuencia')
    plt.subplot(2, 2, 3), points_df['DepthEvent'].hist(), plt.title('Hist Depth'), plt.xlabel('Profundidad'), plt.ylabel('Frecuencia')
    plt.subplot(2, 2, 4), points_df['MagEvent'].hist(), plt.title('Hist Mag'), plt.xlabel('Magnitud'), plt.ylabel('Frecuencia')



    print(points_df)
    print("Se finalizo la consulta")
    points_df.to_pickle("descargaArchivosSGC.pkl")
        # return None
        # print(st[0].stats)

    '''Estaciones en Colombia'''
    # print("Estaciones en Colombia")
    # print (client.get_stations(network="CM"))
def desactivador(Event=None):

    if comboBox3.get() == "(Min-Max)->LatLong":
        textFieldMaxLongitud["state"] = 'normal'
        textFieldMaxLatitud["state"] = 'normal'
        textFieldMinLongitud["state"] = 'normal'
        textFieldMinLatitud["state"] = 'normal'

    else:
        textFieldMaxLongitud["state"] = 'disabled'
        textFieldMaxLatitud["state"] = 'disabled'
        textFieldMinLongitud["state"] = 'disabled'
        textFieldMinLatitud["state"] = 'disabled'

if __name__ == '__main__':
    ventana = tk.Tk()
    ventana.geometry("800x600")  # Tamanio de la ventana
    ventana.title("Automatic SGC donwload files...")  # titulo de la ventana
    ventana.eval('tk::PlaceWindow . center')  # Ubicacion de la ventana
    ventana.resizable(False, False)  # No permitir modificiar el tamanio
    my_font1 = ('times', 18, 'bold')
    my_font2 = ('times', 12, 'bold')

    Label(ventana, text="Carga excel:", font=my_font1).grid(pady=5, row=0, column=0, columnspan=2)
    Label(ventana, text="Archivo sismos:", font=my_font2).grid(padx=1,  pady=5, row=1, column=0)
    textFieldRutaArchivo = tk.Entry(ventana, width=40)
    textFieldRutaArchivo.grid(padx=5, row=1, column=2, columnspan=2, sticky="w")
    textFieldRutaArchivo.insert(0, "")
    Label(ventana, text="Min Magnitud:", font=my_font2).grid(padx=20, pady=1, row=4, column=0, sticky="w")
    Label(ventana, text="Max Magnitud:", font=my_font2).grid(padx=20, pady=1, row=4, column=2, sticky="w")
    Label(ventana, text="Busqueda por:", font=my_font2).grid(padx=20, pady=1, row=5, column=0, sticky="w")


    Label(ventana, text="MinLatitud:", font=my_font2).grid(padx=20, pady=5, row=8, column=0, sticky="w")
    Label(ventana, text="MinLongitud:", font=my_font2).grid(padx=20, pady=5, row=8, column=2, sticky="w")
    Label(ventana, text="MaxLatitud:", font=my_font2).grid(padx=20, pady=5, row=9, column=0, sticky="w")
    Label(ventana, text="MaxLongitud:", font=my_font2).grid(padx=20, pady=5, row=9, column=2, sticky="w")
    Label(ventana, text="Min Depth(Km):", font=my_font2).grid(padx=20, pady=5, row=10, column=0, sticky="w")
    Label(ventana, text="Max Depth(Km):", font=my_font2).grid(padx=20, pady=5, row=10, column=2, sticky="w")
    Label(ventana, text="Tiempo Antes(min):", font=my_font2).grid(padx=20, pady=5, row=11, column=0, sticky="w")
    Label(ventana, text="Tiempo Despues (min):", font=my_font2).grid(padx=20, pady=5, row=11, column=2, sticky="w")

    Label(ventana, text="Canal:", font=my_font2).grid(padx=20, pady=5, row=12, column=0, sticky="w")
    Label(ventana, text="Registros:", font=my_font2).grid(padx=20, pady=5, row=13, column=0, sticky="w")

    # Entradas de texto
    textFieldMinMagnitud = tk.Entry(ventana, width=20)
    textFieldMinMagnitud.grid(padx=5, row=4, column=1, sticky="w")
    textFieldMinMagnitud.insert(0, "1.0")

    textFieldMaxMagnitud = tk.Entry(ventana, width=20)
    textFieldMaxMagnitud.grid(padx=5, row=4, column=3, sticky="w")
    textFieldMaxMagnitud.insert(0, "9.0")



    textFieldMinLatitud = tk.Entry(ventana, width=20)
    textFieldMinLatitud.grid(padx=5, row=8, column=1, sticky="w")
    textFieldMinLatitud.insert(0, "6.72")
    textFieldMinLatitud["state"] = 'normal'

    textFieldMinLongitud = tk.Entry(ventana, width=20)
    textFieldMinLongitud.grid(padx=5, row=8, column=3, sticky="w")
    textFieldMinLongitud.insert(0, "-73.92")
    textFieldMinLongitud["state"] = 'normal'

    textFieldMaxLatitud = tk.Entry(ventana, width=20)
    textFieldMaxLatitud.grid(padx=5, row=9, column=1, sticky="w")
    textFieldMaxLatitud.insert(0, "7.42")
    textFieldMaxLatitud["state"] = 'normal'

    textFieldMaxLongitud = tk.Entry(ventana, width=20)
    textFieldMaxLongitud.grid(padx=5, row=9, column=3, sticky="w")
    textFieldMaxLongitud.insert(0, "-72.94")
    textFieldMaxLongitud["state"] = 'normal'


    textFieldMinDepth = tk.Entry(ventana, width=20)
    textFieldMinDepth.grid(padx=5, row=10, column=1, sticky="w")
    textFieldMinDepth.insert(0, "0")

    textFieldMaxDepth = tk.Entry(ventana, width=20)
    textFieldMaxDepth.grid(padx=5, row=10, column=3, sticky="w")
    textFieldMaxDepth.insert(0, "700") # en Km


    textFieldTimeRecordBefore = tk.Entry(ventana, width=20)
    textFieldTimeRecordBefore.grid(padx=5, row=11, column=1, sticky="w")
    textFieldTimeRecordBefore.insert(0, "1")

    textFieldTimeRecordAfter = tk.Entry(ventana, width=20)
    textFieldTimeRecordAfter.grid(padx=5, row=11, column=3, sticky="w")
    textFieldTimeRecordAfter.insert(0, "10")



    '''ComboBox'''
    comboBox = Combobox(ventana)
    comboBox.grid(padx=5, pady=10, row=12, column=1, sticky="w")
    comboBox["values"] = ["HH?", "HHZ", "BH?", "BHZ"]
    comboBox.current(0)

    comboBox2 = Combobox(ventana, width="35")
    comboBox2.grid(padx=5, pady=10, row=13, column=1, sticky="w", columnspan=2)
    comboBox2["values"] = [""]
    comboBox2.current(0)

    comboBox3 = Combobox(ventana)
    comboBox3.grid(padx=5, pady=10, row=5, column=1, sticky="w", columnspan=1)
    comboBox3["values"] = ["(Min-Max)->LatLong"]
    comboBox3.current(0)
    comboBox3.bind("<<ComboboxSelected>>", desactivador)

    ''' Declaracion de objetos calendario'''
    BotonLecturaExcel = Button(ventana, text="Archivo", width=20, command=upload_file)
    BotonLecturaExcel.grid(padx=1, pady=5, row=1, column=1)

    ''' Botones '''
    # Boton para enviar la consulta
    BotonConsulta = Button(ventana, text="Consultar", width=10, command=consultaSGC)
    BotonConsulta.grid(padx=20, pady=5, row=13, column=3)
    # Boton para graficar eventos y estaciones
    #BotonMapa = Button(ventana, text="Mapa", width=10, command=graphStationsEvents)
    # BotonMapa.grid(padx=20, pady=5, row=14, column=3)
    # Boton para graficar traza
    BotonTraza = Button(ventana, text="Graficar", width=10, command=graphTrace)
    BotonTraza.grid(padx=20, pady=5, row=15, column=3)
    #

    # date.pack(pady=20)
    # cal.pack(pady=20)
    ventana.mainloop()
