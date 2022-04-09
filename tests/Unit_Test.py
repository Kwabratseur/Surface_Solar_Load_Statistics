import datetime
import csv
import surface_solar_load_statistics.Get_Radiation as R
import matplotlib.pyplot as plt #pip install matplotlib
import numpy as np # pip install numpy


Surfaces = [[35,170],[90,170],[35,-10],[90,-10]]# wall and roof angles for stinsburg
c = 0
Fieldnames = []
for i in Surfaces:
    Fieldnames.append("Date_%d_%d_%d"%(i[0],i[1],c))
    Fieldnames.append("Itot_%d_%d_%d"%(i[0],i[1],c))
    Fieldnames.append("Ib_%d_%d_%d"%(i[0],i[1],c))
    Fieldnames.append("Iid_%d_%d_%d"%(i[0],i[1],c))
    Fieldnames.append("Icd_%d_%d_%d"%(i[0],i[1],c))
    Fieldnames.append("Ihd_%d_%d_%d"%(i[0],i[1],c))
    Fieldnames.append("Igr_%d_%d_%d"%(i[0],i[1],c))
    Fieldnames.append("theta_%d_%d_%d"%(i[0],i[1],c))
    c += 1
print(Fieldnames)
print("Amount of fields for surfaces: {}".format(len(Fieldnames)))

#sample = R.CalcRad("2016-1-1","2017-11-6",6.9,52.2,900)
startdate = "2016-1-1"
stopdate = "2017-11-6"
lat = 6.9 # stinsburg lat & lon
lon = 52.2
dT = 900 # 15 minutes
Ktfile = "hourkt_Twenthe_hourly.csv"
statsfile = "Stats_Twenthe_hourly.csv"

Rhog = None # ground reflection coefficient. is location specific, needs 12 values, 1 per month.
TimeFormat = "%Y-%m-%d_%H:%M:%S"
plot=True # automatic plotting of results

sample = R.CalcRad(startdate,stopdate,lat,lon,dT,Ktfile,statsfile,Surfaces,Rhog,TimeFormat,plot,Fieldnames)
