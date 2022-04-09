#!/bin/sh

#   Copyright  Jeroen van 't Ende 09/03/2020
#   This work is based On Pysolar V0.6 module, some simple modifications have been made.
#   https://github.com/pingswept/pysolar
#   docs: https://pysolar.readthedocs.io/en/latest/#


#    Copyright Brandon Stafford
#
#    Pysolar is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 3 of the License, or
#    (at your option) any later version.
#
#    Pysolar is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with Pysolar. If not, see <http://www.gnu.org/licenses/>.


from pysolar import *
import os
import datetime
from datetime import timezone
import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


#   for datetime timeformatting see bottom of page:
#   https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior

#  Duurzame Energietechniek (ISBN: 978 90 395 2789 4) p.18 refers to
#  Solar Engineering of thermal processes (ISBN: 978-1-118-67160-3)
#  The latter describes methods for calculating direct and diffuse components of solar radiation
#  P.37 shows a less accurate method for calculating G0 (global radiation)(method implemented here is more accurate)
#  and explains that the integrated extraterrestial radiation is needed.

#  p.71 explains that KT ,Kt and kt are calculated from integrated extraterrestial radiation and measurements.
#  Note that values mentioned above are distributions of monthly, daily and hourly clearness
#  p.74 shows how the the ratio Id/I (fraction of diffuse light on horizontal plane) can be calculated from kt.
#  The advised correlation, Erbs et al. correlation is:
#
#     {1.0 -0.09kt                                         | for        kt <= 0.22}
#Id/i={0.9511 -0.1604kt +4.388kt^2 -16.638kt^3 +12.336kt^4 | for 0.22 < kt <= 0.80}
#     {0.165                                               | for        kt > 0.80}
#


BC_AS = [[1.000,-0.008,0.588,-0.062,-0.06,0.072,-0.022],# brightness coefficients for anisotropic sky for perez model.
         [1.065,0.13,0.683,-0.151,-0.019,0.066,-0.029],
         [1.230,0.330,0.487,-0.221,0.055,-0.064,-0.026],
         [1.500,0.568,0.187,-0.295,0.109,-0.152,0.014],
         [1.950,0.873,-0.392,-0.362,0.226,-0.462,0.001],
         [2.800,1.132,-1.237,-0.412,0.288,-0.823,0.056],
         [4.500,1.060,-1.600,-0.359,0.264,-1.127,0.131],
         [6.200,0.678,-0.327,-0.250,0.156,-1.1377,0.251]]

def GetDayOfYear(Date):
    day = Date.day
    month = Date.month
    Days = [31,28,31,30,31,30,31,31,30,31,30,31]
    for i in range(month-1):
        day += Days[i]
    return day

def GetBrightnessCoefficients(epsilon):
    for i in range(len(BC_AS),0,-1):
        if epsilon >= BC_AS[i-1][0]:
            return BC_AS[i-1]

def Epsilon(Id,Ibn,thetaz):
    const = 5.535*10**-6
    const = const*math.pow(thetaz, 3)
    if Id == 0:
        P1 = 0
    else:
        P1 = (Id+Ibn)/Id
    return (P1+const)/const

def applyf(f1,f2,f3,delta,thetaz): # combined with GetBrightnessCoefficients and Epsilon this will return F1 and F2
    return f1 + f2*delta + ((math.pi*thetaz)/180)*f3

def Beam_component_integrated(delta,phi,beta,gamma,omega1,omega2):
    delta = math.radians(delta)
    phi = math.radians(phi)
    beta = math.radians(beta)
    gamma = math.radians(gamma)
    omega1 = math.radians(omega1)
    omega2 = math.radians(omega2)
    a = (math.sin(delta)*math.sin(phi)*math.cos(beta) - math.sin(delta)*math.cos(phi)*math.sin(beta)*math.cos(gamma))*(1/180)*(math.degrees(omega2)-math.degrees(omega1))*math.pi\
    +(math.cos(delta)*math.cos(phi)*math.cos(beta) + math.cos(delta)*math.sin(phi)*math.sin(beta)*math.cos(gamma))*(math.sin(omega2)-math.sin(omega1))\
    -(math.cos(delta)*math.sin(beta)*math.sin(gamma))*(math.cos(omega2)-math.cos(omega1))
    b=(math.cos(phi)*math.cos(delta))*(math.sin(omega2)-math.sin(omega1)) + (math.sin(phi)*math.sin(delta))*(1/180)*(math.degrees(omega2)-math.degrees(omega1))*math.pi
    return a/b#math.degrees(a)/math.degrees(b)

def InclanatedRadiationComponents(slope,slope_orientation,Dir,Dif,lat,lon,alt,azi,date,Rhog = None):
    if Rhog == None:
        Rhog = [0.7,0.7,0.4,0.2,0.2,0.2,0.2,0.2,0.2,0.4,0.7,0.7]
    DOY = int(GetDayOfYear(date+datetime.timedelta(minutes=30))) #midpoint of hour is needed.
    Ion = radiation.get_apparent_extraterrestrial_flux(DOY) # extraterestial radiation
    thetaz = 90-abs(alt) # complement of azimuth
    epsilon,f11,f12,f13,f21,f22,f23 = GetBrightnessCoefficients(Epsilon(Dif,Dir,thetaz))
    m = radiation.get_air_mass_ratio(alt)
    Delta = m*(Dif/Ion) # delta term for 2.16.11
    theta = math.degrees(math.acos(math.cos(math.radians(thetaz))*math.cos(math.radians(slope_orientation))+math.sin(math.radians(thetaz))*math.sin(math.radians(slope_orientation))*math.cos(math.radians(azi-slope_orientation))))
    # beam incidence angle on tilted surface, tilted with [slope] from horizontal plane and [slope orientation] from meridian
    a = math.cos(math.radians(theta)) # a term for 2.16.9
    b = math.cos(math.radians(thetaz))# b term for 2.16.9
    #delta = GetDeclination(DOY)
    #w1 = GetHourAngle(date, lon)
    #w2 = GetHourAngle((date + datetime.timedelta(hours = 1)),lon)
    if 0 > a:
        a = 0
    if math.cos(math.radians(85)) > b:
        b = math.cos(math.radians(85))
    #Rb = Beam_component_integrated(delta, lat, slope, slope_orientation, w1, w2) # apperently not much better then the simple way of calculating Rb
    Rb = math.cos(math.radians(theta))/math.cos(math.radians(thetaz))  # Gbt/gb; 1.8.1
    F1 = applyf(f11,f12,f13,Delta,thetaz)  # 2.16.12
    F2 = applyf(f21,f22,f23,Delta,thetaz)  # 2.16.13
    if 0 > F1:
        F1 = 0
    Ib, Iid, Icd, Ihd, Igr = GetIncidentRadiation(Dir,Dif,(Dir+Dif),Rb,Rhog[(date.month-1)],slope,F1,F2,a,b)
    if Ib > 1160: # main area of improvement!
        Ib = 1160
        #print "{} , incident:{} , deltafactor:{}, declination:{} , RB:{} , Ib:{}".format(date,theta,Delta,delta,Rb,Ib)
    return Ib, Iid, Icd, Ihd, Igr, theta

def GetIncidentRadiation(Ib,Id,I,Rb,Rhog,Beta,F1,F2,a,b):# literally 2.16.14 but split in 5 terms and calculations.
    beam_component = Ib*Rb
    isotropic_diffuse_component = Id*(1-F1)*((1+math.cos(math.radians(Beta)))/2)
    circumsolar_diffuse_component = Id*F1*(a/b)
    horizon_diffuse_component = Id*F2*math.sin(math.radians(Beta))
    ground_reflection_component = I*Rhog*((1-math.sin(math.radians(Beta)))/2)
    if beam_component < 0:
        beam_component = 0
    if isotropic_diffuse_component < 0:
        isotropic_diffuse_component = 0
    if circumsolar_diffuse_component < 0:
        circumsolar_diffuse_component = 0
    if horizon_diffuse_component < 0:
        horizon_diffuse_component = 0
    if ground_reflection_component < 0:
        ground_reflection_component = 0
    return beam_component, isotropic_diffuse_component, circumsolar_diffuse_component, horizon_diffuse_component, ground_reflection_component

def TND(mu,sigma,LB,UB,N):# wrapper for truncated normal distributions
    if sigma <= 0.0:
        sigma = 1e-16
    if mu <= 0.0:
        mu = 1e-16
    if LB <= 0.0:
        LB = 1e-16
    if UB > 1367 or UB <= 0:
        UB = 1367
    if LB == UB:
        UB = 2*LB
    Set = stats.truncnorm.rvs((LB-mu)/sigma,(UB-mu)/sigma,loc=mu,scale=sigma,size=int(N))
    return Set

#pysolar.tzinfo_check.NoTimeZoneInfoError: datetime value '2016-01-01 00:00:00' given for arg 'when' should be made timezone-aware.
#You have to specify the 'tzinfo' attribute of 'datetime.datetime' objects.


def GetRad(lat,lon,date,kt=1.0,Sf=1.0):#takes latitude, longitude, initialized datetime object.
    #alt = GetAltitude(lat,lon,date) # these are both deprecated and replaced by get_position
    #azi = GetAzimuth(lat,lon,date)
    # Another change is the timezone aware handling of problems. This needs to be addressed
    azi, alt = solar.get_position(lat, lon, date)
    rad = radiation.get_radiation_direct(date,alt)
    rad = rad*Sf #correct with statistical factor
    Diffuse = rad*IddivI(kt) # kt = hourly radiation [J/m2]/hourly extraterestial radiation [J/m2] (both local)
    Direct = rad - Diffuse
    if rad < 0:
        rad = 0
    if Direct < 0:
        Direct = 0
    if Diffuse < 0:
        Diffuse = 0
    return alt, azi, rad, Direct, Diffuse

def IddivI(kt): # Id/I as function of kt, Erbs et al.
    kt = float(kt)
    if kt != 0:
        if kt <= 0.22:
            return 1.0 - (0.09*kt)
        elif kt > 0.22 and kt <= 0.8:
            return 0.9511 - (0.1604*kt) + (4.388*kt**2) - (16.638*kt**3) + (12.336*kt**4)
        else:
            return 0.165
    else:
        return 0

def GetHourOfYear(utc_datetime):
    year_start = datetime.datetime(utc_datetime.year, 1, 1, tzinfo=timezone.utc)
    utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
    delta = (utc_datetime - year_start)
    dH = delta.days*24
    return float(dH + float(delta.seconds/3600))

def GetStats(hourfile="Stats_DeBilt_1960-2016.csv"):
    hour = []
    with open(hourfile,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hour.append([float(row["mu"]),float(row["sigma"]),float(row["median"]),float(row["min"]),float(row["max"]),float(row["variance"])])
    return hour

def GetKx(hourfile="hourkt_DeBilt_1960-2016.csv"):
    hour = []
    with open(hourfile,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hour.append(row["kt"])
    return hour

def CalcStats(Array,verbose = False):
    if len(Array) < 1:
        return 0,0,0,0,0,0
    else:
        stdev = np.std(Array)
        mean = np.mean(Array)
        median = np.median(Array)
        variance = np.var(Array)
        Min = np.min(Array)
        Max = np.max(Array)
        if verbose:
            NormDist = TND(stdev,mean,Min,Max,len(Array))
            #NormDist = np.random.normal(mean,stdev,len(Array))
            stdevErr = abs(stdev-np.std(NormDist))
            meanErr = abs(mean-np.mean(NormDist))
            medianErr = abs(median-np.median(NormDist))
            varianceErr = abs(variance-np.var(NormDist))
            print( "standard deviation: {}, mean: {}, median: {}, variance: {}, min: {}, max: {}".format(stdev,mean,median,variance,Min,Max))
            print( "stdevErr: {}, meanErr: {}, medianErr: {}, varianceErr: {}".format(stdevErr,meanErr,medianErr,varianceErr))
               #mu,sig,med,var,Min,Max
        return stdev,mean,median,variance, Min, Max

#InclanatedRadiationComponents(slope,slope_orientation,Dir,Dif,lat,lon,alt,azi,date,Rhog = None)
#return Ib, Iid, Icd, Ihd, Igr, theta       |        beam, incident_diffuse,

def CalcRad(startDate,stopDate,latitude,longitude,TimeStep = 3600,KxFile="hourkt_DeBilt_1960-2016.csv",StatFile="Stats_DeBilt_1960-2016.csv",slopes = None, Rhog = None, TimeFormat = "%Y-%m-%d_%H:%M:%S",plot=False,fieldnames=["Date","Globaln","Directn","Diffusen","Altitude",'Azimuth']):
    hourkt=GetKx(KxFile)
    hourstats = GetStats(StatFile)
    TimeStep = int(TimeStep)
    latitude = float(latitude)
    longitude = float(longitude)
    start = datetime.datetime.strptime(startDate+"-UTC", "%Y-%m-%d-%Z").replace(tzinfo=datetime.timezone.utc)
    stop = datetime.datetime.strptime(stopDate+"-UTC", "%Y-%m-%d-%Z").replace(tzinfo=datetime.timezone.utc)
    Data = []
    DataSlopes = []
    c = 0
    hour = int(GetHourOfYear(start))
    #RandomDistribution = np.random.normal(hourstats[hour][1],hourstats[hour][0],(3600/TimeStep))
    RandomDistribution =  TND(hourstats[hour][0],hourstats[hour][1],hourstats[hour][2],hourstats[hour][3],(3600/TimeStep)) #mu,sig,med,Min,Max
    while start < stop:
        if c >=(3600/TimeStep):
            #RandomDistribution = np.random.normal(hourstats[hour][1],hourstats[hour][0],(3600/TimeStep))
            RandomDistribution =  TND(hourstats[hour][0],hourstats[hour][1],hourstats[hour][3],hourstats[hour][4],(3600/TimeStep))
            c = 0
        alt,azi,rad,Dir,Dif = GetRad(latitude, longitude, start,hourkt[hour],RandomDistribution[c])
        if math.isnan(Dir):
            Dir = 0
        if math.isnan(Dif):
            Dif = 0
        if math.isnan(rad):
            rad = 0
        azi = (azi*-1)-180
        if slopes is not None:
            temp = []
            for i in slopes: #slopes is to expected to be a nested array #[[0,30],[83,50],[160,10],[270,88]] will return radiation for 4 objects, [[0,30]] for one.
                Ib, Iid, Icd, Ihd, Igr, theta = InclanatedRadiationComponents(i[0],i[1],Dir,Dif,latitude,longitude,alt,azi,start,Rhog)
                temp.append(start.strftime(TimeFormat))
                temp.append(Ib+Iid+Icd+Ihd+Igr)
                temp.append(Ib)
                temp.append(Iid)
                temp.append(Icd)
                temp.append(Ihd)
                temp.append(Igr)
                temp.append(theta)
            DataSlopes.append(temp)
        Data.append([start.strftime(TimeFormat),rad,Dir,Dif,alt,azi])
        start = start + datetime.timedelta(seconds=TimeStep)# increase time with timestep
        c += 1
        hour = int(GetHourOfYear(start))
    if plot:
        PlotSampleNew(Data)
        PlotSampleNew(DataSlopes,SlopeColumnMapping(slopes),fieldnames)
    return Data, DataSlopes

def SlopeColumnMapping(slopes):
    AmountOfColumns = 6
    counter = 1
    PlotStructure = []
    for i in slopes:
        temp = []
        for j in range(AmountOfColumns):
            temp.append(counter)
            counter += 1
        PlotStructure.append(temp)
        counter += 2
    return PlotStructure

def ToCSV(FileName,Data,fieldnames = None):
    if fieldnames == None:
        fieldnames = ["Date","Globaln","Directn","Diffusen","Altitude",'Azimuth']
    with open("%s.csv"%FileName,'wb') as csvfile:
        #fieldnames = ["Date","[W/m^2]","Altitude",'Azimuth']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for i in Data:
            datadict = {}
            for j in range(len(fieldnames)):
                if type(i) is float:
                    datadict[fieldnames[j]] = i
                else:
                    datadict[fieldnames[j]] = i[j]
            #writer.writerow({"Date":i[0],"[W/m^2]":i[1],"Altitude":i[2],'Azimuth':i[3]})
            writer.writerow(datadict)
    print( "Done writing file with filename:\n {}".format(FileName))

def CalculateLocalVariation(Filename,lat,lon,verbal=False): # takes KNMI fileformat. Q, (j/mm2), HH(hour of day)are needed at least.
    dates = []
    hourdatakt = [0.0 for i in range(366*24)]
    hourdata = [[] for i in range(366*24)]
    with open("%s.csv"%Filename,'r') as csvfile:
          reader = csv.DictReader(csvfile)
          for row in reader:
              try:
                  h = int(row["HH"])
                  h = h-1
                  if h < 10:
                      h = "0%s"%h
                  date = "%s-%s"%(row['YYYYMMDD'],h)
                  date = datetime.datetime.strptime(date+"-UTC","%Y%m%d-%H-%Z")
                  DOY = int(GetDayOfYear(date+datetime.timedelta(minutes=30))) #midpoint of hour is needed.
                  HOY = int(GetHourOfYear(date))

                  I = float((int(row["Q"])*100*100))/3600
                  alt,azi,Qsim,Qdir,Qdif = GetRad(lat,lon,date)
                  Qext = radiation.get_apparent_extraterrestrial_flux(DOY)
                  Ferror = False
                  if(I != 0 and Qsim != 0):
                    Ferror = True
                  if Ferror:
                      if hourdatakt[HOY] == 0: # init case
                          hourdatakt[HOY] = (I/Qext)
                      else:
                          hourdatakt[HOY] = ((I/Qext) + hourdatakt[HOY])/2

                      if ((I/Qsim) > 2):
                          hourdata[HOY].append(0)
                          if verbal:
                              print( "anomalous data: {}".format(I/Qsim))
                      else:
                          hourdata[HOY].append(I/Qsim)
                  if verbal and Ferror:
                      print( "I: {} Sim: {} [I/Qsim]: {} date: {}, HOY:{}".format(I,Qsim,(I/Qext),date,HOY))
              except ValueError:
                  if verbal:
                      print( "EOF")
    if verbal:
        print( hourdata)
    return hourdata,hourdatakt

def CalcStatsSet(HourData):
    HourStats = []
    for i in HourData:
        mu,sig,med,var,Min,Max = CalcStats(i)
        HourStats.append([mu,sig,med,Min,Max,var])
    return HourStats

def PlotSample(sample):
    date = []
    glob = []
    Dir = []
    Dif = []
    for i in sample:
        date.append(i[0])
        glob.append(i[1])
        Dir.append(i[2])
        Dif.append(i[3])
    g = plt.plot(glob, label='global')
    gdir = plt.plot(Dir, label='direct')
    gdif = plt.plot(Dif, label='diffuse')
    plt.ylabel('Radiation [W/m^2]')
    plt.xlabel('Time dt: {} - {}'.format(date[0],date[1]))
    plt.title('Solar radiation for specific location')
    plt.grid(True)
    plt.legend()
    plt.show()

def PlotSampleNew(sample,PlotList=[[1,2,3]],fieldnames=["Date","Globaln","Directn","Diffusen","Altitude",'Azimuth']):
    for L in PlotList:
        plt.figure()
        ColorList = ['r','g','b','k','y','m','c']
        colList = []
        Fnames = []
        i=0
        for col in range(len(sample[0])):
            Plot = False
            for j in L:
                if j == i:
                    Fnames.append(fieldnames[j])
                    Plot = True
            if Plot:
                temp = [ sample[k][col] for k in range(len(sample)) ]
                colList.append(temp)
            i += 1
        date = [ [k] for k in range(len(sample)) ]
        for i in range(len(colList)):
            plt.plot(date,colList[i],ColorList[i],label=Fnames[i])
            plt.ylabel('Radiation [W/m^2]')
            plt.xlabel('Time dt: {} - {}'.format(sample[0][0],sample[-1][0]))
            plt.title('Solar radiation for specific location')
            plt.grid(True)
            plt.legend()
    plt.show()

def CreateSet(FileName,Lat,Lon,dT=3600,start="1990-1-1",stop="2000-1-1",plot=False):
    HourData,HourDataKt = CalculateLocalVariation(FileName,Lat,Lon)
    HourStats = CalcStatsSet(HourData)
    ToCSV("hourkt_%s"%FileName, HourDataKt, fieldnames = ["kt"])
    ToCSV("Stats_%s"%FileName, HourStats, fieldnames = ["mu","sigma","median","min","max","variance"])
    sample = CalcRad(start,stop,Lat,Lon,dT,"hourkt_%s.csv"%FileName,"Stats_%s.csv"%FileName)
    if plot:
        PlotSample(sample)   #[start.strftime(TimeFormat),rad,Dir,Dif,alt,azi]
    ToCSV("result_%s"%FileName, sample[0], fieldnames = ["date","Qg","Qdir","Qdif","alt","azi"])
    return sample


def SanityCheck(check,type=None,checklist=None):
    Valid = False
    if type == "filename":
        if os.path.isfile("%s.csv"%check):
            Valid = True
        else:
            print( "Not a valid filename!!")
    if type == "number":
        try:
            check = float(check)
        except ValueError:
            valid = False
        if isinstance(check,(int,long,float)):
            if checklist is not None:
                if check >= checklist[0] and check <= checklist[1]:
                    Valid = True
            else:
                Valid = True
        else:
            print( "Not a Number!!")
            if checklist is not None:
                print( "value should be between")
                print( checklist)
    if type == "date":
        try:
            datetime.datetime.strptime(check, '%Y-%m-%d')
            Valid = True
        except ValueError:
            print(("Incorrect data format, should be YYYY-MM-DD"))
    if type == "string":
        if isinstance(check,basestring):
            if checklist is not None:
                for i in checklist:
                    if i == check:
                        Valid = True
            else:
                Valid = True
        else:
            print( "No valid text!")
            if checklist is not None:
                print( "choose from:")
                print( checklist)
            Valid = False
    return Valid


def main():
    Running = True
    Error = False
    while Running:
        while not Error:
            TimeFormat = "%Y/%m/%d %H:%M:%S"
            print( '#  Sun Radiation Calculator')
            print( '#  based on pysolar')
            print( '#  Some information is needed')
            print( '#  Latitude, longitude, filename for output,\n start-date, stop-date and timestep.')
            lat = input("Latitude(E.g. 52.100)?:")
            if not SanityCheck(lat,"number",[0,90]):
                Error = True
                break

            lon = input("Longitude(E.g. 5.180)?:")
            if not SanityCheck(lon,"number",[0,180]):
                Error = True
                break

            lat = float(lat)
            lon = float(lon)
            print( "Date format: 2016-7-28")
            startdate = input("Start date?:")
            if not SanityCheck(startdate,"date"):
                Error = True
                break

            stopdate = input("Stop date?:")
            if not SanityCheck(stopdate,"date"):
                Error = True
                break

            FileName = input("Filename?:")
            if not SanityCheck(FileName,"filename"):
                Error = True
                break

            print( "Chosen Filename: %s"%FileName)
            dT = input("Timestep in seconds?:")
            if not SanityCheck(dT,"number",[1,7200]):
                Error = True
                break
            dT = int(dT)

            print( "Modes:")
            print( "set : Create new set (generate set of Kt values and statistical data)")
            print( "simple : generates direct and diffuse radiation for horizontal plane")
            print( "surface : generates radiation components incident to tilted plane")
            CalcMode = input("Choose Mode: set/simple/surface?:")
            if not SanityCheck(CalcMode,"string",["set","simple","surface"]):
                Error = True
                break
            print( "Show interactive plot after calculation? \nDepending on mode, number of plots will change.")
            PlotResult = input("Y/N?:")
            if not SanityCheck(PlotResult,"string"):
                Error = True
                break
            if PlotResult == "Y" or PlotResult == "y" or PlotResult == "yes":
                PlotResult = True

            print( "advanced mode allows selection of custom kt-file and statistics file.")
            adv = input("Advanced mode:Y/N?:")
            if not SanityCheck(adv,"string"):
                Error = True
                break

            if adv == "Y" or adv == "y" or adv == "yes":
                standard = input("Use standard file extensions (works when using set-mode):Y/N?:")
                if not SanityCheck(standard,"string"):
                    Error = True
                    break
                if standard == "Y" or standard == "y" or standard == "yes":
                    Ktfile = "hourkt_%s.csv"%FileName
                    statsfile = "Stats_%s.csv"%FileName
                else:
                    Ktfile = input("hourly-Kt filename?:")
                    if not SanityCheck(Ktfile,"filename"):
                        Error = True
                        break
                    statsfile = input("hourly-stats filename?:")
                    if not SanityCheck(statsfile,"filename"):
                        Error = True
                        break
                adv = True
            else:
                adv = False

            if CalcMode == "set":
                CreateSet(FileName,lat,lon,dT,startdate,stopdate,True)
            elif CalcMode == "surface":
                Surfaces = []
                Fieldnames = []
                print( "The orientation of the surface in question is needed. Consult example_orientation.png for instructions.")
                NoAngles = input("How many angled surfaces do you want to calculate 1-100?")
                if not SanityCheck(NoAngles,"number",[1,100]):
                    Error = True
                    break
                for i in range(int(NoAngles)):
                    slope = input("angle in degrees measured from horizontal plane?:")
                    if not SanityCheck(slope,"number",[0,90]):
                        Error = True
                        break
                    slope_orientation = input("angle in degrees measured from meridian. South is 0, west 90 east -90?:")
                    if not SanityCheck(slope_orientation,"number",[-180,180]):
                        Error = True
                        break
                    Surfaces.append([float(slope),float(slope_orientation)])
                    Fieldnames.append("Date_%s_%s_%d"%(slope,slope_orientation,i))
                    Fieldnames.append("Itot_%s_%s_%d"%(slope,slope_orientation,i))
                    Fieldnames.append("Ib_%s_%s_%d"%(slope,slope_orientation,i))
                    Fieldnames.append("Iid_%s_%s_%d"%(slope,slope_orientation,i))
                    Fieldnames.append("Icd_%s_%s_%d"%(slope,slope_orientation,i))
                    Fieldnames.append("Ihd_%s_%s_%d"%(slope,slope_orientation,i))
                    Fieldnames.append("Igr_%s_%s_%d"%(slope,slope_orientation,i))
                    Fieldnames.append("theta_%s_%s_%d"%(slope,slope_orientation,i))
                if adv:
                    Result = CalcRad(startdate,stopdate,lat,lon,dT,Ktfile,statsfile,Surfaces,None,TimeFormat,plot=PlotResult,fieldnames=Fieldnames)
                    ToCSV("result_horizontal_%s"%FileName,Result[0])
                    ToCSV("result_surface_%s"%FileName,Result[1],fieldnames = Fieldnames)#, fieldnames = ["Ib1","Iid1","Icd1","Ihd1","Igr1","theta1","Ib2","Iid2","Icd2","Ihd2","Igr2","theta2","Ib3","Iid3","Icd3","Ihd3","Igr3","theta3","Ib4","Iid4","Icd4","Ihd4","Igr4","theta4"])
                else:
                    Result = CalcRad(startdate,stopdate,lat,lon,dT,slopes=Surfaces,TimeFormat=TimeFormat,plot=PlotResult,fieldnames=Fieldnames)
                    ToCSV("result_horizontal_%s"%FileName,Result[0])
                    ToCSV("result_surface_%s"%FileName,Result[1], fieldnames = Fieldnames)

        #(startDate,stopDate,latitude,longitude,TimeStep=,KxFile=,StatFile=,slopes=,Rhog=,TimeFormat=):
            else:
                if adv:
                    ToCSV("result_horizontal_%s"%FileName, CalcRad(startdate,stopdate,lat,lon,dT,Ktfile,statsfile,None,None,TimeFormat,plot=PlotResult,fieldnames=Fieldnames)[0])
                else:
                    ToCSV("result_horizontal_%s"%FileName, CalcRad(startdate,stopdate,lat,lon,dT,TimeFormat=TimeFormat,plot=PlotResult,fieldnames=Fieldnames)[0])

            print( 'Show help?')
            Help = input("Y/N?")
            if not SanityCheck(Help,"string"):
                Error = True
                break
            if Help =="Y" or Help == "y" or Help == "yes":
                print( "\n\n This script can also be called directly from the commandline with arguments.\n")
                print( "The simpelest way: Navigate to the folder on the commandline and run:")
                print( "python Get_Radiation.py output.csv 2017-1-12 2017-12-18 5.180 52.100\n")
                print( "arguments are seperated with space. No arguments invokes this interactive mode")
                print( "")
                print( "output.csv will be the output filename, 2017-1-12 is the startdate,")
                print( "5.180 and 52.100 are latitude and longitude.")
                print( "The Timestep will be 1 hour for the output data.\n")
                print( "the Timestep can also be specified, as the timeformat e.g.")
                print( "python Get_Radiation.py output.csv 2017-12-12 2017-12-18 5.180 52.100 60 %%Y%%M%%D-%%H%%M%%s")
                print( "this will result in data with 1 minute interval in the format 1990/09/24-23:08:46")
                print( "the following page specifies how the timeformatting can be set:")
                print( "https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior")
            quit = input("Press q to quit")
            if not SanityCheck(quit,"string"):
                Error = True
                break
            if quit == "q" or quit == "Q":
                Running = False
                break
            else:
                continue

        reset = input("Error, see above message for tip. Restart:Y/N?:")
        if not SanityCheck(reset,"string"):
            Error = True
            break
        if reset == "Y" or reset == "y" or reset == "yes":
            Error = False
        else:
            break


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        main()
    else:
        #slopes = map(float,sys.argv[].split(","))
        if len(sys.argv) == 6:
            ToCSV(sys.argv[1], CalcRad(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]))
            print( "startdate: {}, stopdate: {}, \n timestep: 3600, output format: YYYY-MM-DD_HH:MM:SS".format(sys.argv[1],sys.argv[2]))
            #python Get_Radiation.py output.csv 2017-1-12 2017-12-18 5.180 52.100
        elif len(sys.argv) == 7:
            ToCSV(sys.argv[1], CalcRad(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6]))
            print( "startdate: {}, stopdate: {}, \n timestep: {}, output format: YYYY-MM-DD_HH:MM:SS".format(sys.argv[1],sys.argv[2],sys.argv[6]))
            #python Get_Radiation.py output.csv 2017-10-12 2017-12-18 5.180 52.100 60
        elif len(sys.argv) == 8:
            ToCSV(sys.argv[1], CalcRad(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7]))
            print( "startdate: {}, stopdate: {}, \n timestep: {}, output format: {}".format(sys.argv[1],sys.argv[2],sys.argv[5],sys.argv[7]))
            #python Get_Radiation.py output.csv 2017-12-12 2017-12-18 5.180 52.100 60 %Y%M%D-%H%M%s
        else:
            print( "not enough or too many arguments")
