# os
import os
#import netCDF4
#from netCDF4 import Dataset as netcdf_dataset
# cartopy
#import cartopy.crs as ccrs
#from cartopy.mpl.geoaxes import GeoAxes
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.util import add_cyclic_point
# matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import AxesGrid
#import matplotlib.colors as colors
# numpy
import numpy as np
    
lat=np.linspace(-90,90,96)
delt_lon=360./144.
nlat=96
nlon=144
icefrac=np.zeros((nlat,nlon))
icefrac=icefrac+1.
print(icefrac)

# 1. area weighted average 
#convert latitude to radians
latr=np.deg2rad(lat)
delt_lon=np.deg2rad(delt_lon)
print(latr)
print(delt_lon)

#use cosine of latitudes as weights for the mean
r_earth=6371.0 # Earth Radius in km

#weights=np.cos(latr) * 2. * np.pi * (r_earth**2)

ice_ext=np.zeros((1))

for ilat in range(1,nlat):   
    #for ilon in range(0,nlon):
    #    ice_ext = icefrac[ilat,ilon] * delt_lon * (np.cos(latr[ilat-1])-np.cos(latr[ilat])) * (r_earth**2.) \
    #            + ice_ext
   ice_ext = icefrac[ilat,:].mean() * 2.*np.pi* np.absolute(np.cos(latr[ilat-1])-np.cos(latr[ilat])) * (r_earth**2.) \
          + ice_ext
print(ice_ext)
