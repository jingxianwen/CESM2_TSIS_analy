#=====================================================
#
#=====================================================
# os
import os
#import netCDF4
from netCDF4 import Dataset as netcdf_dataset
# cartopy
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
# matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.colors as colors
# numpy
import numpy as np
# parameters
from get_parameters import get_area_mean_min_max

# data path
ctl_name="CTL" #"CTL" #os.environ["ctl_name"]
ctl_pref="solar_"+ctl_name+"_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"

f1=fpath_ctl+ctl_pref+"_climo_ANN.nc"

print(f1)

varlst=[ \
        "SOLIN","FSNTOA","FSNTOAC","FLUT","FLUTC", \
        "FSDS","FSDSC","FSNS","FSNSC","FLDS","FLNS","FLNSC","LHFLX","SHFLX"]

# open data file
file_ctl=netcdf_dataset(f1,"r")

# read lat and lon
lat=file_ctl.variables["lat"]
lon=file_ctl.variables["lon"]
#lev=file_ctl.variables["lev"]
#lev500=np.min(np.where(lev[:]>500.))
lat_N=np.min(np.where(lat[:]>60.))
lat_S=np.max(np.where(lat[:]<-60.))+1
for var in varlst:
    dtctl=file_ctl.variables[var][0,:,:] #[time,lat,lon]
    stat=get_area_mean_min_max(dtctl[0:lat_S,:],lat[lat_N:])
    print(var,round(stat[0],1))
