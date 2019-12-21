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

#def lon_lat_contour_model_vs_model(varnm,season,scale_ctl,scale_exp,table):
# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
#fpath_ctl=os.environ["fpath_ctl"]+"/"+os.environ["ctl_run_id"]+"_climo_"+season+".nc"
#fpath_exp=os.environ["fpath_exp"]+"/"+os.environ["exp_run_id"]+"_climo_"+season+".nc"
fpath_ctl="../DATA/solar_TSIS_cesm211_standard-ETEST-f19_g17-ens1/atm/hist/"
#fpath_ctl="../DATA/solar_ctl_cesm211_standard-ETEST-f19_g17-ens0/atm/hist/"
fpath_exp="../DATA/tsis_ctl_cesm211_standard-ETEST-f19_g17-ens1/atm/hist/"
 
f1=fpath_ctl+"solar_TSIS_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
#f1=fpath_ctl+"solar_ctl_cesm211_standard-ETEST-f19_g17-ens0.cam.h0.0001-01.nc"
f2=fpath_exp+"tsis_ctl_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
# open data file
file_ctl=netcdf_dataset(f1,"r")
file_exp=netcdf_dataset(f2,"r")

# read lat and lon
lat=file_ctl.variables["lat"]
lon=file_ctl.variables["lon"]

varnms=["FSSU13","FSSU12","FSSU11","FSSU10","FSSU09",\
        "FSSU08","FSSU07","FSSU06","FSSU05","FSSU04",\
        "FSSU03","FSSU02","FSSU01","FSSU14"]

stats_ctl=np.zeros((14))
stats_exp=np.zeros((14))
stats_dif=np.zeros((14))
stats_difp=np.zeros((14))
print(stats_ctl)
# read data and calculate mean/min/max
for i in range(14):
    print(i)
    dtctl=file_ctl.variables[varnms[i]] #*scale_ctl
    dtexp=file_exp.variables[varnms[i]] #*scale_exp
    dtdif=dtexp[:,:,:]-dtctl[:,:,:]
    stats_ctl[i]=get_area_mean_min_max(dtctl[:,:,:],lat[:])[0]
    stats_exp[i]=get_area_mean_min_max(dtexp[:,:,:],lat[:])[0]
    stats_dif[i]=get_area_mean_min_max(dtdif[:,:,:],lat[:])[0]
    stats_difp[i]=stats_dif[0]/stats_ctl[0]*100.

print(stats_dif)

fig=plt.figure(figsize=(7,4))
ax=fig.add_axes([0.15,0.25,0.7,0.6])
x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
bands=["0.2-0.26","0.26-0.34","0.34-0.44","0.44-0.63","0.63-0.78","0.78-1.24","1.24-1.3","1.3-1.63","1.63-1.94","1.94-2.15","2.15-2.5","2.5-3.08","3.08-3.85","3.85-12.2"]
bars=[None]*14
ax.bar(bands,stats_dif)
ax.set_title("Diff in clear-sky downward solar radiation at surface",fontsize=12)
ax.set_ylabel("Diff in radiation (W/m2)",fontsize=12)
ax.set_xlabel("Band wave length",fontsize=12)
ax.grid(True)
plt.xticks(x,bands,rotation=-90)
#plt.savefig("solar.png")
plt.show()

exit()
