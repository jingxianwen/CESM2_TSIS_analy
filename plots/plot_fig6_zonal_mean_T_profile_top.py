#=====================================================
# import modules
#=====================================================
# os
import os

#import netCDF4
from netCDF4 import Dataset as netcdf_dataset

# cartopy
#import cartopy.crs as ccrs
#from cartopy.mpl.geoaxes import GeoAxes
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.util import add_cyclic_point

# matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.colors as colors

# numpy
import numpy as np

# scipy
from scipy import stats

# parameters
from get_parameters import get_area_mean_min_max

#---------------------
#  start here
#---------------------

# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/data/cesm211_solar_exp/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/data/cesm211_solar_exp/"+exp_pref+"/climo/"
 
years=np.arange(2010,2020) 
#months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

var_group_todo=1
# variable group 1:
varnms=np.array(["T"])
#varnms=np.array(["FSNTOA","FSNS","TS"])
var_long_name="Atmosphere_Temperature"
figure_name="fig6_Atmosphere_Temperature_zonal_ANN"
units="K"

nlat=np.int64(96)
nlev=np.int64(32)
means_yby_ctl=np.zeros((years.size,varnms.size,nlev,nlat)) #year by year mean for each variable
means_yby_exp=np.zeros((years.size,varnms.size,nlev,nlat)) #year by year mean for each variable
means_ctl=np.zeros((varnms.size,nlev,nlat)) #multi-year mean for each variable
means_exp=np.zeros((varnms.size,nlev,nlat)) #multi-year mean for each variable
diffs=np.zeros((varnms.size,nlev,nlat)) #multi-year exp-ctl diff for each variable
pvals=np.zeros((varnms.size,nlev,nlat)) #pvalues of ttest

means_yby_ps=np.zeros((years.size,nlat))
means_ps=np.zeros((nlat))

for iy in range(0,years.size): 
    # open data file
    fctl=fpath_ctl+ctl_pref+"_ANN_"+str(years[iy])+".nc"
    fexp=fpath_exp+exp_pref+"_ANN_"+str(years[iy])+".nc"
    file_ctl=netcdf_dataset(fctl,"r")
    file_exp=netcdf_dataset(fexp,"r")
    
    # read lat and lon
    lat=file_ctl.variables["lat"]
    lon=file_ctl.variables["lon"]
    lev=file_ctl.variables["lev"]
    
    # read data and calculate mean/min/max
    for iv in range(0,varnms.size):
        dtctl=file_ctl.variables[varnms[iv]]
        dtexp=file_exp.variables[varnms[iv]] 

        means_yby_ctl[iy,iv,:]=np.mean(dtctl[:,:,:,:],axis=3)[0,:,:]
        means_yby_exp[iy,iv,:]=np.mean(dtexp[:,:,:,:],axis=3)[0,:,:]
        ps=file_ctl.variables["PS"]
        means_yby_ps[iy,:]=np.mean(ps[:,:,:],axis=2)[0,:]

# compute multi-year mean and ttest
siglev=0.05
means_ctl=np.mean(means_yby_ctl,axis=0)
means_exp=np.mean(means_yby_exp,axis=0)
means_ps=np.mean(means_yby_ps,axis=0)*0.01

for ilat in range(0,nlat):
    means_ctl[0,:,ilat]=np.where(lev<means_ps[ilat],means_ctl[0,:,ilat],np.nan)
    means_exp[0,:,ilat]=np.where(lev<means_ps[ilat],means_exp[0,:,ilat],np.nan)

# stadard deviation
s1=np.std(means_yby_exp,axis=0)
s2=np.std(means_yby_exp,axis=0)
nn=years.size
stddev_diffs=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))

diffs=means_exp-means_ctl
ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=ttest.pvalue
diffs_sig=np.zeros(diffs.shape)
diffs_sig[:,:,:]=np.nan

zeros=np.zeros(diffs.shape)

for iv in range(pvalues.shape[0]):
   for ip in range(pvalues.shape[1]):
      for ix in range(pvalues.shape[2]):
       if pvalues[iv,ip,ix] < siglev:
           diffs_sig[iv,ip,ix]=diffs[iv,ip,ix]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

#--------------------
# make the plot
#--------------------

fig=plt.figure(figsize=(8,5))
panel = [(0.2,0.2,0.45,0.6),(0.6,0.1,0.35,0.6)]
cnlevels= np.linspace(-0.5,0.5,11)
ax1=fig.add_axes(panel[0])
p1 = ax1.contourf(lat[:],lev[:],diffs[0,:,:],levels=cnlevels,cmap="bwr",extend="both")

ax1.set_title("\u0394T (TSIS-1 - CESM2)",fontsize=14)
ax1.set_xlim(-90,90)
ax1.set_ylim(1000,3.6)
ax1.set_xticks([-80,-60,-40,-20,0,20,40,60,80])
ax1.set_xticklabels(["-80","-60","-40","-20","0","20","40","60","80"],fontsize=12)
ax1.set_yscale("log")
ax1.set_yticks([1000,800,600,400,300,200,100,3.6])
ax1.set_yticklabels(["1000","800","600","400","300","200","100","3"],fontsize=12)

# color bar
cbax = fig.add_axes((panel[0][0] + 0.47, panel[0][1]+ 0.0235, 0.02, 0.5))
cbar = fig.colorbar(p1, cax=cbax, ticks=cnlevels)
cbar.ax.tick_params(labelsize=13.0, length=0)

ax1.text(99, 130, 'K',fontsize=14)

p2 = ax1.contourf(lat[:],lev[:],diffs_sig[0,:,:],levels=cnlevels,cmap="bwr",hatches=['...'],extend="both")

ax1.set_ylabel("Pressure (hPa)",fontsize=13)
ax1.set_xlabel("Latitude",fontsize=13)

#plt.savefig("./figures/"+figure_name+".pdf")
#plt.savefig("./figures/"+figure_name+".png",dpi=150)
plt.show()

exit()
