#=====================================================
#  import modules
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
from get_parameters import get_area_mean_min_max, get_seaice_extent

#------------------
#  start here
#------------------

# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/monthly/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/monthly/"
 
years=np.arange(2010,2020) 
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

varnms="ICEFRAC"
pole='S'  # N for Arctic, S for Antarctic

if pole is 'N':
   var_long_name="Arctic Sea Ice Extent"
   figure_name="Arctic_Sea_Ice_Extent_Monthly"
elif pole is 'S':
   var_long_name="Antarctic Sea Ice Extent"
   figure_name="Antarctic_Sea_Ice_Extent_Monthly"
units=r"km$^2$" #"Fraction"

nmon=np.int64(12)
means_yby_ctl=np.zeros((years.size,nmon)) #year by year mean for each variable
means_yby_exp=np.zeros((years.size,nmon)) #year by year mean for each variable
means_ctl=np.zeros((nmon)) #multi-year mean for each variable
means_exp=np.zeros((nmon)) #multi-year mean for each variable
diffs=np.zeros((nmon)) #multi-year exp-ctl diff for each variable
pvals=np.zeros((nmon)) #pvalues of ttest
zeros=np.zeros(nmon)
for iy in range(0,years.size): 
    # read data and calculate mean/min/max
    for im in range(0,nmon):
        # open data file
        fctl=fpath_ctl+ctl_pref+".cam.h0."+str(years[iy])+"-"+months_all[im]+".nc"
        fexp=fpath_exp+exp_pref+".cam.h0."+str(years[iy])+"-"+months_all[im]+".nc"
        file_ctl=netcdf_dataset(fctl,"r")
        file_exp=netcdf_dataset(fexp,"r")
        
        # read lat and lon
        lat=file_ctl.variables["lat"]
        lon=file_ctl.variables["lon"]
        dtctl=file_ctl.variables[varnms]
        dtexp=file_exp.variables[varnms] 

        nlat=96
        if pole == "N":
           latbound1=np.min(np.where(lat[:]>50))
           latbound2=nlat
        elif pole == "S":
           latbound1=0
           latbound2=np.max(np.where(lat[:]<-50))+1
        means_yby_ctl[iy,im]=get_seaice_extent(dtctl[0,latbound1:latbound2,:], \
                lat[latbound1:latbound2],lon,pole)
        means_yby_exp[iy,im]=get_seaice_extent(dtexp[0,latbound1:latbound2,:], \
                lat[latbound1:latbound2],lon,pole)

# compute multi-year mean and ttest
means_ctl=np.mean(means_yby_ctl,axis=0)
means_exp=np.mean(means_yby_exp,axis=0)
diffs_yby=means_yby_exp-means_yby_ctl
# stadard deviation
s1=np.std(means_yby_exp,axis=0)
s2=np.std(means_yby_exp,axis=0)
nn=years.size
stddev_diffs=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))
diffs=means_exp-means_ctl
ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=ttest.pvalue

siglev=0.05
diffs_sig=np.zeros(diffs.shape[0])
diffs_sig[:]=np.nan
diffs_unsig=np.zeros(diffs.shape[0])
for ip in range(diffs.shape[0]):
    if pvalues[ip] < siglev:
        diffs_sig[ip]=diffs[ip]
    else:
        diffs_unsig[ip]=diffs[ip]

#----------------
# make the plot
#----------------

fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.13,0.57,0.78,0.35])
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
labels_fig=np.array(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax1.plot(x[:],means_ctl[:],color="k",lw=2)
if pole == "S":
    ax1.set_title("Antarctic Sea Ice Extent (CESM2)",fontsize=14)
if pole == "N":
    ax1.set_title("Arctic Sea Ice Extent (CESM2)",fontsize=14)
ax1.set_ylabel(units,fontsize=14)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='lightgray', linestyle=':')
ax1.yaxis.grid(color='lightgray', linestyle=':')
ax1.set_xticks(x)
ax1.set_xticklabels(labels=labels_fig,rotation=0,fontsize=14)
plt.yticks(fontsize=14)
ax1.set_xlim(1,12)

ax2=fig.add_axes([0.13,0.12,0.78,0.35])
ax2.fill_between(x[:],diffs[:]-stddev_diffs[:],diffs[:]+stddev_diffs[:],facecolor="orangered",alpha=0.3)
ax2.plot(x[:],diffs[:],color="k",lw=1)
ax2.plot(x[:],diffs_sig[:],color="orangered",lw=4,alpha=1.0)
ax2.plot(x[:],zeros[:],color="lightgray",lw=1)
ax2.set_title("Differences"+" (TSIS-1 - CESM2)",fontsize=14)
ax2.set_ylabel(units,fontsize=14)
ax2.set_xlabel("Month",fontsize=14)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.xaxis.grid(color='lightgray', linestyle=':')
ax2.yaxis.grid(color='lightgray', linestyle=':')
ax2.set_xticks(x)
ax2.set_xticklabels(labels=labels_fig,rotation=0,fontsize=14)
ax2.set_xlim(1,12)
plt.yticks(fontsize=14)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig("./figures/"+figure_name+".pdf")
plt.savefig("./figures/"+figure_name+".png",dpi=(150))
plt.show()

exit()
