#=====================================================
#  import modules
#=====================================================
# os
import os

# netCDF4
from netCDF4 import Dataset as netcdf_dataset

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

from pylab import *

#----------------------
#   start here
#----------------------

# data path
ctl_name="CTL" #os.environ["ctl_name"]
#exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
#exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/data/cesm211_solar_exp/"+ctl_pref+"/climo/"
#fpath_exp="/raid00/xianwen/data/cesm211_solar_exp/"+exp_pref+"/climo/"
 
#years=np.arange(2010,2020) 
#months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

var_long_name="ocean_ice_fraction"
figure_name="fig_bar_ocn_ice_fraction"
#units="Wm$^-$$^2$"
varnms=[ "OCNFRAC", "ICEFRAC", "SNOWHICE" ]
seasons=["DJF","MAM","JJA","SON"]
#define empty variables to save global means-->
means_NH=np.zeros((4,3)) # ocean frac, bare ice frac, snow frac
means_SH=np.zeros((4,3)) # ocean frac, bare ice frac, snow frac
diffs=np.zeros((4,3)) #multi-year exp-ctl diff for each variable

for iss in range(0,4): 
    # open data file
    fctl=fpath_ctl+ctl_pref+"_climo_"+seasons[iss]+".nc"
    #fexp=fpath_exp+exp_pref+"_climo_"+seasons[iss]+".nc"

    file_ctl=netcdf_dataset(fctl,"r")
    #file_exp=netcdf_dataset(fexp,"r")
    
    # read lat and lon
    lat=file_ctl.variables["lat"]
    lon=file_ctl.variables["lon"]

    # read ocean/ice fraction and snowdepth
    ocnf_ctl=file_ctl.variables["OCNFRAC"][:,:,:]
    icef_ctl=file_ctl.variables["ICEFRAC"][:,:,:]
    snwd_ctl=file_ctl.variables["SNOWHICE"][:,:,:]

    icef_snw=np.where(snwd_ctl[:,:,:]>=0.05,icef_ctl,0.0)
    icef_bare=np.where(snwd_ctl[:,:,:]<0.05,icef_ctl,0.0)

    # compute Arctic and Antarctic averages
    # 1. Arctic 
    latbound1=np.min(np.where(lat[:]>55))
    latbound2=lat.size
    means_NH[iss,0]=get_area_mean_min_max( \
                     ocnf_ctl[0,latbound1:latbound2,:],lat[latbound1:latbound2])[0]
    means_NH[iss,1]=get_area_mean_min_max( \
                     icef_snw[0,latbound1:latbound2,:],lat[latbound1:latbound2])[0]
    means_NH[iss,2]=get_area_mean_min_max( \
                     icef_bare[0,latbound1:latbound2,:],lat[latbound1:latbound2])[0]

    # 2. Antarctic
    latbound1=0
    latbound2=np.max(np.where(lat[:]<-55))+1    
    means_SH[iss,0]=get_area_mean_min_max( \
                     ocnf_ctl[0,latbound1:latbound2,:],lat[latbound1:latbound2])[0]
    means_SH[iss,1]=get_area_mean_min_max( \
                     icef_snw[0,latbound1:latbound2,:],lat[latbound1:latbound2])[0]
    means_SH[iss,2]=get_area_mean_min_max( \
                     icef_bare[0,latbound1:latbound2,:],lat[latbound1:latbound2])[0]

print(means_NH)
print(means_SH)
#exit()
#-----------------------------
#       make the plot
#-----------------------------
fig=plt.figure(figsize=(10,5))
ax1=fig.add_axes([0.1,0.25,0.35,0.4])
x=np.array([0,2,4,6])
bands=["DJF","MAM","JJA","SON"]
color1="tab:cyan"
color2="tab:olive"
color3="tab:gray"
ax1.bar(x-0.3,means_NH[:,0],width=0.3,color=color1,label="Open_water",edgecolor="white")
ax1.bar(x    ,means_NH[:,1],width=0.3,color=color2,label="Snow_ice",edgecolor="white")
ax1.bar(x+0.3,means_NH[:,2],width=0.3,color=color3,label="Bare_ice",edgecolor="white")
ax1.set_ylim(0,0.6)
#ax1.text(13.7, -24.0, r'$\mu$m',fontsize=12)
ax1.set_title("Arctic",fontsize=12)
ax1.set_ylabel("Fraction",fontsize=12)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='lightgray', linestyle=':')
ax1.yaxis.grid(color='lightgray', linestyle=':')
plt.xticks(x,bands,rotation=0,fontsize=12)
plt.yticks(fontsize=12)
ax1.legend(fontsize=10,loc="upper left")

ax2=fig.add_axes([0.55,0.25,0.35,0.4])
ax2.bar(x-0.3,means_SH[:,0],width=0.3,color=color1,label="Open_water",edgecolor="white")
ax2.bar(x    ,means_SH[:,1],width=0.3,color=color2,label="Snow_ice",edgecolor="white")
ax2.bar(x+0.3,means_SH[:,2],width=0.3,color=color3,label="Bare_ice",edgecolor="white")
ax2.set_ylim(0,0.6)
#ax2.text(13.7, -24.0, r'$\mu$m',fontsize=12)
ax2.set_title("Antarctic",fontsize=12)
ax2.set_ylabel("Fraction",fontsize=12)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.xaxis.grid(color='lightgray', linestyle=':')
ax2.yaxis.grid(color='lightgray', linestyle=':')
plt.xticks(x,bands,rotation=0,fontsize=12)
plt.yticks(fontsize=12)
#ax2.legend(fontsize=8)

#----------------------
# save/show the figure
#----------------------
plt.savefig("./figures/"+figure_name+".pdf")
plt.savefig("./figures/"+figure_name+".png",dpi=150)
plt.show()

exit()
