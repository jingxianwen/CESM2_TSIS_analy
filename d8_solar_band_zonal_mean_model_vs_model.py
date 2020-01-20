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

# scipy
from scipy import stats

# parameters
from get_parameters import get_area_mean_min_max

#def lon_lat_contour_model_vs_model(varnm,season,scale_ctl,scale_exp,table):
# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"
 
years=np.arange(2010,2020) 
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

var_group_todo=2
# variable group 1:
varnms=np.array(["FSSU13","FSSU12","FSSU11","FSSU10","FSSU09",\
        "FSSU08","FSSU07","FSSU06","FSSU05","FSSU04",\
        "FSSU03","FSSU02","FSSU01","FSSU14"])
var_long_name="Zonal mean TOA Upward SW"
figure_name="Band_by_Band_TOA_Upward_SW_zonal_mean_ANN"
units=r"W/m$^2$"

# variable group 2:
varnms=np.array(["FSSUS13","FSSUS12","FSSUS11","FSSUS10","FSSUS09",\
        "FSSUS08","FSSUS07","FSSUS06","FSSUS05","FSSUS04",\
        "FSSUS03","FSSUS02","FSSUS01","FSSUS14"])
varnms_sub=np.array(["FSSDS13","FSSDS12","FSSDS11","FSSDS10","FSSDS09",\
        "FSSDS08","FSSDS07","FSSDS06","FSSDS05","FSSDS04",\
        "FSSDS03","FSSDS02","FSSDS01","FSSDS14"])
var_long_name="Zonal mean Surface net Upward SW"
figure_name="Band_by_Band_surface_net_Upward_SW_zonal_mean_ANN"
units=r"W/m$^2$"

nlat=96
nbnd=varnms.size

#f1=fpath_ctl+"solar_TSIS_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
#f2=fpath_exp+"tsis_ctl_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"

means_yby_ctl=np.zeros((years.size,varnms.size,nlat)) #year by year mean for each variable
means_yby_exp=np.zeros((years.size,varnms.size,nlat)) #year by year mean for each variable
means_ctl=np.zeros((varnms.size,nlat)) #multi-year mean for each variable
means_exp=np.zeros((varnms.size,nlat)) #multi-year mean for each variable
diffs=np.zeros((varnms.size,nlat)) #multi-year exp-ctl diff for each variable
pvals=np.zeros((varnms.size,nlat)) #pvalues of ttest

for iy in range(0,years.size): 
    # open data file
    fctl=fpath_ctl+ctl_pref+"_ANN_"+str(years[iy])+".nc"
    fexp=fpath_exp+exp_pref+"_ANN_"+str(years[iy])+".nc"
    file_ctl=netcdf_dataset(fctl,"r")
    file_exp=netcdf_dataset(fexp,"r")
    
    # read lat and lon
    lat=file_ctl.variables["lat"]
    lon=file_ctl.variables["lon"]
    
    #stats_ctl=np.zeros((14))
    #stats_exp=np.zeros((14))
    #stats_dif=np.zeros((14))
    #stats_difp=np.zeros((14))
    #print(stats_ctl)
    # read data and calculate mean/min/max
    for iv in range(0,varnms.size):
        if var_group_todo is 1:
           dtctl=file_ctl.variables[varnms[iv]]
           dtexp=file_exp.variables[varnms[iv]] 
        elif var_group_todo is 2:
           dtctl=file_ctl.variables[varnms[iv]][:,:,:]-file_ctl.variables[varnms_sub[iv]][:,:,:]
           dtexp=file_exp.variables[varnms[iv]][:,:,:]-file_exp.variables[varnms_sub[iv]][:,:,:]
        #dtdif=dtexp[:,:,:]-dtctl[:,:,:]
        #means_yby_ctl[iy,iv]=get_area_mean_min_max(dtctl[:,:,:],lat[:])[0]
        #means_yby_exp[iy,iv]=get_area_mean_min_max(dtexp[:,:,:],lat[:])[0]
        means_yby_ctl[iy,iv,:]=np.mean(dtctl[:,:,:],axis=2)
        means_yby_exp[iy,iv,:]=np.mean(dtexp[:,:,:],axis=2)
        #stats_dif[i]=get_area_mean_min_max(dtdif[:,:,:],lat[:])[0]
        #stats_difp[i]=stats_dif[0]/stats_ctl[0]*100.

# compute multi-year mean and ttest
means_ctl=np.mean(means_yby_ctl,axis=0)
means_exp=np.mean(means_yby_exp,axis=0)
diffs=means_exp-means_ctl
ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=ttest.pvalue
#print(pvalues)
siglev=0.05
diffs_sig=np.zeros(diffs.shape)
diffs_sig[:,:]=np.nan
#diffs_unsig=np.zeros(diffs.shape)
for ib in range(nbnd):
   for ilat in range(nlat):
       if pvalues[ib,ilat] < siglev:
           diffs_sig[ib,ilat]=diffs[ib,ilat]
       #else:
       #    diffs_unsig[ip]=diffs[ip]


# make the plot
fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.18,0.56,0.7,0.37])
ax2=fig.add_axes([0.18,0.1,0.7,0.37])
yloc=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
bands=np.array(["0.2-0.26","0.26-0.34","0.34-0.44","0.44-0.63","0.63-0.78","0.78-1.24","1.24-1.3","1.3-1.63","1.63-1.94","1.94-2.15","2.15-2.5","2.5-3.08","3.08-3.85","3.85-12.2"])

#ax1.bar(bands,means_ctl,color="tab:blue")
cntr1=ax1.contourf(lat[:],yloc[:],means_ctl[:,:],cmap="hot_r")
ax1.set_title(var_long_name+" "+"(CTL)",fontsize=12)
ax1.set_xlabel("Latitude",fontsize=12)
ax1.set_ylabel("Band wave length",fontsize=12)
ax1.set_yticks(yloc[:])
ax1.set_yticklabels(labels=bands) #,rotation=-45)
fig.colorbar(cntr1, ax=ax1)

cmap2="bwr"
cntr2=ax2.contourf(lat[:],yloc[:],diffs[:,:],cmap=cmap2)
ax2.contourf(lat[:],yloc[:],diffs_sig[:,:],cmap=cmap2,hatches=[".."])
ax2.set_title(var_long_name+" "+"(TSIS-CTL)",fontsize=12)
ax2.set_xlabel("Latitude",fontsize=12)
ax2.set_ylabel("Band wave length",fontsize=12)
ax2.set_yticks(yloc[:])
ax2.set_yticklabels(labels=bands) #,rotation=-45)
fig.colorbar(cntr2, ax=ax2)

#bars=[None]*diffs_sig.size
#ax2.bar(bands,diffs_sig,color="tab:blue",hatch="//",edgecolor="black")
#ax2.bar(bands,diffs_unsig,color="tab:blue")
#
#ax2.set_title("Diff in "+var_long_name,fontsize=12)
#ax2.set_ylabel(units,fontsize=12)
#ax2.set_xlabel("Band wave length",fontsize=12)
#ax2.grid(True)
#ax2.set_axisbelow(True)
#ax2.xaxis.grid(color='gray', linestyle=':')
#ax2.yaxis.grid(color='gray', linestyle=':')
#plt.xticks(x,bands,rotation=-45)
#plt.savefig(figure_name+".png")
plt.show()

exit()
