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
#ctl_pref="solar_CTL_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"
#exp_pref="solar_TSIS_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/monthly/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/monthly/"
 
years=np.arange(2010,2020) 
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

var_group_todo=1
# variable group 1:
#varnms=np.array(["ICEFRAC"])
varnms="ICEFRAC"
pole='N'
if pole is 'N':
   var_long_name="Arctic Sea Ice Fraction"
   figure_name="Arctic_Sea_Ice_Monthly"
elif pole is 'S':
   var_long_name="Antarctic Sea Ice Fraction"
   figure_name="Antarctic_Sea_Ice_Monthly"
units=" " #"Fraction"
#units=r"W/m$^2$"

# variable group 2:
#varnms=np.array(["FSSUS13","FSSUS12","FSSUS11","FSSUS10","FSSUS09",\
#        "FSSUS08","FSSUS07","FSSUS06","FSSUS05","FSSUS04",\
#        "FSSUS03","FSSUS02","FSSUS01","FSSUS14"])
#varnms_sub=np.array(["FSSDS13","FSSDS12","FSSDS11","FSSDS10","FSSDS09",\
#        "FSSDS08","FSSDS07","FSSDS06","FSSDS05","FSSDS04",\
#        "FSSDS03","FSSDS02","FSSDS01","FSSDS14"])
#var_long_name="Band-by-Band Surface net Upward SW"
#figure_name="Band_by_Band_surface_net_Upward_SW_ANN"
#units=r"W/m$^2$"

#f1=fpath_ctl+"solar_TSIS_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
#f2=fpath_exp+"tsis_ctl_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
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
        if var_group_todo is 1:
           dtctl=file_ctl.variables[varnms]
           dtexp=file_exp.variables[varnms] 
        elif var_group_todo is 2:
           dtctl=file_ctl.variables[varnms][:,:,:]-file_ctl.variables[varnms_sub][:,:,:]
           dtexp=file_exp.variables[varnms][:,:,:]-file_exp.variables[varnms_sub][:,:,:]
        nlat=96
        if pole == "N":
           latbound1=np.min(np.where(lat[:]>50))
           latbound2=nlat
        elif pole == "S":
           latbound1=0
           latbound2=np.max(np.where(lat[:]<-50))+1
        #print(lat[latbound1:latbound2])
        #exit()
        #dtdif=dtexp[:,:,:]-dtctl[:,:,:]
        means_yby_ctl[iy,im]=get_area_mean_min_max(dtctl[:,latbound1:latbound2,:],lat[latbound1:latbound2])[0]
        means_yby_exp[iy,im]=get_area_mean_min_max(dtexp[:,latbound1:latbound2,:],lat[latbound1:latbound2])[0]
        #stats_dif[i]=get_area_mean_min_max(dtdif[:,:,:],lat[:])[0]
        #stats_difp[i]=stats_dif[0]/stats_ctl[0]*100.

# compute multi-year mean and ttest
means_ctl=np.mean(means_yby_ctl,axis=0)
means_exp=np.mean(means_yby_exp,axis=0)
diffs=means_exp-means_ctl
ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=ttest.pvalue
#print(means_ctl)
#print(means_exp)
#print(pvalues)
#exit()
siglev=0.05
diffs_sig=np.zeros(diffs.shape[0])
diffs_sig[:]=np.nan
diffs_unsig=np.zeros(diffs.shape[0])
for ip in range(diffs.shape[0]):
    if pvalues[ip] < siglev:
        diffs_sig[ip]=diffs[ip]
    else:
        diffs_unsig[ip]=diffs[ip]

#compute annual mean for each year
ym_ctl=np.mean(means_yby_ctl,axis=1)
ym_exp=np.mean(means_yby_exp,axis=1)
diffs_ym=np.mean(ym_exp-ym_ctl)
ttest_ym=stats.ttest_ind(ym_ctl,ym_exp,axis=0)
print(diffs_ym)
print(ttest_ym.pvalue)

exit()
#print(diffs)
#print(diffs_sig)
#print(diffs_unsig)

# make the plot
fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.13,0.57,0.78,0.35])
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
labels_fig=np.array(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax1.plot(x[:],means_ctl[:],color="k",lw=2)
ax1.set_title("Antarctic Sea Ice Fraction (CESM2)",fontsize=14)
ax1.set_ylabel(units,fontsize=14)
#ax1.set_xlabel("month",fontsize=14)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='lightgray', linestyle=':')
ax1.yaxis.grid(color='lightgray', linestyle=':')
ax1.set_xticks(x)
ax1.set_xticklabels(labels=labels_fig,rotation=0,fontsize=14)
plt.yticks(fontsize=14)
ax1.set_xlim(1,12)
#ax1.set_ylim=([0,means_ctl.max*1.1])

ax2=fig.add_axes([0.13,0.12,0.78,0.35])
#bars=[None]*diffs_sig.size
#ax2.plot(x[:],diffs[:],color="tab:blue")
ax2.plot(x[:],diffs[:],color="k",lw=2)
ax2.plot(x[:],diffs_sig[:],color="blueviolet",lw=4,alpha=1.0)
ax2.plot(x[:],zeros[:],color="lightgray",lw=1)
ax2.set_title("Differences"+" (TSIS-1 - CESM2)",fontsize=14)
ax2.set_ylabel(units,fontsize=14)
ax2.set_xlabel("month",fontsize=14)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.xaxis.grid(color='lightgray', linestyle=':')
ax2.yaxis.grid(color='lightgray', linestyle=':')
ax2.set_xticks(x)
ax2.set_xticklabels(labels=labels_fig,rotation=0,fontsize=14)
ax2.set_xlim(1,12)
ax2.set_ylim(0.0025,0.0085)
plt.yticks(fontsize=14)
plt.savefig(figure_name+".eps")
plt.show()

exit()
