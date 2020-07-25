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

from pylab import *

#def lon_lat_contour_model_vs_model(varnm,season,scale_ctl,scale_exp,table):
# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
#ctl_pref="solar_CTL_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"
#exp_pref="solar_TSIS_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens0_fssd"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens0_fssd"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"
 
years=np.arange(2000,2001) 
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

var_group_todo=4
# variable group 1:
if var_group_todo==1:
   varnms=np.array(["FSSU13","FSSU12","FSSU11","FSSU10","FSSU09",\
           "FSSU08","FSSU07","FSSU06","FSSU05","FSSU04",\
           "FSSU03","FSSU02","FSSU01","FSSU14"])
   var_long_name="Band-by-Band TOA Upward SW"
   figure_name="Band_by_Band_TOA_Upward_SW_ANN_VIS_icealb"
   units=r"Wm$^-^2$"

# variable group 2:
if var_group_todo==2:
   varnms=np.array(["FSSUS13","FSSUS12","FSSUS11","FSSUS10","FSSUS09",\
           "FSSUS08","FSSUS07","FSSUS06","FSSUS05","FSSUS04",\
           "FSSUS03","FSSUS02","FSSUS01","FSSUS14"])
   varnms_sub=np.array(["FSSDS13","FSSDS12","FSSDS11","FSSDS10","FSSDS09",\
           "FSSDS08","FSSDS07","FSSDS06","FSSDS05","FSSDS04",\
           "FSSDS03","FSSDS02","FSSDS01","FSSDS14"])
   var_long_name="Band-by-Band Surface net Upward SW"
   figure_name="Band_by_Band_surface_net_Upward_SW_ANN_VIS_icealb"
   units=r"Wm$^-^2$"

# variable group 3 (clear sky):
if var_group_todo==3:
   varnms=np.array(["FSSU13","FSSU12","FSSU11","FSSU10","FSSU09",\
           "FSSU08","FSSU07","FSSU06","FSSU05","FSSU04",\
           "FSSU03","FSSU02","FSSU01","FSSU14"])
   varnms_sub=np.array(["FSSUCLR13","FSSUCLR12","FSSUCLR11","FSSUCLR10","FSSUCLR09",\
           "FSSUCLR08","FSSUCLR07","FSSUCLR06","FSSUCLR05","FSSUCLR04",\
           "FSSUCLR03","FSSUCLR02","FSSUCLR01","FSSUCLR14"])
   var_long_name="Band-by-Band TOA SWCF"
   figure_name="Band_by_Band_TOA_SWCF_ANN_VIS_icealb"
   units=r"Wm$^-^2$"

# variable group 4:
if var_group_todo==4:
   #varnms=np.array(["FSSD13","FSSD12","FSSD11","FSSD10","FSSD09",\
   #        "FSSD08","FSSD07","FSSD06","FSSD05","FSSD04",\
   #        "FSSD03","FSSD02","FSSD01","FSSD14"])
   varnms=np.array(['SOLIN'])
   var_long_name="Band-by-Band TOA Downward SW"
   figure_name="Band_by_Band_TOA_downward_SW_ANN"
   units=r"Wm$^-$$^2$"

#f1=fpath_ctl+"solar_TSIS_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
#f2=fpath_exp+"tsis_ctl_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"

means_yby_ctl=np.zeros((years.size,varnms.size)) #year by year mean for each variable
means_yby_exp=np.zeros((years.size,varnms.size)) #year by year mean for each variable
means_ctl=np.zeros((varnms.size)) #multi-year mean for each variable
means_exp=np.zeros((varnms.size)) #multi-year mean for each variable
diffs=np.zeros((varnms.size)) #multi-year exp-ctl diff for each variable
pvals=np.zeros((varnms.size)) #pvalues of ttest

for iy in range(0,years.size): 
    # open data file
    #fctl=fpath_ctl+ctl_pref+"_ANN_"+str(years[iy])+".nc"
    #fexp=fpath_exp+exp_pref+"_ANN_"+str(years[iy])+".nc"

    fctl=fpath_ctl+ctl_pref+"_climo_ANN"+".nc"
    fexp=fpath_exp+exp_pref+"_climo_ANN"+".nc"

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
        elif var_group_todo is 3:
           dtctl=file_ctl.variables[varnms[iv]][:,:,:]-file_ctl.variables[varnms_sub[iv]][:,:,:]
           dtexp=file_exp.variables[varnms[iv]][:,:,:]-file_exp.variables[varnms_sub[iv]][:,:,:]
        elif var_group_todo is 4:
           dtctl=file_ctl.variables[varnms[iv]]
           dtexp=file_exp.variables[varnms[iv]] 
        #dtdif=dtexp[:,:,:]-dtctl[:,:,:]
        means_yby_ctl[iy,iv]=get_area_mean_min_max(dtctl[0,:,:],lat[:])[0]
        means_yby_exp[iy,iv]=get_area_mean_min_max(dtexp[0,:,:],lat[:])[0]
        #stats_dif[i]=get_area_mean_min_max(dtdif[:,:,:],lat[:])[0]
        #stats_difp[i]=stats_dif[0]/stats_ctl[0]*100.

# compute multi-year mean and ttest
means_ctl=np.mean(means_yby_ctl,axis=0)
means_exp=np.mean(means_yby_exp,axis=0)
diffs=means_exp-means_ctl

print(np.round(means_ctl*4.,3))
print(np.round(means_exp*4.,3))
print(np.round(diffs*4.,3))
print(np.sum(means_ctl*4))
print(np.sum(means_exp*4))
print(np.sum(diffs*4))
exit()
#ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=0. # ttest.pvalue
print(pvalues)
siglev=0.05
diffs_sig=np.zeros(diffs.size)
diffs_unsig=np.zeros(diffs.size)
diffs_sig=diffs
#for ip in range(pvalues.size):
#    if pvalues[ip] < siglev:
#        diffs_sig[ip]=diffs[ip]
#    else:
#        diffs_unsig[ip]=diffs[ip]


# make the plot
fig=plt.figure(figsize=(7,8.5))
ax1=fig.add_axes([0.13,0.62,0.78,0.33])
#ax2=fig.add_axes([0.13,0.14,0.78,0.33])
x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
bands=["0.2-0.26","0.26-0.34","0.34-0.44","0.44-0.63","0.63-0.78","0.78-1.24","1.24-1.3","1.3-1.63","1.63-1.94","1.94-2.15","2.15-2.5","2.5-3.08","3.08-3.85","3.85-12.2"]

ax1.bar(bands,means_ctl,color="indigo") #"tab:blue"
#ax1.set_title(var_long_name+" (CESM2)",fontsize=14)
ax1.set_title("Band-Integrated SSI" +" (CESM2)",fontsize=14)
ax1.set_ylabel(units,fontsize=14)
#ax1.set_xlabel("Band wave length",fontsize=12)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='lightgray', linestyle=':')
ax1.yaxis.grid(color='lightgray', linestyle=':')
#plt.xticks(x,bands,rotation=-45)
ax1.set_xticklabels(labels=bands,rotation=-45,fontsize=12)
#ylocs,ylabels=yticks()
#print(ylocs,ylabels)
#ax1.set_yticklabels(labels=ylabels,fontsize=12)
plt.yticks(fontsize=14)

ax2=fig.add_axes([0.13,0.14,0.78,0.33])
#bars=[None]*diffs_sig.size
ax2.bar(bands,diffs_sig,color="indigo",hatch="//",edgecolor="white")
ax2.bar(bands,diffs_unsig,color="indigo")

#ax2.set_title("Diff in "+var_long_name+" (TSIS-1 - CESM2)",fontsize=14)
ax2.set_title("Differences"+" (TSIS-1 - CESM2)",fontsize=14)
ax2.set_ylabel(units,fontsize=14)
ax2.set_xlabel("Band wave length",fontsize=14)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_ylim(-1.25,1.25)
ax2.xaxis.grid(color='lightgray', linestyle=':')
ax2.yaxis.grid(color='lightgray', linestyle=':')
ax2.set_xticklabels(labels=bands,rotation=-45,fontsize=12)
plt.yticks(fontsize=14)
plt.savefig(figure_name+".eps")
plt.savefig(figure_name+".png")
plt.show()

exit()
