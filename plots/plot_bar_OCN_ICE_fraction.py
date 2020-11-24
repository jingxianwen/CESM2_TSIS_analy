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
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
ctl_fssd_pref="solar_CTL_cesm211_ETEST-f19_g17-ens0_fssd"
exp_fssd_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens0_fssd"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"
fpath_ctl_fssd="/raid00/xianwen/cesm211_solar/"+ctl_fssd_pref+"/climo/"
fpath_exp_fssd="/raid00/xianwen/cesm211_solar/"+exp_fssd_pref+"/climo/"
 
years=np.arange(2010,2020) 
#months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]


varnms_sub_toa=np.array(["FSSU13","FSSU12","FSSU11","FSSU10","FSSU09",\
        "FSSU08","FSSU07","FSSU06","FSSU05","FSSU04",\
        "FSSU03","FSSU02","FSSU01","FSSU14"])
varnms_toa=np.array(["FSSD13","FSSD12","FSSD11","FSSD10","FSSD09",\
        "FSSD08","FSSD07","FSSD06","FSSD05","FSSD04",\
        "FSSD03","FSSD02","FSSD01","FSSD14"])
varnms_sub_sfc=np.array(["FSSUS13","FSSUS12","FSSUS11","FSSUS10","FSSUS09",\
        "FSSUS08","FSSUS07","FSSUS06","FSSUS05","FSSUS04",\
        "FSSUS03","FSSUS02","FSSUS01","FSSUS14"])
varnms_sfc=np.array(["FSSDS13","FSSDS12","FSSDS11","FSSDS10","FSSDS09",\
        "FSSDS08","FSSDS07","FSSDS06","FSSDS05","FSSDS04",\
        "FSSDS03","FSSDS02","FSSDS01","FSSDS14"])
var_long_name="Band-by-Band Surface net Upward SW"
figure_name="fig2_Band_by_Band_TOA_SFC_SW_ANN"
units=r"Wm$^-$$^2$"

#define empty variables to save global means-->
means_yby_ctl_toa=np.zeros((years.size,varnms_toa.size)) #year by year mean for each variable
means_yby_exp_toa=np.zeros((years.size,varnms_toa.size)) #year by year mean for each variable
means_ctl_toa=np.zeros((varnms_toa.size)) #multi-year mean for each variable
means_exp_toa=np.zeros((varnms_toa.size)) #multi-year mean for each variable
diffs_toa=np.zeros((varnms_toa.size)) #multi-year exp-ctl diff for each variable
pvals_toa=np.zeros((varnms_toa.size)) #pvalues of ttest

means_yby_ctl_sfc=np.zeros((years.size,varnms_sfc.size)) #year by year mean for each variable
means_yby_exp_sfc=np.zeros((years.size,varnms_sfc.size)) #year by year mean for each variable
means_ctl_sfc=np.zeros((varnms_sfc.size)) #multi-year mean for each variable
means_exp_sfc=np.zeros((varnms_sfc.size)) #multi-year mean for each variable
diffs_sfc=np.zeros((varnms_sfc.size)) #multi-year exp-ctl diff for each variable
pvals_sfc=np.zeros((varnms_sfc.size)) #pvalues of ttest

for iy in range(0,years.size): 
    # open data file
    fctl=fpath_ctl+ctl_pref+"_ANN_"+str(years[iy])+".nc"
    fexp=fpath_exp+exp_pref+"_ANN_"+str(years[iy])+".nc"
    fctl_fssd=fpath_ctl_fssd+ctl_fssd_pref+"_climo_ANN.nc"
    fexp_fssd=fpath_exp_fssd+exp_fssd_pref+"_climo_ANN.nc"

    file_ctl=netcdf_dataset(fctl,"r")
    file_exp=netcdf_dataset(fexp,"r")
    file_ctl_fssd=netcdf_dataset(fctl_fssd,"r")
    file_exp_fssd=netcdf_dataset(fexp_fssd,"r")
    
    # read lat and lon
    lat=file_ctl.variables["lat"]
    lon=file_ctl.variables["lon"]
    
    # read data and calculate mean/min/max
    for iv in range(0,varnms_toa.size):
        dtctl_toa=file_ctl_fssd.variables[varnms_toa[iv]][:,:,:] - \
                  file_ctl.variables[varnms_sub_toa[iv]][:,:,:]
        dtexp_toa=file_exp_fssd.variables[varnms_toa[iv]][:,:,:] - \
                  file_exp.variables[varnms_sub_toa[iv]][:,:,:]
        dtctl_sfc=file_ctl.variables[varnms_sfc[iv]][:,:,:] - \
                  file_ctl.variables[varnms_sub_sfc[iv]][:,:,:]
        dtexp_sfc=file_exp.variables[varnms_sfc[iv]][:,:,:] - \
                  file_exp.variables[varnms_sub_sfc[iv]][:,:,:]

        means_yby_ctl_toa[iy,iv]=get_area_mean_min_max(dtctl_toa[0,:,:],lat[:])[0]
        means_yby_exp_toa[iy,iv]=get_area_mean_min_max(dtexp_toa[0,:,:],lat[:])[0]
        means_yby_ctl_sfc[iy,iv]=get_area_mean_min_max(dtctl_sfc[0,:,:],lat[:])[0]
        means_yby_exp_sfc[iy,iv]=get_area_mean_min_max(dtexp_sfc[0,:,:],lat[:])[0]

# compute multi-year mean at TOA -->
means_ctl_toa=np.mean(means_yby_ctl_toa,axis=0)
means_exp_toa=np.mean(means_yby_exp_toa,axis=0)

# stadard deviation at TOA -->
s1=np.std(means_yby_ctl_toa,axis=0)
s2=np.std(means_yby_exp_toa,axis=0)
nn=years.size
stddev_diffs_toa=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))

# EXP-CTL differences and pvalue at TOA -->
diffs_toa=means_exp_toa-means_ctl_toa
ttest_toa=stats.ttest_ind(means_yby_ctl_toa,means_yby_exp_toa,axis=0)
pvalues_toa= ttest_toa.pvalue

#print(diffs_toa[:])
#print("UV+VIS_toa=",np.sum(diffs_toa[0:5]))
#print("NIR_toa=",np.sum(diffs_toa[5:]))

# compute multi-year mean at surface -->
means_ctl_sfc=np.mean(means_yby_ctl_sfc,axis=0)
means_exp_sfc=np.mean(means_yby_exp_sfc,axis=0)

# stadard deviation at surface -->
s1=np.std(means_yby_ctl_sfc,axis=0)
s2=np.std(means_yby_exp_sfc,axis=0)
nn=years.size
stddev_diffs_sfc=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))

# EXP-CTL differences and pvalue at surface -->
diffs_sfc=means_exp_sfc-means_ctl_sfc
ttest_sfc=stats.ttest_ind(means_yby_ctl_sfc,means_yby_exp_sfc,axis=0)
pvalues_sfc= ttest_sfc.pvalue

# Students' T test -->
siglev=0.05
diffs_sig_toa=np.zeros(diffs_toa.size)
diffs_unsig_toa=np.zeros(diffs_toa.size)
diffs_sig_sfc=np.zeros(diffs_toa.size)
diffs_unsig_sfc=np.zeros(diffs_toa.size)
stddev_diffs_toa_sig=np.zeros(diffs_toa.size)
stddev_diffs_toa_unsig=np.zeros(diffs_toa.size)
stddev_diffs_sfc_sig=np.zeros(diffs_toa.size)
stddev_diffs_sfc_unsig=np.zeros(diffs_toa.size)
for ip in range(pvalues_toa.size):
    if pvalues_toa[ip] < siglev:
        diffs_sig_toa[ip]=diffs_toa[ip]
        stddev_diffs_toa_sig[ip]=stddev_diffs_toa[ip]
    else:
        diffs_unsig_toa[ip]=diffs_toa[ip]
        stddev_diffs_toa_unsig[ip]=stddev_diffs_toa[ip]

for ip in range(pvalues_sfc.size):
    if pvalues_sfc[ip] < siglev:
        diffs_sig_sfc[ip]=diffs_sfc[ip]
        stddev_diffs_sfc_sig[ip]=stddev_diffs_sfc[ip]
    else:
        diffs_unsig_sfc[ip]=diffs_sfc[ip]
        stddev_diffs_sfc_unsig[ip]=stddev_diffs_sfc[ip]

#-----------------------------
#       make the plot
#-----------------------------
fig=plt.figure(figsize=(10,5))
ax1=fig.add_axes([0.1,0.25,0.35,0.4])
x=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
bands=["0.2-0.26","0.26-0.34","0.34-0.44","0.44-0.63","0.63-0.78","0.78-1.24","1.24-1.3","1.3-1.63","1.63-1.94","1.94-2.15","2.15-2.5","2.5-3.08","3.08-3.85","3.85-12.2"]
color1="k"
color2="r"
ax1.bar(x-0.2,means_ctl_toa,width=0.5,color=color1,label="TOA",edgecolor="white")
ax1.bar(x+0.2,means_ctl_sfc,width=0.5,color=color2,label="Surface",edgecolor="white")
ax1.text(13.7, -24.0, r'$\mu$m',fontsize=12)
ax1.set_title("Net SW Flux at TOA&Surface" +" (CESM2)",fontsize=12)
ax1.set_ylabel(units,fontsize=12)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='lightgray', linestyle=':')
ax1.yaxis.grid(color='lightgray', linestyle=':')
plt.xticks(x,bands,rotation=-90,fontsize=12)
plt.yticks(fontsize=12)
ax1.legend(fontsize=12)

ax2=fig.add_axes([0.55,0.25,0.35,0.4])
ax2.bar(x-0.2,diffs_sig_toa,width=0.5,yerr=stddev_diffs_toa_sig,color=color1,ecolor="gray",edgecolor="white")
ax2.bar(x-0.2,diffs_unsig_toa,width=0.5,yerr=stddev_diffs_toa_unsig,color=color1,ecolor="gray",)
ax2.bar(x+0.2,diffs_sig_sfc,width=0.5,yerr=stddev_diffs_sfc_sig,color=color2,ecolor="gray",edgecolor="white")
ax2.bar(x+0.2,diffs_unsig_sfc,width=0.5,yerr=stddev_diffs_sfc_unsig,color=color2,ecolor="gray",)
ax2.text(13.7, -1.98, r'$\mu$m',fontsize=12)
ax2.set_title("Differences"+" (TSIS-1 - CESM2)",fontsize=12)
ax2.set_ylabel(units,fontsize=12)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.xaxis.grid(color='lightgray', linestyle=':')
ax2.yaxis.grid(color='lightgray', linestyle=':')
plt.xticks(x,bands,rotation=-90,fontsize=12)
plt.yticks(fontsize=12)

#----------------------
# save/show the figure
#----------------------
plt.savefig("./figures/"+figure_name+".pdf")
plt.savefig("./figures/"+figure_name+".png",dpi=150)
plt.show()

exit()
