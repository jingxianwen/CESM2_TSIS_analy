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
ctl_pref="solar_CTL_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"
ctl_fssd_pref="solar_CTL_cesm211_ETEST-f19_g17-ens0_fssd"
exp_fssd_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens0_fssd"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"
fpath_ctl_fssd="/raid00/xianwen/cesm211_solar/"+ctl_fssd_pref+"/climo/"
fpath_exp_fssd="/raid00/xianwen/cesm211_solar/"+exp_fssd_pref+"/climo/"
 
years=np.arange(2010,2020) 
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

var_group_todo=22
# variable group 1:
if var_group_todo==1:
   varnms=np.array(["FSSU13","FSSU12","FSSU11","FSSU10","FSSU09",\
           "FSSU08","FSSU07","FSSU06","FSSU05","FSSU04",\
           "FSSU03","FSSU02","FSSU01","FSSU14"])
   var_long_name="Band-by-Band TOA Upward SW"
   figure_name="Band_by_Band_TOA_Upward_SW_ANN_VIS_icealb"
   units=r"Wm$^-$$^2$"

# variable group 2:
if var_group_todo==22:

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
   figure_name="Band_by_Band_net_absorb_TOA_SFC_SW_ANN_update"
   units=r"Wm$^-$$^2$"

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
   units=r"Wm$^-$$^2$"

# variable group 4:
if var_group_todo==4:
   varnms=np.array(["FSSD13","FSSD12","FSSD11","FSSD10","FSSD09",\
           "FSSD08","FSSD07","FSSD06","FSSD05","FSSD04",\
           "FSSD03","FSSD02","FSSD01","FSSD14"])
   var_long_name="Band-by-Band TOA Downward SW"
   figure_name="Band_by_Band_TOA_downward_SW_ANN"
   units=r"Wm$^-$$^2$"

#f1=fpath_ctl+"solar_TSIS_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
#f2=fpath_exp+"tsis_ctl_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"

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
    #fctl_fssd=fpath_ctl_fssd+ctl_fssd_pref+"_ANN_"+"2000.nc"
    #fexp_fssd=fpath_exp_fssd+exp_fssd_pref+"_ANN_"+"2000.nc"
    fctl_fssd=fpath_ctl_fssd+ctl_fssd_pref+"_climo_ANN.nc"
    fexp_fssd=fpath_exp_fssd+exp_fssd_pref+"_climo_ANN.nc"

    file_ctl=netcdf_dataset(fctl,"r")
    file_exp=netcdf_dataset(fexp,"r")
    file_ctl_fssd=netcdf_dataset(fctl_fssd,"r")
    file_exp_fssd=netcdf_dataset(fexp_fssd,"r")
    
    # read lat and lon
    lat=file_ctl.variables["lat"]
    lon=file_ctl.variables["lon"]
    
    #stats_ctl=np.zeros((14))
    #stats_exp=np.zeros((14))
    #stats_dif=np.zeros((14))
    #stats_difp=np.zeros((14))
    #print(stats_ctl)
    # read data and calculate mean/min/max
    for iv in range(0,varnms_toa.size):
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
        elif var_group_todo is 22:
           dtctl_toa=file_ctl_fssd.variables[varnms_toa[iv]][:,:,:]-file_ctl.variables[varnms_sub_toa[iv]][:,:,:]
           dtexp_toa=file_exp_fssd.variables[varnms_toa[iv]][:,:,:]-file_exp.variables[varnms_sub_toa[iv]][:,:,:]
           dtctl_sfc=file_ctl.variables[varnms_sfc[iv]][:,:,:]-file_ctl.variables[varnms_sub_sfc[iv]][:,:,:]
           dtexp_sfc=file_exp.variables[varnms_sfc[iv]][:,:,:]-file_exp.variables[varnms_sub_sfc[iv]][:,:,:]
        #dtdif=dtexp[:,:,:]-dtctl[:,:,:]
        means_yby_ctl_toa[iy,iv]=get_area_mean_min_max(dtctl_toa[0,:,:],lat[:])[0]
        means_yby_exp_toa[iy,iv]=get_area_mean_min_max(dtexp_toa[0,:,:],lat[:])[0]
        means_yby_ctl_sfc[iy,iv]=get_area_mean_min_max(dtctl_sfc[0,:,:],lat[:])[0]
        means_yby_exp_sfc[iy,iv]=get_area_mean_min_max(dtexp_sfc[0,:,:],lat[:])[0]
        #stats_dif[i]=get_area_mean_min_max(dtdif[:,:,:],lat[:])[0]
        #stats_difp[i]=stats_dif[0]/stats_ctl[0]*100.

# compute multi-year mean and ttest
means_ctl_toa=np.mean(means_yby_ctl_toa,axis=0)
means_exp_toa=np.mean(means_yby_exp_toa,axis=0)
# stadard deviation
s1=np.std(means_yby_ctl_toa,axis=0)
s2=np.std(means_yby_exp_toa,axis=0)
nn=years.size
stddev_diffs_toa=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))

diffs_toa=means_exp_toa-means_ctl_toa
ttest_toa=stats.ttest_ind(means_yby_ctl_toa,means_yby_exp_toa,axis=0)
pvalues_toa= ttest_toa.pvalue

means_ctl_sfc=np.mean(means_yby_ctl_sfc,axis=0)
means_exp_sfc=np.mean(means_yby_exp_sfc,axis=0)

#print(diffs_toa)
#print(diffs_toa.sum())

# stadard deviation
s1=np.std(means_yby_ctl_sfc,axis=0)
s2=np.std(means_yby_exp_sfc,axis=0)
nn=years.size
stddev_diffs_sfc=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))

diffs_sfc=means_exp_sfc-means_ctl_sfc
ttest_sfc=stats.ttest_ind(means_yby_ctl_sfc,means_yby_exp_sfc,axis=0)
pvalues_sfc= ttest_sfc.pvalue

#print(diffs_sfc)
#print(diffs_sfc.sum())
#print(stddev_diffs_toa)
#print(stddev_diffs_sfc)

#compute annual mean for each year
#print("**** TOT SW-> ****")
#ym_ctl_toa=np.sum(means_yby_ctl_toa[:,:],axis=1)
#ym_exp_toa=np.sum(means_yby_exp_toa[:,:],axis=1)
#diffs_ym_toa=np.mean(ym_exp_toa-ym_ctl_toa)
#ttest_ym_toa=stats.ttest_ind(ym_ctl_toa,ym_exp_toa,axis=0)
#print(" toa->")
#print(diffs_ym_toa)
#print(ttest_ym_toa.pvalue)
#ym_ctl_sfc=np.sum(means_yby_ctl_sfc[:,:],axis=1)
#ym_exp_sfc=np.sum(means_yby_exp_sfc[:,:],axis=1)
#diffs_ym_sfc=np.mean(ym_exp_sfc-ym_ctl_sfc)
#ttest_ym_sfc=stats.ttest_ind(ym_ctl_sfc,ym_exp_sfc,axis=0)
#print(" sfc->")
#print(diffs_ym_sfc)
#print(ttest_ym_sfc.pvalue)
#ym_ctl_atm=ym_ctl_toa-ym_ctl_sfc
#ym_exp_atm=ym_exp_toa-ym_exp_sfc
#diffs_ym_atm=np.mean(ym_exp_atm-ym_ctl_atm)
#ttest_ym_atm=stats.ttest_ind(ym_ctl_atm,ym_exp_atm,axis=0)
#print(" atm->")
#print(diffs_ym_atm)
#print(ttest_ym_atm.pvalue)
#
#print("**** VIS-> ****")
#ym_ctl_toa=np.sum(means_yby_ctl_toa[:,0:5],axis=1)
#ym_exp_toa=np.sum(means_yby_exp_toa[:,0:5],axis=1)
#diffs_ym_toa=np.mean(ym_exp_toa-ym_ctl_toa)
#ttest_ym_toa=stats.ttest_ind(ym_ctl_toa,ym_exp_toa,axis=0)
#print(" toa->")
#print(diffs_ym_toa)
#print(ttest_ym_toa.pvalue)
#ym_ctl_sfc=np.sum(means_yby_ctl_sfc[:,0:5],axis=1)
#ym_exp_sfc=np.sum(means_yby_exp_sfc[:,0:5],axis=1)
#diffs_ym_sfc=np.mean(ym_exp_sfc-ym_ctl_sfc)
#ttest_ym_sfc=stats.ttest_ind(ym_ctl_sfc,ym_exp_sfc,axis=0)
#print(" sfc->")
#print(diffs_ym_sfc)
#print(ttest_ym_sfc.pvalue)
#ym_ctl_atm=ym_ctl_toa-ym_ctl_sfc
#ym_exp_atm=ym_exp_toa-ym_exp_sfc
#diffs_ym_atm=np.mean(ym_exp_atm-ym_ctl_atm)
#ttest_ym_atm=stats.ttest_ind(ym_ctl_atm,ym_exp_atm,axis=0)
#print(" atm->")
#print(diffs_ym_atm)
#print(ttest_ym_atm.pvalue)
#
#print("**** NIR-> ***")
#ym_ctl_toa=np.sum(means_yby_ctl_toa[:,5:],axis=1)
#ym_exp_toa=np.sum(means_yby_exp_toa[:,5:],axis=1)
#diffs_ym_toa=np.mean(ym_exp_toa-ym_ctl_toa)
#ttest_ym_toa=stats.ttest_ind(ym_ctl_toa,ym_exp_toa,axis=0)
#print(" toa->")
#print(diffs_ym_toa)
#print(ttest_ym_toa.pvalue)
#ym_ctl_sfc=np.sum(means_yby_ctl_sfc[:,5:],axis=1)
#ym_exp_sfc=np.sum(means_yby_exp_sfc[:,5:],axis=1)
#diffs_ym_sfc=np.mean(ym_exp_sfc-ym_ctl_sfc)
#ttest_ym_sfc=stats.ttest_ind(ym_ctl_sfc,ym_exp_sfc,axis=0)
#print(" sfc->")
#print(diffs_ym_sfc)
#print(ttest_ym_sfc.pvalue)
#ym_ctl_atm=ym_ctl_toa-ym_ctl_sfc
#ym_exp_atm=ym_exp_toa-ym_exp_sfc
#diffs_ym_atm=np.mean(ym_exp_atm-ym_ctl_atm)
#ttest_ym_atm=stats.ttest_ind(ym_ctl_atm,ym_exp_atm,axis=0)
#print(" atm->")
#print(diffs_ym_atm)
#print(ttest_ym_atm.pvalue)
#exit()

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


# make the plot
#fig=plt.figure(figsize=(7,8.5))
#ax1=fig.add_axes([0.15,0.62,0.78,0.33])
#ax2=fig.add_axes([0.15,0.14,0.78,0.33])
fig=plt.figure(figsize=(10,5))
ax1=fig.add_axes([0.1,0.25,0.35,0.4])
#ax2=fig.add_axes([0.6,0.2,0.35,0.5])
#ax2=fig.add_axes([0.13,0.14,0.78,0.33])
x=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
#x=[0.5,1.0,1.5,2.0,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.]
bands=["0.2-0.26","0.26-0.34","0.34-0.44","0.44-0.63","0.63-0.78","0.78-1.24","1.24-1.3","1.3-1.63","1.63-1.94","1.94-2.15","2.15-2.5","2.5-3.08","3.08-3.85","3.85-12.2"]
#color1="indigo"
#color2="limegreen"
color1="k"
color2="r"
ax1.bar(x-0.2,means_ctl_toa,width=0.5,color=color1,label="TOA") #"tab:blue"
ax1.bar(x+0.2,means_ctl_sfc,width=0.5,color=color2,label="Surface") #"tab:blue"
#ax1.set_title(var_long_name+" (CESM2)",fontsize=14)
ax1.set_title("Net Flux at TOA&Surface" +" (CESM2)",fontsize=14)
ax1.set_ylabel(units,fontsize=14)
#ax1.set_xlabel("Band wave length",fontsize=12)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='lightgray', linestyle=':')
ax1.yaxis.grid(color='lightgray', linestyle=':')
plt.xticks(x,bands,rotation=-90,fontsize=12)
#ax1.set_xticklabels(labels=bands,rotation=-45,fontsize=12)
plt.yticks(fontsize=14)
ax1.legend(fontsize=14)

#ax2=fig.add_axes([0.15,0.14,0.78,0.33])
ax2=fig.add_axes([0.55,0.25,0.35,0.4])
#bars=[None]*diffs_sig.size
#ax2.bar(bands,diffs_sig_toa,color="indigo",hatch="//",edgecolor="white")
#ax2.bar(bands,diffs_unsig_toa,color="indigo")
#ax2.bar(x-0.2,diffs_sig_toa,width=0.5,yerr=stddev_diffs_toa_sig,color="indigo",hatch="//",edgecolor="white")
ax2.bar(x-0.2,diffs_sig_toa,width=0.5,yerr=stddev_diffs_toa_sig,color=color1,ecolor="gray",edgecolor="white")
ax2.bar(x-0.2,diffs_unsig_toa,width=0.5,yerr=stddev_diffs_toa_unsig,color=color1,ecolor="gray",)
#ax2.bar(x+0.2,diffs_sig_sfc,width=0.5,yerr=stddev_diffs_sfc_sig,color="limegreen",hatch="//",edgecolor="white")
ax2.bar(x+0.2,diffs_sig_sfc,width=0.5,yerr=stddev_diffs_sfc_sig,color=color2,ecolor="gray",edgecolor="white")
ax2.bar(x+0.2,diffs_unsig_sfc,width=0.5,yerr=stddev_diffs_sfc_unsig,color=color2,ecolor="gray",)

#ax2.set_title("Diff in "+var_long_name+" (TSIS-1 - CESM2)",fontsize=14)
ax2.set_title("Differences"+" (TSIS-1 - CESM2)",fontsize=14)
ax2.set_ylabel(units,fontsize=14)
#ax2.set_xlabel("Band wave length",fontsize=14)
ax2.grid(True)
ax2.set_axisbelow(True)
#ax2.set_ylim(-1.05,0.8)
ax2.xaxis.grid(color='lightgray', linestyle=':')
ax2.yaxis.grid(color='lightgray', linestyle=':')
#ax2.set_xticklabels(labels=bands,rotation=-45,fontsize=12)
plt.xticks(x,bands,rotation=-90,fontsize=12)
plt.yticks(fontsize=14)
plt.savefig(figure_name+".eps")
plt.savefig(figure_name+".png",dpi=(150))
plt.show()

exit()
