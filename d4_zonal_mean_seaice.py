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

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"
 
years=np.arange(2010,2020) 
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

var_group_todo=1
# variable group 1:
varnms=np.array(["ICEFRAC"])
#varnms=np.array(["FSNTOA","FSNS","TS"])
var_long_name="Sea Ice Fraction"
figure_name="Seaice_zonal_ANN"
units="Fraction"
#var_long_name="Surface Net SW"
#figure_name="Surface_Net_SW_zonal_ANN"
#var_long_name="TOA Net SW"
#figure_name="TOA_Net_SW_zonal_ANN_VIS_icealb"
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
nlat=np.int64(96)
means_yby_ctl=np.zeros((years.size,varnms.size,nlat)) #year by year mean for each variable
means_yby_exp=np.zeros((years.size,varnms.size,nlat)) #year by year mean for each variable
means_ctl=np.zeros((varnms.size,nlat)) #multi-year mean for each variable
means_exp=np.zeros((varnms.size,nlat)) #multi-year mean for each variable
diffs=np.zeros((varnms.size,nlat)) #multi-year exp-ctl diff for each variable
pvals=np.zeros((varnms.size,nlat)) #pvalues of ttest

means_yby_ctl_DJF=np.zeros((years.size,varnms.size,nlat)) #year by year mean for each variable
means_yby_exp_DJF=np.zeros((years.size,varnms.size,nlat)) #year by year mean for each variable
means_ctl_DJF=np.zeros((varnms.size,nlat)) #multi-year mean for each variable
means_exp_DJF=np.zeros((varnms.size,nlat)) #multi-year mean for each variable
diffs_DJF=np.zeros((varnms.size,nlat)) #multi-year exp-ctl diff for each variable
pvals_DJF=np.zeros((varnms.size,nlat)) #pvalues of ttest

means_yby_ctl_JJA=np.zeros((years.size,varnms.size,nlat)) #year by year mean for each variable
means_yby_exp_JJA=np.zeros((years.size,varnms.size,nlat)) #year by year mean for each variable
means_ctl_JJA=np.zeros((varnms.size,nlat)) #multi-year mean for each variable
means_exp_JJA=np.zeros((varnms.size,nlat)) #multi-year mean for each variable
diffs_JJA=np.zeros((varnms.size,nlat)) #multi-year exp-ctl diff for each variable
pvals_JJA=np.zeros((varnms.size,nlat)) #pvalues of ttest

gm_yby_ctl=np.zeros((years.size)) #year by year mean for each variable
gm_yby_exp=np.zeros((years.size)) #year by year mean for each variable
for iy in range(0,years.size): 
    # open data file
    fctl=fpath_ctl+ctl_pref+"_ANN_"+str(years[iy])+".nc"
    fexp=fpath_exp+exp_pref+"_ANN_"+str(years[iy])+".nc"
    fctl_DJF=fpath_ctl+ctl_pref+"_DJF_"+str(years[iy])+".nc"
    fexp_DJF=fpath_exp+exp_pref+"_DJF_"+str(years[iy])+".nc"
    fctl_JJA=fpath_ctl+ctl_pref+"_JJA_"+str(years[iy])+".nc"
    fexp_JJA=fpath_exp+exp_pref+"_JJA_"+str(years[iy])+".nc"
    file_ctl=netcdf_dataset(fctl,"r")
    file_exp=netcdf_dataset(fexp,"r")
    file_ctl_DJF=netcdf_dataset(fctl_DJF,"r")
    file_exp_DJF=netcdf_dataset(fexp_DJF,"r")
    file_ctl_JJA=netcdf_dataset(fctl_JJA,"r")
    file_exp_JJA=netcdf_dataset(fexp_JJA,"r")
    
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
           dtctl_DJF=file_ctl_DJF.variables[varnms[iv]]
           dtexp_DJF=file_exp_DJF.variables[varnms[iv]] 
           dtctl_JJA=file_ctl_JJA.variables[varnms[iv]]
           dtexp_JJA=file_exp_JJA.variables[varnms[iv]] 
        elif var_group_todo is 2:
           dtctl=file_ctl.variables[varnms[iv]][:,:,:]-file_ctl.variables[varnms_sub[iv]][:,:,:]
           dtexp=file_exp.variables[varnms[iv]][:,:,:]-file_exp.variables[varnms_sub[iv]][:,:,:]
        #dtdif=dtexp[:,:,:]-dtctl[:,:,:]
        means_yby_ctl[iy,iv,:]=np.mean(dtctl[:,:,:],axis=2)[0,:]
        means_yby_exp[iy,iv,:]=np.mean(dtexp[:,:,:],axis=2)[0,:]
        means_yby_ctl_DJF[iy,iv,:]=np.mean(dtctl_DJF[:,:,:],axis=2)[0,:]
        means_yby_exp_DJF[iy,iv,:]=np.mean(dtexp_DJF[:,:,:],axis=2)[0,:]
        means_yby_ctl_JJA[iy,iv,:]=np.mean(dtctl_JJA[:,:,:],axis=2)[0,:]
        means_yby_exp_JJA[iy,iv,:]=np.mean(dtexp_JJA[:,:,:],axis=2)[0,:]
        #stats_dif[i]=get_area_mean_min_max(dtdif[:,:,:],lat[:])[0]
        #stats_difp[i]=stats_dif[0]/stats_ctl[0]*100.

        #gm_yby_ctl[iy]=get_area_mean_min_max(dtctl[:,:,:],lat[:])[0]
        #gm_yby_exp[iy]=get_area_mean_min_max(dtexp[:,:,:],lat[:])[0]

# compute globam mean
#diffs_ym_toa=np.mean(gm_yby_exp-gm_yby_ctl)
#ttest_ym_toa=stats.ttest_ind(gm_yby_ctl,gm_yby_exp,axis=0)
#print(diffs_ym_toa)
#print(ttest_ym_toa.pvalue)
#exit()

# compute multi-year mean and ttest
siglev=0.05
means_ctl=np.mean(means_yby_ctl,axis=0)
means_exp=np.mean(means_yby_exp,axis=0)
#diffs_yby=means_yby_exp-means_yby_ctl
#mins_diffs=np.min(diffs_yby,axis=0)
#maxs_diffs=np.max(diffs_yby,axis=0)
#stddev_diffs=np.std(diffs_yby,axis=0)
# stadard deviation
s1=np.std(means_yby_exp,axis=0)
s2=np.std(means_yby_exp,axis=0)
nn=years.size
stddev_diffs=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))

diffs=means_exp-means_ctl
ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=ttest.pvalue
diffs_sig=np.zeros(diffs.shape)
diffs_sig[:,:]=np.nan

means_ctl_DJF=np.mean(means_yby_ctl_DJF,axis=0)
means_exp_DJF=np.mean(means_yby_exp_DJF,axis=0)
#diffs_yby_DJF=means_yby_exp_DJF-means_yby_ctl_DJF
#mins_diffs_DJF=np.min(diffs_yby_DJF,axis=0)
#maxs_diffs_DJF=np.max(diffs_yby_DJF,axis=0)
#stddev_diffs_DJF=np.std(diffs_yby_DJF,axis=0)
# stadard deviation
s1=np.std(means_yby_exp_DJF,axis=0)
s2=np.std(means_yby_exp_DJF,axis=0)
nn=years.size
stddev_diffs_DJF=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))

diffs_DJF=means_exp_DJF-means_ctl_DJF
ttest_DJF=stats.ttest_ind(means_yby_ctl_DJF,means_yby_exp_DJF,axis=0)
pvalues_DJF=ttest_DJF.pvalue
diffs_sig_DJF=np.zeros(diffs_DJF.shape)
diffs_sig_DJF[:,:]=np.nan

means_ctl_JJA=np.mean(means_yby_ctl_JJA,axis=0)
means_exp_JJA=np.mean(means_yby_exp_JJA,axis=0)
#diffs_yby_JJA=means_yby_exp_JJA-means_yby_ctl_JJA
#mins_diffs_JJA=np.min(diffs_yby_JJA,axis=0)
#maxs_diffs_JJA=np.max(diffs_yby_JJA,axis=0)
#stddev_diffs_JJA=np.std(diffs_yby_JJA,axis=0)
# stadard deviation
s1=np.std(means_yby_exp_JJA,axis=0)
s2=np.std(means_yby_exp_JJA,axis=0)
nn=years.size
stddev_diffs_JJA=np.sqrt(((nn-1)*(s1**2.) + (nn-1)*s2**2.)/(nn+nn-2))

mins_ctl_JJA=np.min(means_yby_ctl_JJA,axis=0)
mins_exp_JJA=np.min(means_yby_exp_JJA,axis=0)
maxs_ctl_JJA=np.max(means_yby_ctl_JJA,axis=0)
maxs_exp_JJA=np.max(means_yby_exp_JJA,axis=0)
diffs_JJA=means_exp_JJA-means_ctl_JJA
ttest_JJA=stats.ttest_ind(means_yby_ctl_JJA,means_yby_exp_JJA,axis=0)
pvalues_JJA=ttest_JJA.pvalue
diffs_sig_JJA=np.zeros(diffs_JJA.shape)
diffs_sig_JJA[:,:]=np.nan


zeros=np.zeros(diffs.shape)

#print(diffs_sig.size)
#exit()
for iv in range(pvalues.shape[0]):
   for ip in range(pvalues.shape[1]):
       if pvalues[iv,ip] < siglev:
           diffs_sig[iv,ip]=diffs[iv,ip]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

for iv in range(pvalues_DJF.shape[0]):
   for ip in range(pvalues_DJF.shape[1]):
       if pvalues_DJF[iv,ip] < siglev:
           diffs_sig_DJF[iv,ip]=diffs_DJF[iv,ip]

for iv in range(pvalues_JJA.shape[0]):
   for ip in range(pvalues_JJA.shape[1]):
       if pvalues_JJA[iv,ip] < siglev:
           diffs_sig_JJA[iv,ip]=diffs_JJA[iv,ip]

# make the plot
fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.12,0.58,0.8,0.36])

ax1.plot(lat[:],means_ctl[0,:],color="k",lw=2,ls="-") #,label="ANN")
#ax1.plot(lat[:],means_ctl_DJF[0,:],color="royalblue",lw=2,ls="-",label="DJF")
#ax1.plot(lat[:],means_ctl_JJA[0,:],color="darkorange",lw=2,ls="-",label="JJA")
#ax1.plot(lat[:],means_exp[0,:],color="k",lw=2,ls=":",label="TSIS")
#ax1.legend(loc="upper left",fontsize=12)
#ax1.legend(fontsize=12)
ax1.set_title("Sea Ice Fraction (CESM2)",fontsize=14)
ax1.set_ylabel(units,fontsize=14)
#ax1.set_xlabel("Latitude",fontsize=14)
ax1.set_xlim(-90,90)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax2=fig.add_axes([0.12,0.12,0.8,0.36])
#ax2.fill_between(lat[:],diffs[0,:]-stddev_diffs[0,:],diffs[0,:]+stddev_diffs[0,:],facecolor='k', alpha=0.3)
#ax2.fill_between(lat[:],diffs_DJF[0,:]-stddev_diffs_DJF[0,:],diffs_DJF[0,:]+stddev_diffs_DJF[0,:],facecolor='royalblue', alpha=0.3)
ax2.fill_between(lat[:],diffs[0,:]-stddev_diffs[0,:],diffs[0,:]+stddev_diffs[0,:],facecolor='k', alpha=0.3)
#ax2.fill_between(lat[:],diffs_JJA[0,:]-stddev_diffs_JJA[0,:],diffs_JJA[0,:]+stddev_diffs_JJA[0,:],facecolor='darkorange', alpha=0.3)
#ax2.fill_between(lat[:],mins_diffs[0,:],maxs_diffs[0,:],facecolor='k', alpha=0.3)
#ax2.fill_between(lat[:],mins_diffs_DJF[0,:],maxs_diffs_DJF[0,:],facecolor='royalblue', alpha=0.5)
#ax2.fill_between(lat[:],mins_diffs_JJA[0,:],maxs_diffs_JJA[0,:],facecolor='darkorange', alpha=0.5)
ax2.plot(lat[:],diffs[0,:],color="k",lw=1)
#ax2.plot(lat[:],diffs_DJF[0,:],color="royalblue",lw=1)
#ax2.plot(lat[:],diffs_JJA[0,:],color="darkorange",lw=1)
ax2.plot(lat[:],diffs_sig[0,:],color="k",lw=4,alpha=1.)
#ax2.plot(lat[:],diffs_sig_DJF[0,:],color="royalblue",lw=4,alpha=1.)
#ax2.plot(lat[:],diffs_sig_JJA[0,:],color="darkorange",lw=4,alpha=1.)
ax2.plot(lat[:],zeros[0,:],color="lightgray",lw=1)
ax2.set_title("Differences (TSIS-1 - CESM2)",fontsize=14) #+var_long_name,fontsize=12)
ax2.set_ylabel(units,fontsize=14)
ax2.set_xlabel("Latitude",fontsize=14)
ax2.set_xlim(-90,90)
#ax2.set_ylim(-1.0,0.2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#plt.savefig(figure_name+".eps")
plt.savefig(figure_name+".pdf")
plt.show()

exit()
