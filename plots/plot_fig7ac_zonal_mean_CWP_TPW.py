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
import matplotlib.collections as collections

# numpy
import numpy as np

# scipy
from scipy import stats

# parameters
from get_parameters import get_area_mean_min_max

#----------------
#  start here
#----------------

# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"
 
years=np.arange(2010,2020) 
#months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

figure_name="fig7ab_zonal_tpw_cwp_shaded"
units=r"kg m$^-$$^2$"
units2=r"g m$^-$$^2$"

nlat=np.int64(96)
means_yby_ctl_wv=np.zeros((years.size,nlat)) #year by year mean for each variable
means_yby_exp_wv=np.zeros((years.size,nlat)) #year by year mean for each variable
means_ctl_wv=np.zeros((nlat)) #multi-year mean for each variable
means_exp_wv=np.zeros((nlat)) #multi-year mean for each variable
diffs_wv=np.zeros((nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_wv=np.zeros((years.size)) #year by year mean for each variable
gm_yby_exp_wv=np.zeros((years.size)) #year by year mean for each variable

means_yby_ctl_lwp=np.zeros((years.size,nlat)) #year by year mean for each variable
means_yby_exp_lwp=np.zeros((years.size,nlat)) #year by year mean for each variable
means_ctl_lwp=np.zeros((nlat)) #multi-year mean for each variable
means_exp_lwp=np.zeros((nlat)) #multi-year mean for each variable
diffs_lwp=np.zeros((nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_lwp=np.zeros((years.size)) #year by year mean for each variable
gm_yby_exp_lwp=np.zeros((years.size)) #year by year mean for each variable

means_yby_ctl_ts=np.zeros((years.size,nlat)) #year by year mean for each variable
means_yby_exp_ts=np.zeros((years.size,nlat)) #year by year mean for each variable
means_ctl_ts=np.zeros((nlat)) #multi-year mean for each variable
means_exp_ts=np.zeros((nlat)) #multi-year mean for each variable

means_yby_exp_fice=np.zeros((years.size,nlat)) #year by year mean for each variable
means_exp_fice=np.zeros((nlat)) #multi-year mean for each variable

for iy in range(0,years.size): 
    # open data file
    fctl=fpath_ctl+ctl_pref+"_ANN_"+str(years[iy])+".nc"
    fexp=fpath_exp+exp_pref+"_ANN_"+str(years[iy])+".nc"
    file_ctl=netcdf_dataset(fctl,"r")
    file_exp=netcdf_dataset(fexp,"r")
    
    # read lat and lon
    lat=file_ctl.variables["lat"]
    lon=file_ctl.variables["lon"]
 
    means_yby_exp_fice[iy,:]=means_yby_exp_fice[iy,:]+np.mean(file_exp.variables["ICEFRAC"][0,:,:],axis=1)   

    # read data and calculate mean/min/max
    vn="TMQ"
    dtctl_wv=file_ctl.variables[vn][0,:,:]
    dtexp_wv=file_exp.variables[vn][0,:,:] 
    means_yby_ctl_wv[iy,:]= means_yby_ctl_wv[iy,:] + np.mean(dtctl_wv[:,:],axis=1)
    means_yby_exp_wv[iy,:]= means_yby_exp_wv[iy,:] + np.mean(dtexp_wv[:,:],axis=1)
    gm_yby_ctl_wv[iy]=gm_yby_ctl_wv[iy]+get_area_mean_min_max(dtctl_wv[:,:],lat[:])[0]
    gm_yby_exp_wv[iy]=gm_yby_exp_wv[iy]+get_area_mean_min_max(dtexp_wv[:,:],lat[:])[0]

    vn="TGCLDCWP"
    dtctl_lwp=file_ctl.variables[vn][0,:,:]*1000.
    dtexp_lwp=file_exp.variables[vn][0,:,:]*1000. 
    means_yby_ctl_lwp[iy,:]= means_yby_ctl_lwp[iy,:] + np.mean(dtctl_lwp[:,:],axis=1)
    means_yby_exp_lwp[iy,:]= means_yby_exp_lwp[iy,:] + np.mean(dtexp_lwp[:,:],axis=1)
    gm_yby_ctl_lwp[iy]=gm_yby_ctl_lwp[iy]+get_area_mean_min_max(dtctl_lwp[:,:],lat[:])[0]
    gm_yby_exp_lwp[iy]=gm_yby_exp_lwp[iy]+get_area_mean_min_max(dtexp_lwp[:,:],lat[:])[0]

    vn="TS"
    dtctl_ts=file_ctl.variables[vn][0,:,:]
    dtexp_ts=file_exp.variables[vn][0,:,:] 
    means_yby_ctl_ts[iy,:]= means_yby_ctl_ts[iy,:] + np.mean(dtctl_ts[:,:],axis=1)
    means_yby_exp_ts[iy,:]= means_yby_exp_ts[iy,:] + np.mean(dtexp_ts[:,:],axis=1)

# compute multi-year mean and ttest
siglev=0.05

means_ctl_wv=np.mean(means_yby_ctl_wv,axis=0)
means_exp_wv=np.mean(means_yby_exp_wv,axis=0)
diffs_wv=means_exp_wv-means_ctl_wv
ttest=stats.ttest_ind(means_yby_ctl_wv,means_yby_exp_wv,axis=0)
pvalues_wv=ttest.pvalue
diffs_sig_wv=np.zeros(diffs_wv.shape)
diffs_sig_wv[:]=np.nan

means_ctl_lwp=np.mean(means_yby_ctl_lwp,axis=0)
means_exp_lwp=np.mean(means_yby_exp_lwp,axis=0)
diffs_lwp=means_exp_lwp-means_ctl_lwp
ttest=stats.ttest_ind(means_yby_ctl_lwp,means_yby_exp_lwp,axis=0)
pvalues_lwp=ttest.pvalue
diffs_sig_lwp=np.zeros(diffs_lwp.shape)
diffs_sig_lwp[:]=np.nan

means_ctl_ts=np.mean(means_yby_ctl_ts,axis=0)
means_exp_ts=np.mean(means_yby_exp_ts,axis=0)
diffs_ts=means_exp_ts-means_ctl_ts

means_exp_fice=np.mean(means_yby_exp_fice,axis=0)

zeros=np.zeros(diffs_wv.shape)

#print(diffs_sig.size)

for iv in range(pvalues_wv.shape[0]):
   #for ip in range(pvalues_wv.shape[1]):
   if pvalues_wv[iv] < siglev:
       diffs_sig_wv[iv]=diffs_wv[iv]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

for iv in range(pvalues_lwp.shape[0]):
   #for ip in range(pvalues_lwp.shape[1]):
   if pvalues_lwp[iv] < siglev:
       diffs_sig_lwp[iv]=diffs_lwp[iv]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

diffs_wv_cc=diffs_ts[:]*means_ctl_wv[:]*0.075

# make the plot
fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.14,0.58,0.7,0.36])
ax1.plot(lat[:],means_ctl_wv[:],color="k",lw=2,ls="-" ) #,label="TPW")
ax1.set_title("Cloud and Atmospheric Water (CESM2)",fontsize=14)
ax1.set_ylabel("TPW ("+units+")",fontsize=14)
ax1.set_xlim(-90,90)
ax1.tick_params(axis="y",labelcolor="k")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax1_r=ax1.twinx()
ax1_r.set_ylabel("CWP ("+units2+")",fontsize=14,c="r")
ax1_r.plot(lat[:],means_ctl_lwp[:],color="r",lw=2,ls="-" ) #,label="CWP")
ax1_r.tick_params(axis="y",labelcolor="r")
#ax1_r.set_ylim(-0.6,0.6)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax2=fig.add_axes([0.14,0.12,0.7,0.36])
ax2.plot(lat[:],diffs_sig_wv[:],color="k",lw=4,alpha=1.0)
ax2.plot(lat[:],diffs_wv[:],color="k",lw=1 ) #,label="\u0394TPW"
ax2.plot(lat[:],diffs_wv_cc[:],color="k",lw=2,ls=":") #,label="\u0394TPW"
ax2.plot(lat[:],zeros[:],color="gray",lw=1)
ax2.set_title("Differences (TSIS-1 - CESM2)",fontsize=14) #+var_long_name,fontsize=12)
ax2.set_xlabel("Latitude",fontsize=14)
ax2.set_ylabel("\u0394TPW ("+units+")",fontsize=14)
ax2.set_xlim(-90,90)
ax2.set_ylim(-0.6,0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# add shading 
collection = collections.BrokenBarHCollection.span_where(lat[:], ymin=-1.6, ymax=2.15, \
             where=means_exp_fice >0.1,facecolor='y',alpha=0.3)
ax2.add_collection(collection)

ax2_r=ax2.twinx()
ax2_r.set_ylabel("\u0394CWP ("+units2+")",fontsize=14,c="r")
ax2_r.plot(lat[:],diffs_sig_lwp[:],color="r",lw=4,alpha=1.0)
ax2_r.plot(lat[:],diffs_lwp[:],color="r",lw=1,ls="-" ) #,label="\u0394CWP")
ax2_r.tick_params(axis="y",labelcolor="r")
ax2_r.set_ylim(-2.5,2.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("./figures/"+figure_name+".pdf")
plt.savefig("./figures/"+figure_name+".png",dpi=150)
plt.show()

exit()
