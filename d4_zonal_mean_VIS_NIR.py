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

#var_long_name="Surface Net SW Radiation (all-sky)"
figure_name="zonal_sfc_down_uv+vis_nir_ANN_2"
units=r"Wm$^-$$^2$"

varnms_vis=np.array(["FSSDS13","FSSDS12","FSSDS11","FSSDS10","FSSDS09"])
varnms_nir=np.array(["FSSDS08","FSSDS07","FSSDS06","FSSDS05","FSSDS04",\
                     "FSSDS03","FSSDS02","FSSDS01","FSSDS14"])

#f1=fpath_ctl+"solar_TSIS_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
#f2=fpath_exp+"tsis_ctl_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
nlat=np.int64(96)
means_yby_ctl=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_yby_exp=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_ctl=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp=np.zeros((2,nlat)) #multi-year mean for each variable
diffs=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
pvals=np.zeros((2,nlat)) #pvalues of ttest

gm_yby_ctl=np.zeros((2,years.size)) #year by year mean for each variable
gm_yby_exp=np.zeros((2,years.size)) #year by year mean for each variable

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
    for vn in varnms_vis:
        dtctl=file_ctl.variables[vn][0,:,:]
        dtexp=file_exp.variables[vn][0,:,:] 
        means_yby_ctl[iy,0,:]= means_yby_ctl[iy,0,:] + np.mean(dtctl[:,:],axis=1) #[0,:]
        means_yby_exp[iy,0,:]= means_yby_exp[iy,0,:] + np.mean(dtexp[:,:],axis=1) #[0,:]
        gm_yby_ctl[0,iy]=gm_yby_ctl[0,iy]+get_area_mean_min_max(dtctl[:,:],lat[:])[0]
        gm_yby_exp[0,iy]=gm_yby_exp[0,iy]+get_area_mean_min_max(dtexp[:,:],lat[:])[0]
    #means_yby_ctl[iy,0,:] = means_yby_ctl[iy,0,:]/len(varnms_vis)
    #means_yby_exp[iy,0,:] = means_yby_exp[iy,0,:]/len(varnms_vis)
    #gm_yby_ctl[0,iy]=gm_yby_ctl[0,iy]/len(varnms_vis)
    #gm_yby_exp[0,iy]=gm_yby_exp[0,iy]/len(varnms_vis)

    for vn in varnms_nir:
        dtctl=file_ctl.variables[vn][0,:,:]
        dtexp=file_exp.variables[vn][0,:,:] 
        means_yby_ctl[iy,1,:]= means_yby_ctl[iy,1,:] + np.mean(dtctl[:,:],axis=1) #[0,:]
        means_yby_exp[iy,1,:]= means_yby_exp[iy,1,:] + np.mean(dtexp[:,:],axis=1) #[0,:]
        gm_yby_ctl[1,iy]=gm_yby_ctl[1,iy]+get_area_mean_min_max(dtctl[:,:],lat[:])[0]
        gm_yby_exp[1,iy]=gm_yby_exp[1,iy]+get_area_mean_min_max(dtexp[:,:],lat[:])[0]
    #means_yby_ctl[iy,1,:] = means_yby_ctl[iy,1,:]/len(varnms_nir)
    #means_yby_exp[iy,1,:] = means_yby_exp[iy,1,:]/len(varnms_nir)
    #gm_yby_ctl[1,iy]=gm_yby_ctl[1,iy]/len(varnms_nir)
    #gm_yby_exp[1,iy]=gm_yby_exp[1,iy]/len(varnms_nir)
 
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
diffs=means_exp-means_ctl
ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=ttest.pvalue
diffs_sig=np.zeros(diffs.shape)
diffs_sig[:,:]=np.nan

zeros=np.zeros(diffs.shape)

#print(diffs_sig.size)
#exit()
for iv in range(pvalues.shape[0]):
   for ip in range(pvalues.shape[1]):
       if pvalues[iv,ip] < siglev:
           diffs_sig[iv,ip]=diffs[iv,ip]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

# make the plot
fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.14,0.58,0.8,0.36])

ax1.plot(lat[:],means_ctl[0,:],color="k",lw=2,ls="-",label="UV+VIS")
ax1.plot(lat[:],means_ctl[1,:],color="r",lw=2,ls="-",label="NIR")
#ax1.plot(lat[:],means_ctl_DJF[0,:],color="royalblue",lw=2,ls="-",label="DJF")
#ax1.plot(lat[:],means_ctl_JJA[0,:],color="darkorange",lw=2,ls="-",label="JJA")
#ax1.plot(lat[:],means_exp[0,:],color="k",lw=2,ls=":",label="TSIS")
#ax1.legend(loc="upper left",fontsize=12)
ax1.legend(fontsize=12)
ax1.set_title("Surface Net Down Flux (CESM2)",fontsize=14)
ax1.set_ylabel(units,fontsize=14)
#ax1.set_xlabel("Latitude",fontsize=14)
ax1.set_xlim(-90,90)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax2=fig.add_axes([0.14,0.12,0.8,0.36])
ax2.plot(lat[:],diffs[0,:],color="k",lw=1,label="\u0394UV+VIS")
ax2.plot(lat[:],diffs[1,:],color="r",lw=1,label="\u0394NIR")
#ax2.plot(lat[:],diffs_DJF[0,:],color="royalblue",lw=1)
#ax2.plot(lat[:],diffs_JJA[0,:],color="darkorange",lw=1)
ax2.plot(lat[:],diffs_sig[0,:],color="blue",lw=8,alpha=0.5)
ax2.plot(lat[:],diffs_sig[1,:],color="orange",lw=8,alpha=0.5)
#ax2.plot(lat[:],diffs_sig_DJF[0,:],color="royalblue",lw=4,alpha=1.)
#ax2.plot(lat[:],diffs_sig_JJA[0,:],color="darkorange",lw=4,alpha=1.)
ax2.plot(lat[:],zeros[0,:],color="lightgray",lw=1)
ax2.legend(fontsize=12)
ax2.set_title("Diff in Net Down Flux (TSIS-1 - CESM2)",fontsize=14) #+var_long_name,fontsize=12)
ax2.set_ylabel(units,fontsize=14)
ax2.set_xlabel("Latitude",fontsize=14)
ax2.set_xlim(-90,90)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig(figure_name+".png",dpi=150)
plt.show()

exit()
