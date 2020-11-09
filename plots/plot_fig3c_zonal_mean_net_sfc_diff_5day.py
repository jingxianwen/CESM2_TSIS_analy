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

#--------------------
#   start here
#--------------------

# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens0_5days"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens0_5days"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/"
 
figure_name="fig3c_zonal_sfc_net_5day-diag_uv+vis_nir_ANN_shaded"
units=r"Wm$^-$$^2$"

varnms_vis_dn=np.array(["FSSDS13","FSSDS12","FSSDS11","FSSDS10","FSSDS09"])
varnms_nir_dn=np.array(["FSSDS08","FSSDS07","FSSDS06","FSSDS05","FSSDS04",\
                     "FSSDS03","FSSDS02","FSSDS01","FSSDS14"])

varnms_vis_up=np.array(["FSSUS13","FSSUS12","FSSUS11","FSSUS10","FSSUS09"])
varnms_nir_up=np.array(["FSSUS08","FSSUS07","FSSUS06","FSSUS05","FSSUS04",\
                     "FSSUS03","FSSUS02","FSSUS01","FSSUS14"])

nlat=np.int64(96)
means_yby_ctl_dn=np.zeros((2,nlat)) #year by year mean for each variable
means_yby_exp_dn=np.zeros((2,nlat)) #year by year mean for each variable
means_ctl_dn=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_dn=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_dn=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_dn=np.zeros((2)) #year by year mean for each variable
gm_yby_exp_dn=np.zeros((2)) #year by year mean for each variable

means_yby_ctl_up=np.zeros((2,nlat)) #year by year mean for each variable
means_yby_exp_up=np.zeros((2,nlat)) #year by year mean for each variable
means_ctl_up=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_up=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_up=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_up=np.zeros((2)) #year by year mean for each variable
gm_yby_exp_up=np.zeros((2)) #year by year mean for each variable

means_yby_ctl_net=np.zeros((2,nlat)) #year by year mean for each variable
means_yby_exp_net=np.zeros((2,nlat)) #year by year mean for each variable
means_ctl_net=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_net=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_net=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_net=np.zeros((2)) #year by year mean for each variable
gm_yby_exp_net=np.zeros((2)) #year by year mean for each variable

means_yby_ctl_fice=np.zeros((nlat)) #year by year mean for each variable
means_yby_exp_fice=np.zeros((nlat)) #year by year mean for each variable

# open data file
fctl=fpath_ctl+ctl_pref+"_climo_ANN.nc"
fexp=fpath_exp+exp_pref+"_climo_ANN.nc"
file_ctl=netcdf_dataset(fctl,"r")
file_exp=netcdf_dataset(fexp,"r")

# read lat and lon
lat=file_ctl.variables["lat"]
lon=file_ctl.variables["lon"]

dtctl_fice=file_ctl.variables["ICEFRAC"][0,:,:]
means_yby_ctl_fice[:]= np.mean(dtctl_fice[:,:],axis=1)

# read data and calculate mean/min/max
for vn in varnms_vis_dn:
    dtctl_dn=file_ctl.variables[vn][0,:,:]
    dtexp_dn=file_exp.variables[vn][0,:,:] 
    means_yby_ctl_dn[0,:]= means_yby_ctl_dn[0,:] + np.mean(dtctl_dn[:,:],axis=1)
    means_yby_exp_dn[0,:]= means_yby_exp_dn[0,:] + np.mean(dtexp_dn[:,:],axis=1)
    gm_yby_ctl_dn[0]=gm_yby_ctl_dn[0]+get_area_mean_min_max(dtctl_dn[:,:],lat[:])[0]
    gm_yby_exp_dn[0]=gm_yby_exp_dn[0]+get_area_mean_min_max(dtexp_dn[:,:],lat[:])[0]

for vn in varnms_nir_dn:
    dtctl_dn=file_ctl.variables[vn][0,:,:]
    dtexp_dn=file_exp.variables[vn][0,:,:] 
    means_yby_ctl_dn[1,:]= means_yby_ctl_dn[1,:] + np.mean(dtctl_dn[:,:],axis=1) #[0,:]
    means_yby_exp_dn[1,:]= means_yby_exp_dn[1,:] + np.mean(dtexp_dn[:,:],axis=1) #[0,:]
    gm_yby_ctl_dn[1]=gm_yby_ctl_dn[1]+get_area_mean_min_max(dtctl_dn[:,:],lat[:])[0]
    gm_yby_exp_dn[1]=gm_yby_exp_dn[1]+get_area_mean_min_max(dtexp_dn[:,:],lat[:])[0]

for vn in varnms_vis_up:
    dtctl_up=file_ctl.variables[vn][0,:,:]
    dtexp_up=file_exp.variables[vn][0,:,:] 
    means_yby_ctl_up[0,:]= means_yby_ctl_up[0,:] + np.mean(dtctl_up[:,:],axis=1)
    means_yby_exp_up[0,:]= means_yby_exp_up[0,:] + np.mean(dtexp_up[:,:],axis=1)
    gm_yby_ctl_up[0]=gm_yby_ctl_up[0]+get_area_mean_min_max(dtctl_up[:,:],lat[:])[0]
    gm_yby_exp_up[0]=gm_yby_exp_up[0]+get_area_mean_min_max(dtexp_up[:,:],lat[:])[0]

for vn in varnms_nir_up:
    dtctl_up=file_ctl.variables[vn][0,:,:]
    dtexp_up=file_exp.variables[vn][0,:,:] 
    means_yby_ctl_up[1,:]= means_yby_ctl_up[1,:] + np.mean(dtctl_up[:,:],axis=1) #[0,:]
    means_yby_exp_up[1,:]= means_yby_exp_up[1,:] + np.mean(dtexp_up[:,:],axis=1) #[0,:]
    gm_yby_ctl_up[1]=gm_yby_ctl_up[1]+get_area_mean_min_max(dtctl_up[:,:],lat[:])[0]
    gm_yby_exp_up[1]=gm_yby_exp_up[1]+get_area_mean_min_max(dtexp_up[:,:],lat[:])[0]

means_yby_ctl_net[:,:]=means_yby_ctl_dn[:,:]-means_yby_ctl_up[:,:]
means_yby_exp_net[:,:]=means_yby_exp_dn[:,:]-means_yby_exp_up[:,:]
gm_yby_ctl_net[:]=gm_yby_ctl_dn[:]-gm_yby_ctl_up[:]
gm_yby_exp_net[:]=gm_yby_exp_dn[:]-gm_yby_exp_up[:]

# compute multi-year mean and ttest
###siglev=0.05
###
means_ctl_dn=means_yby_ctl_dn
means_exp_dn=means_yby_exp_dn
diffs_dn=means_exp_dn-means_ctl_dn
###ttest=stats.ttest_ind(means_yby_ctl_dn,means_yby_exp_dn,axis=0)
###pvalues_dn=ttest.pvalue
###diffs_sig_dn=np.zeros(diffs_dn.shape)
###diffs_sig_dn[:,:]=np.nan
###
means_ctl_up=means_yby_ctl_up
means_exp_up=means_yby_exp_up
diffs_up=means_exp_up-means_ctl_up
###ttest=stats.ttest_ind(means_yby_ctl_up,means_yby_exp_up,axis=0)
###pvalues_up=ttest.pvalue
###diffs_sig_up=np.zeros(diffs_up.shape)
###diffs_sig_up[:,:]=np.nan
###
means_ctl_net=means_yby_ctl_net
means_exp_net=means_yby_exp_net
diffs_net=means_exp_net-means_ctl_net
diffs_net_bb=diffs_net[0,:]+diffs_net[1,:]

#compute domain mean
#diffs_net_bb_mask=np.where(means_yby_ctl_fice[:]>0.1,diffs_net_bb,np.nan)
diffs_net_bb_mask=np.ma.MaskedArray(diffs_net_bb,mask=means_yby_ctl_fice[:]>0.1)
#print(diffs_net_bb_mask)
latr=np.deg2rad(lat)
weights=np.cos(latr)
avg_Antarctic=np.average(diffs_net_bb_mask[0:40],axis=0,weights=weights[0:40]) 
avg_Arctic=np.average(diffs_net_bb_mask[60:],axis=0,weights=weights[60:]) 

#print(avg_Antarctic)
#print(avg_Arctic)
#exit()

###ttest=stats.ttest_ind(means_yby_ctl_net,means_yby_exp_net,axis=0)
###pvalues_net=ttest.pvalue
###diffs_sig_net=np.zeros(diffs_net.shape)
###diffs_sig_net[:,:]=np.nan
###
zeros=np.zeros(diffs_dn.shape)
###
####print(diffs_sig.size)
###
###for iv in range(pvalues_up.shape[0]):
###   for ip in range(pvalues_up.shape[1]):
###       if pvalues_up[iv,ip] < siglev:
###           diffs_sig_up[iv,ip]=diffs_up[iv,ip]
###       #else:
###       #    diffs_unsig[iv,ip]=diffs[iv,ip]
###
###for iv in range(pvalues_dn.shape[0]):
###   for ip in range(pvalues_dn.shape[1]):
###       if pvalues_dn[iv,ip] < siglev:
###           diffs_sig_dn[iv,ip]=diffs_dn[iv,ip]
###       #else:
###       #    diffs_unsig[iv,ip]=diffs[iv,ip]
###
###for iv in range(pvalues_net.shape[0]):
###   for ip in range(pvalues_net.shape[1]):
###       if pvalues_net[iv,ip] < siglev:
###           diffs_sig_net[iv,ip]=diffs_net[iv,ip]
###       #else:
###       #    diffs_unsig[iv,ip]=diffs[iv,ip]

#----------------
# make the plot
#----------------

fig=plt.figure(figsize=(7,4))

#ax1=fig.add_axes([0.14,0.58,0.8,0.36])
#ax1.plot(lat[:],means_ctl_dn[0,:],color="k",lw=2,ls="-",label="UV+VIS down")
#ax1.plot(lat[:],means_ctl_dn[1,:],color="r",lw=2,ls="-",label="NIR down")
#ax1.plot(lat[:],means_ctl_up[0,:],color="g",lw=2,ls="-",label="UV+VIS up")
#ax1.plot(lat[:],means_ctl_up[1,:],color="darkorchid",lw=2,ls="-",label="NIR up")
#ax1.legend(fontsize=8)
#ax1.set_title("SFC Fluxes (CESM2)",fontsize=14)
#ax1.set_ylabel(units,fontsize=14)
#ax1.set_xlim(-90,90)
#ax1.set_ylim(-4,160)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)

ax2=fig.add_axes([0.14,0.15,0.8,0.72])
ax2.plot(lat[:],diffs_dn[0,:],color="k",lw=1,label="\u0394UV+VIS down")
ax2.plot(lat[:],diffs_dn[1,:],color="r",lw=1,label="\u0394NIR down")
ax2.plot(lat[:],diffs_up[0,:],color="g",lw=1,ls="-",label="\u0394UV+VIS up")
ax2.plot(lat[:],diffs_up[1,:],color="darkorchid",lw=1,ls="-",label="\u0394NIR up")
ax2.plot(lat[:],diffs_net[0,:],color="k",lw=2,ls="--",label="\u0394UV+VIS net")
ax2.plot(lat[:],diffs_net[1,:],color="r",lw=2,ls="--",label="\u0394NIR net")
ax2.plot(lat[:],zeros[0,:],color="gray",lw=1)
ax2.legend(fontsize=8)
ax2.set_title("Diff in SFC Flux (TSIS-1 - CESM2, diag)",fontsize=14) #+var_long_name,fontsize=12)
ax2.set_ylabel(units,fontsize=14)
ax2.set_xlabel("Latitude",fontsize=14)
ax2.set_xlim(-90,90)
ax2.set_ylim(-1.6,2.15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# add shading 
collection = collections.BrokenBarHCollection.span_where(lat[:], ymin=-1.6, ymax=2.15, \
             where=means_yby_ctl_fice >0.1,facecolor='y',alpha=0.3)
ax2.add_collection(collection)

plt.savefig("./figures/"+figure_name+".pdf")
plt.savefig("./figures/"+figure_name+".png",dpi=150)
plt.show()

exit()
