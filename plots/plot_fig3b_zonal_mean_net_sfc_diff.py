#=====================================================
#
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
#   start here
#----------------

# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
ctl_pref_diag="solar_CTL_cesm211_ETEST-f19_g17-ens0_5days"
exp_pref_diag="solar_TSIS_cesm211_ETEST-f19_g17-ens0_5days"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"
 
fpath_ctl_diag="/raid00/xianwen/cesm211_solar/"+ctl_pref_diag+"/"
fpath_exp_diag="/raid00/xianwen/cesm211_solar/"+exp_pref_diag+"/"

years=np.arange(2010,2020) 
#months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

figure_name="fig3b_zonal_sfc_net_ANN_shaded"
units=r"Wm$^-$$^2$"

varnms_vis_dn=np.array(["FSSDS13","FSSDS12","FSSDS11","FSSDS10","FSSDS09"])
varnms_nir_dn=np.array(["FSSDS08","FSSDS07","FSSDS06","FSSDS05","FSSDS04",\
                     "FSSDS03","FSSDS02","FSSDS01","FSSDS14"])

varnms_vis_up=np.array(["FSSUS13","FSSUS12","FSSUS11","FSSUS10","FSSUS09"])
varnms_nir_up=np.array(["FSSUS08","FSSUS07","FSSUS06","FSSUS05","FSSUS04",\
                     "FSSUS03","FSSUS02","FSSUS01","FSSUS14"])

nlat=np.int64(96)
means_yby_ctl_dn=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_yby_exp_dn=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_ctl_dn=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_dn=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_dn=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_dn=np.zeros((2,years.size)) #year by year mean for each variable
gm_yby_exp_dn=np.zeros((2,years.size)) #year by year mean for each variable

means_yby_ctl_up=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_yby_exp_up=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_ctl_up=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_up=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_up=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_up=np.zeros((2,years.size)) #year by year mean for each variable
gm_yby_exp_up=np.zeros((2,years.size)) #year by year mean for each variable

means_yby_ctl_net=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_yby_exp_net=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_ctl_net=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_net=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_net=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_net=np.zeros((2,years.size)) #year by year mean for each variable
gm_yby_exp_net=np.zeros((2,years.size)) #year by year mean for each variable

means_yby_ctl_dn_diag=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_yby_exp_dn_diag=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_ctl_dn_diag=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_dn_diag=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_dn_dn_diag=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_dn_diag=np.zeros((2,years.size)) #year by year mean for each variable
gm_yby_exp_dn_diag=np.zeros((2,years.size)) #year by year mean for each variable

means_yby_ctl_up_diag=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_yby_exp_up_diag=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_ctl_up_diag=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_up_diag=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_up_diag=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_up_diag=np.zeros((2,years.size)) #year by year mean for each variable
gm_yby_exp_up_diag=np.zeros((2,years.size)) #year by year mean for each variable

means_yby_ctl_net_diag=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_yby_exp_net_diag=np.zeros((years.size,2,nlat)) #year by year mean for each variable
means_ctl_net_diag=np.zeros((2,nlat)) #multi-year mean for each variable
means_exp_net_diag=np.zeros((2,nlat)) #multi-year mean for each variable
diffs_net_diag=np.zeros((2,nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_net_diag=np.zeros((2,years.size)) #year by year mean for each variable
gm_yby_exp_net_diag=np.zeros((2,years.size)) #year by year mean for each variable

means_yby_exp_fice=np.zeros((years.size,nlat)) #year by year mean for each variable
means_exp_fice=np.zeros((nlat)) #multi-year mean for each variable

for iy in range(0,years.size): 
    # open data file
    fctl=fpath_ctl+ctl_pref+"_ANN_"+str(years[iy])+".nc"
    fexp=fpath_exp+exp_pref+"_ANN_"+str(years[iy])+".nc"
    file_ctl=netcdf_dataset(fctl,"r")
    file_exp=netcdf_dataset(fexp,"r")

    fctl_diag=fpath_ctl_diag+ctl_pref_diag+"_climo_ANN.nc"
    fexp_diag=fpath_exp_diag+exp_pref_diag+"_climo_ANN.nc"
    file_ctl_diag=netcdf_dataset(fctl_diag,"r")
    file_exp_diag=netcdf_dataset(fexp_diag,"r")

    # read lat and lon
    lat=file_ctl.variables["lat"]
    lon=file_ctl.variables["lon"]
 
    means_yby_exp_fice[iy,:]=means_yby_exp_fice[iy,:]+np.mean(file_exp.variables["ICEFRAC"][0,:,:],axis=1)   

    # read data and calculate mean/min/max
    for vn in varnms_vis_dn:
        dtctl_dn=file_ctl.variables[vn][0,:,:]
        dtexp_dn=file_exp.variables[vn][0,:,:] 
        means_yby_ctl_dn[iy,0,:]= means_yby_ctl_dn[iy,0,:] + np.mean(dtctl_dn[:,:],axis=1)
        means_yby_exp_dn[iy,0,:]= means_yby_exp_dn[iy,0,:] + np.mean(dtexp_dn[:,:],axis=1)
        gm_yby_ctl_dn[0,iy]=gm_yby_ctl_dn[0,iy]+get_area_mean_min_max(dtctl_dn[:,:],lat[:])[0]
        gm_yby_exp_dn[0,iy]=gm_yby_exp_dn[0,iy]+get_area_mean_min_max(dtexp_dn[:,:],lat[:])[0]

        dtctl_dn_diag=file_ctl_diag.variables[vn][0,:,:]
        dtexp_dn_diag=file_exp_diag.variables[vn][0,:,:] 
        means_yby_ctl_dn_diag[iy,0,:]= means_yby_ctl_dn_diag[iy,0,:] + \
                                       np.mean(dtctl_dn_diag[:,:],axis=1)
        means_yby_exp_dn_diag[iy,0,:]= means_yby_exp_dn_diag[iy,0,:] + \
                                       np.mean(dtexp_dn_diag[:,:],axis=1)
        gm_yby_ctl_dn_diag[0,iy]=gm_yby_ctl_dn_diag[0,iy] + \
                                 get_area_mean_min_max(dtctl_dn_diag[:,:],lat[:])[0]
        gm_yby_exp_dn_diag[0,iy]=gm_yby_exp_dn_diag[0,iy] + \
                                 get_area_mean_min_max(dtexp_dn_diag[:,:],lat[:])[0]

    for vn in varnms_nir_dn:
        dtctl_dn=file_ctl.variables[vn][0,:,:]
        dtexp_dn=file_exp.variables[vn][0,:,:] 
        means_yby_ctl_dn[iy,1,:] = means_yby_ctl_dn[iy,1,:] + \
                                   np.mean(dtctl_dn[:,:],axis=1)
        means_yby_exp_dn[iy,1,:] = means_yby_exp_dn[iy,1,:] + \
                                   np.mean(dtexp_dn[:,:],axis=1)
        gm_yby_ctl_dn[1,iy]=gm_yby_ctl_dn[1,iy] + get_area_mean_min_max(dtctl_dn[:,:],lat[:])[0]
        gm_yby_exp_dn[1,iy]=gm_yby_exp_dn[1,iy] + get_area_mean_min_max(dtexp_dn[:,:],lat[:])[0]
 
        dtctl_dn_diag=file_ctl_diag.variables[vn][0,:,:]
        dtexp_dn_diag=file_exp_diag.variables[vn][0,:,:] 
        means_yby_ctl_dn_diag[iy,1,:]= means_yby_ctl_dn_diag[iy,1,:] + \
                                       np.mean(dtctl_dn_diag[:,:],axis=1)
        means_yby_exp_dn_diag[iy,1,:]= means_yby_exp_dn_diag[iy,1,:] + \
                                       np.mean(dtexp_dn_diag[:,:],axis=1)
        gm_yby_ctl_dn_diag[1,iy]=gm_yby_ctl_dn_diag[1,iy] + \
                                 get_area_mean_min_max(dtctl_dn_diag[:,:],lat[:])[0]
        gm_yby_exp_dn_diag[1,iy]=gm_yby_exp_dn_diag[1,iy] + \
                                 get_area_mean_min_max(dtexp_dn_diag[:,:],lat[:])[0]

    for vn in varnms_vis_up:
        dtctl_up=file_ctl.variables[vn][0,:,:]
        dtexp_up=file_exp.variables[vn][0,:,:] 
        means_yby_ctl_up[iy,0,:]= means_yby_ctl_up[iy,0,:] + np.mean(dtctl_up[:,:],axis=1)
        means_yby_exp_up[iy,0,:]= means_yby_exp_up[iy,0,:] + np.mean(dtexp_up[:,:],axis=1)
        gm_yby_ctl_up[0,iy]=gm_yby_ctl_up[0,iy]+get_area_mean_min_max(dtctl_up[:,:],lat[:])[0]
        gm_yby_exp_up[0,iy]=gm_yby_exp_up[0,iy]+get_area_mean_min_max(dtexp_up[:,:],lat[:])[0]

        dtctl_up_diag=file_ctl_diag.variables[vn][0,:,:]
        dtexp_up_diag=file_exp_diag.variables[vn][0,:,:] 
        means_yby_ctl_up_diag[iy,0,:]= means_yby_ctl_up_diag[iy,0,:] + \
                                       np.mean(dtctl_up_diag[:,:],axis=1)
        means_yby_exp_up_diag[iy,0,:]= means_yby_exp_up_diag[iy,0,:] + \
                                       np.mean(dtexp_up_diag[:,:],axis=1)
        gm_yby_ctl_up_diag[0,iy]=gm_yby_ctl_up_diag[0,iy] + \
                                 get_area_mean_min_max(dtctl_up_diag[:,:],lat[:])[0]
        gm_yby_exp_up_diag[0,iy]=gm_yby_exp_up_diag[0,iy] + \
                                 get_area_mean_min_max(dtexp_up_diag[:,:],lat[:])[0]

    for vn in varnms_nir_up:
        dtctl_up=file_ctl.variables[vn][0,:,:]
        dtexp_up=file_exp.variables[vn][0,:,:] 
        means_yby_ctl_up[iy,1,:]= means_yby_ctl_up[iy,1,:] + \
                                  np.mean(dtctl_up[:,:],axis=1)
        means_yby_exp_up[iy,1,:]= means_yby_exp_up[iy,1,:] + \
                                  np.mean(dtexp_up[:,:],axis=1)
        gm_yby_ctl_up[1,iy]=gm_yby_ctl_up[1,iy] + \
                            get_area_mean_min_max(dtctl_up[:,:],lat[:])[0]
        gm_yby_exp_up[1,iy]=gm_yby_exp_up[1,iy] + \
                            get_area_mean_min_max(dtexp_up[:,:],lat[:])[0]

        dtctl_up_diag=file_ctl_diag.variables[vn][0,:,:]
        dtexp_up_diag=file_exp_diag.variables[vn][0,:,:] 
        means_yby_ctl_up_diag[iy,1,:]= means_yby_ctl_up_diag[iy,1,:] + \
                                       np.mean(dtctl_up_diag[:,:],axis=1)
        means_yby_exp_up_diag[iy,1,:]= means_yby_exp_up_diag[iy,1,:] + \
                                       np.mean(dtexp_up_diag[:,:],axis=1)
        gm_yby_ctl_up_diag[1,iy]=gm_yby_ctl_up_diag[1,iy] + \
                                 get_area_mean_min_max(dtctl_up_diag[:,:],lat[:])[0]
        gm_yby_exp_up_diag[1,iy]=gm_yby_exp_up_diag[1,iy] + \
                                 get_area_mean_min_max(dtexp_up_diag[:,:],lat[:])[0]

    means_yby_ctl_net[iy,:,:]=means_yby_ctl_dn[iy,:,:]-means_yby_ctl_up[iy,:,:]
    means_yby_exp_net[iy,:,:]=means_yby_exp_dn[iy,:,:]-means_yby_exp_up[iy,:,:]
    gm_yby_ctl_net[:,iy]=gm_yby_ctl_dn[:,iy]-gm_yby_ctl_up[:,iy]
    gm_yby_exp_net[:,iy]=gm_yby_exp_dn[:,iy]-gm_yby_exp_up[:,iy]

    means_yby_ctl_net_diag[iy,:,:]=means_yby_ctl_dn_diag[iy,:,:]-means_yby_ctl_up_diag[iy,:,:]
    means_yby_exp_net_diag[iy,:,:]=means_yby_exp_dn_diag[iy,:,:]-means_yby_exp_up_diag[iy,:,:]
    gm_yby_ctl_net_diag[:,iy]=gm_yby_ctl_dn_diag[:,iy]-gm_yby_ctl_up_diag[:,iy]
    gm_yby_exp_net_diag[:,iy]=gm_yby_exp_dn_diag[:,iy]-gm_yby_exp_up_diag[:,iy]

# compute multi-year mean and ttest
siglev=0.05

means_ctl_dn=np.mean(means_yby_ctl_dn,axis=0)
means_exp_dn=np.mean(means_yby_exp_dn,axis=0)
diffs_dn=means_exp_dn-means_ctl_dn
ttest=stats.ttest_ind(means_yby_ctl_dn,means_yby_exp_dn,axis=0)
pvalues_dn=ttest.pvalue
diffs_sig_dn=np.zeros(diffs_dn.shape)
diffs_sig_dn[:,:]=np.nan

means_ctl_up=np.mean(means_yby_ctl_up,axis=0)
means_exp_up=np.mean(means_yby_exp_up,axis=0)
diffs_up=means_exp_up-means_ctl_up
ttest=stats.ttest_ind(means_yby_ctl_up,means_yby_exp_up,axis=0)
pvalues_up=ttest.pvalue
diffs_sig_up=np.zeros(diffs_up.shape)
diffs_sig_up[:,:]=np.nan

means_ctl_net=np.mean(means_yby_ctl_net,axis=0)
means_exp_net=np.mean(means_yby_exp_net,axis=0)
diffs_net=means_exp_net-means_ctl_net
ttest=stats.ttest_ind(means_yby_ctl_net,means_yby_exp_net,axis=0)
pvalues_net=ttest.pvalue
diffs_sig_net=np.zeros(diffs_net.shape)
diffs_sig_net[:,:]=np.nan

means_ctl_net_diag=np.mean(means_yby_ctl_net_diag,axis=0)
means_exp_net_diag=np.mean(means_yby_exp_net_diag,axis=0)
diffs_net_diag=means_exp_net_diag-means_ctl_net_diag
ttest=stats.ttest_ind(means_yby_ctl_net_diag,means_yby_exp_net_diag,axis=0)
pvalues_net_diag=ttest.pvalue
diffs_sig_net_diag=np.zeros(diffs_net_diag.shape)
diffs_sig_net_diag[:,:]=np.nan

means_exp_fice=np.mean(means_yby_exp_fice,axis=0)

zeros=np.zeros(diffs_dn.shape)

#print(diffs_sig.size)

for iv in range(pvalues_up.shape[0]):
   for ip in range(pvalues_up.shape[1]):
       if pvalues_up[iv,ip] < siglev:
           diffs_sig_up[iv,ip]=diffs_up[iv,ip]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

for iv in range(pvalues_dn.shape[0]):
   for ip in range(pvalues_dn.shape[1]):
       if pvalues_dn[iv,ip] < siglev:
           diffs_sig_dn[iv,ip]=diffs_dn[iv,ip]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

for iv in range(pvalues_net.shape[0]):
   for ip in range(pvalues_net.shape[1]):
       if pvalues_net[iv,ip] < siglev:
           diffs_sig_net[iv,ip]=diffs_net[iv,ip]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

for iv in range(pvalues_net_diag.shape[0]):
   for ip in range(pvalues_net_diag.shape[1]):
       if pvalues_net_diag[iv,ip] < siglev:
           diffs_sig_net_diag[iv,ip]=diffs_net_diag[iv,ip]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]


#------------------
# make the plot
#------------------

fig=plt.figure(figsize=(7,4))

#ax1.plot(lat[:],means_ctl_dn[0,:],color="k",lw=2,ls="-",label="UV+VIS down")
#ax1.plot(lat[:],means_ctl_dn[1,:],color="r",lw=2,ls="-",label="NIR down")
#ax1.plot(lat[:],means_ctl_up[0,:],color="g",lw=2,ls="-",label="UV+VIS up")
#ax1.plot(lat[:],means_ctl_up[1,:],color="darkorchid",lw=2,ls="-",label="NIR up")
##ax1.plot(lat[:],means_ctl_DJF[0,:],color="royalblue",lw=2,ls="-",label="DJF")
##ax1.plot(lat[:],means_ctl_JJA[0,:],color="darkorange",lw=2,ls="-",label="JJA")
##ax1.plot(lat[:],means_exp[0,:],color="k",lw=2,ls=":",label="TSIS")
##ax1.legend(loc="upper left",fontsize=12)
#ax1.legend(fontsize=8)
#ax1.set_title("SFC Fluxes (CESM2)",fontsize=14)
#ax1.set_ylabel(units,fontsize=14)
##ax1.set_xlabel("Latitude",fontsize=14)
#ax1.set_xlim(-90,90)
#ax1.set_ylim(-4,160)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)

ax2=fig.add_axes([0.14,0.15,0.8,0.72])
ax2.plot(lat[:],diffs_sig_net[:,:].sum(axis=0),color="r",lw=6,alpha=0.6)
ax2.plot(lat[:],diffs_net[:,:].sum(axis=0),color="r",lw=2,ls="-",label="\u0394SW net")
ax2.plot(lat[:],diffs_net_diag[:,:].sum(axis=0),color="k",lw=2,ls="-",label="\u0394SW net (diag)")
ax2.plot(lat[:],zeros[0,:],color="lightgray",lw=1)
ax2.legend(fontsize=12)
ax2.set_title("Diff in SFC net Flux (TSIS-1 - CESM2)",fontsize=14) #+var_long_name,fontsize=12)
ax2.set_ylabel(units,fontsize=14)
ax2.set_xlabel("Latitude",fontsize=14)
ax2.set_xlim(-90,90)
ax2.set_ylim(-1.6,1.0)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# add shading 
collection = collections.BrokenBarHCollection.span_where(lat[:], ymin=-1.6, ymax=2.15, \
             where=means_exp_fice >0.1,facecolor='y',alpha=0.3)
ax2.add_collection(collection)

plt.savefig("./figures/"+figure_name+".png",dpi=150)
plt.savefig("./figures/"+figure_name+".pdf")
plt.show()

exit()
