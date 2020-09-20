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
import matplotlib.collections as collections

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
figure_name="zonal_cldfrac_shaded"
units="%"
#units=r"kg m$^-$$^2$"
#units2=r"g m$^-$$^2$"

#varnms_vis_cldhgh=np.array(["FSSDS13","FSSDS12","FSSDS11","FSSDS10","FSSDS09"])
#varnms_nir_cldhgh=np.array(["FSSDS08","FSSDS07","FSSDS06","FSSDS05","FSSDS04",\
#                     "FSSDS03","FSSDS02","FSSDS01","FSSDS14"])
#
#varnms_vis_cldhgh=np.array(["FSSUS13","FSSUS12","FSSUS11","FSSUS10","FSSUS09"])
#varnms_nir_cldhgh=np.array(["FSSUS08","FSSUS07","FSSUS06","FSSUS05","FSSUS04",\
#                     "FSSUS03","FSSUS02","FSSUS01","FSSUS14"])

#f1=fpath_ctl+"solar_TSIS_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
#f2=fpath_exp+"tsis_ctl_cesm211_standard-ETEST-f19_g17-ens1.cam.h0.0001-01.nc"
nlat=np.int64(96)
means_yby_ctl_cldhgh=np.zeros((years.size,nlat)) #year by year mean for each variable
means_yby_exp_cldhgh=np.zeros((years.size,nlat)) #year by year mean for each variable
means_ctl_cldhgh=np.zeros((nlat)) #multi-year mean for each variable
means_exp_cldhgh=np.zeros((nlat)) #multi-year mean for each variable
diffs_cldhgh=np.zeros((nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_cldhgh=np.zeros((years.size)) #year by year mean for each variable
gm_yby_exp_cldhgh=np.zeros((years.size)) #year by year mean for each variable

means_yby_ctl_cldmed=np.zeros((years.size,nlat)) #year by year mean for each variable
means_yby_exp_cldmed=np.zeros((years.size,nlat)) #year by year mean for each variable
means_ctl_cldmed=np.zeros((nlat)) #multi-year mean for each variable
means_exp_cldmed=np.zeros((nlat)) #multi-year mean for each variable
diffs_cldmed=np.zeros((nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_cldmed=np.zeros((years.size)) #year by year mean for each variable
gm_yby_exp_cldmed=np.zeros((years.size)) #year by year mean for each variable

means_yby_ctl_cldlow=np.zeros((years.size,nlat)) #year by year mean for each variable
means_yby_exp_cldlow=np.zeros((years.size,nlat)) #year by year mean for each variable
means_ctl_cldlow=np.zeros((nlat)) #multi-year mean for each variable
means_exp_cldlow=np.zeros((nlat)) #multi-year mean for each variable
diffs_cldlow=np.zeros((nlat)) #multi-year exp-ctl diff for each variable
gm_yby_ctl_cldlow=np.zeros((years.size)) #year by year mean for each variable
gm_yby_exp_cldlow=np.zeros((years.size)) #year by year mean for each variable

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
    vn="CLDHGH"
    dtctl_cldhgh=file_ctl.variables[vn][0,:,:]*100.
    dtexp_cldhgh=file_exp.variables[vn][0,:,:]*100. 
    means_yby_ctl_cldhgh[iy,:]= means_yby_ctl_cldhgh[iy,:] + np.mean(dtctl_cldhgh[:,:],axis=1)
    means_yby_exp_cldhgh[iy,:]= means_yby_exp_cldhgh[iy,:] + np.mean(dtexp_cldhgh[:,:],axis=1)
    gm_yby_ctl_cldhgh[iy]=gm_yby_ctl_cldhgh[iy]+get_area_mean_min_max(dtctl_cldhgh[:,:],lat[:])[0]
    gm_yby_exp_cldhgh[iy]=gm_yby_exp_cldhgh[iy]+get_area_mean_min_max(dtexp_cldhgh[:,:],lat[:])[0]

    vn="CLDMED"
    dtctl_cldmed=file_ctl.variables[vn][0,:,:]*100.
    dtexp_cldmed=file_exp.variables[vn][0,:,:]*100.
    means_yby_ctl_cldmed[iy,:]= means_yby_ctl_cldmed[iy,:] + np.mean(dtctl_cldmed[:,:],axis=1)
    means_yby_exp_cldmed[iy,:]= means_yby_exp_cldmed[iy,:] + np.mean(dtexp_cldmed[:,:],axis=1)
    gm_yby_ctl_cldmed[iy]=gm_yby_ctl_cldmed[iy]+get_area_mean_min_max(dtctl_cldmed[:,:],lat[:])[0]
    gm_yby_exp_cldmed[iy]=gm_yby_exp_cldmed[iy]+get_area_mean_min_max(dtexp_cldmed[:,:],lat[:])[0]

    vn="CLDLOW"
    dtctl_cldlow=file_ctl.variables[vn][0,:,:]*100.
    dtexp_cldlow=file_exp.variables[vn][0,:,:]*100. 
    means_yby_ctl_cldlow[iy,:]= means_yby_ctl_cldlow[iy,:] + np.mean(dtctl_cldlow[:,:],axis=1)
    means_yby_exp_cldlow[iy,:]= means_yby_exp_cldlow[iy,:] + np.mean(dtexp_cldlow[:,:],axis=1)
    gm_yby_ctl_cldlow[iy]=gm_yby_ctl_cldlow[iy]+get_area_mean_min_max(dtctl_cldlow[:,:],lat[:])[0]
    gm_yby_exp_cldlow[iy]=gm_yby_exp_cldlow[iy]+get_area_mean_min_max(dtexp_cldlow[:,:],lat[:])[0]

    #vn="TGCLDCWP"
    #dtctl_lwp=file_ctl.variables[vn][0,:,:]*1000.
    #dtexp_lwp=file_exp.variables[vn][0,:,:]*1000. 
    #means_yby_ctl_lwp[iy,:]= means_yby_ctl_lwp[iy,:] + np.mean(dtctl_lwp[:,:],axis=1)
    #means_yby_exp_lwp[iy,:]= means_yby_exp_lwp[iy,:] + np.mean(dtexp_lwp[:,:],axis=1)
    #gm_yby_ctl_lwp[iy]=gm_yby_ctl_lwp[iy]+get_area_mean_min_max(dtctl_lwp[:,:],lat[:])[0]
    #gm_yby_exp_lwp[iy]=gm_yby_exp_lwp[iy]+get_area_mean_min_max(dtexp_lwp[:,:],lat[:])[0]


#print(np.mean(gm_yby_ctl_cldhgh,axis=1))
#print(np.mean(gm_yby_ctl_cldhgh,axis=1))
#print(np.mean(gm_yby_ctl_net,axis=1))
#print(np.mean(gm_yby_exp_cldhgh,axis=1))
#print(np.mean(gm_yby_exp_cldhgh,axis=1))
#print(np.mean(gm_yby_exp_net,axis=1))
#exit()

# compute globam mean
#diffs_ym_toa=np.mean(gm_yby_exp-gm_yby_ctl)
#ttest_ym_toa=stats.ttest_ind(gm_yby_ctl,gm_yby_exp,axis=0)
#print(diffs_ym_toa)
#print(ttest_ym_toa.pvalue)
#exit()

# compute multi-year mean and ttest
siglev=0.05

means_ctl_cldhgh=np.mean(means_yby_ctl_cldhgh,axis=0)
means_exp_cldhgh=np.mean(means_yby_exp_cldhgh,axis=0)
diffs_cldhgh=means_exp_cldhgh-means_ctl_cldhgh
ttest=stats.ttest_ind(means_yby_ctl_cldhgh,means_yby_exp_cldhgh,axis=0)
pvalues_cldhgh=ttest.pvalue
diffs_sig_cldhgh=np.zeros(diffs_cldhgh.shape)
diffs_sig_cldhgh[:]=np.nan

means_ctl_cldmed=np.mean(means_yby_ctl_cldmed,axis=0)
means_exp_cldmed=np.mean(means_yby_exp_cldmed,axis=0)
diffs_cldmed=means_exp_cldmed-means_ctl_cldmed
ttest=stats.ttest_ind(means_yby_ctl_cldmed,means_yby_exp_cldmed,axis=0)
pvalues_cldmed=ttest.pvalue
diffs_sig_cldmed=np.zeros(diffs_cldmed.shape)
diffs_sig_cldmed[:]=np.nan

means_ctl_cldlow=np.mean(means_yby_ctl_cldlow,axis=0)
means_exp_cldlow=np.mean(means_yby_exp_cldlow,axis=0)
diffs_cldlow=means_exp_cldlow-means_ctl_cldlow
ttest=stats.ttest_ind(means_yby_ctl_cldlow,means_yby_exp_cldlow,axis=0)
pvalues_cldlow=ttest.pvalue
diffs_sig_cldlow=np.zeros(diffs_cldlow.shape)
diffs_sig_cldlow[:]=np.nan

#means_ctl_lwp=np.mean(means_yby_ctl_lwp,axis=0)
#means_exp_lwp=np.mean(means_yby_exp_lwp,axis=0)
#diffs_lwp=means_exp_lwp-means_ctl_lwp
#ttest=stats.ttest_ind(means_yby_ctl_lwp,means_yby_exp_lwp,axis=0)
#pvalues_lwp=ttest.pvalue
#diffs_sig_lwp=np.zeros(diffs_lwp.shape)
#diffs_sig_lwp[:]=np.nan

means_exp_fice=np.mean(means_yby_exp_fice,axis=0)

zeros=np.zeros(diffs_cldhgh.shape)

#print(diffs_sig.size)

for iv in range(pvalues_cldhgh.shape[0]):
   #for ip in range(pvalues_cldhgh.shape[1]):
   if pvalues_cldhgh[iv] < siglev:
       diffs_sig_cldhgh[iv]=diffs_cldhgh[iv]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

for iv in range(pvalues_cldmed.shape[0]):
   #for ip in range(pvalues_cldmed.shape[1]):
   if pvalues_cldmed[iv] < siglev:
       diffs_sig_cldmed[iv]=diffs_cldmed[iv]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

for iv in range(pvalues_cldlow.shape[0]):
   #for ip in range(pvalues_cldlow.shape[1]):
   if pvalues_cldlow[iv] < siglev:
       diffs_sig_cldlow[iv]=diffs_cldlow[iv]
       #else:
       #    diffs_unsig[iv,ip]=diffs[iv,ip]

#for iv in range(pvalues_lwp.shape[0]):
#   #for ip in range(pvalues_lwp.shape[1]):
#   if pvalues_lwp[iv] < siglev:
#       diffs_sig_lwp[iv]=diffs_lwp[iv]
#       #else:
#       #    diffs_unsig[iv,ip]=diffs[iv,ip]


# make the plot
fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.14,0.58,0.7,0.36])

ax1.plot(lat[:],means_ctl_cldhgh[:],color="k",lw=2,ls="-" ,label="CFHGH")
ax1.plot(lat[:],means_ctl_cldmed[:],color="b",lw=2,ls="-" ,label="CFMED")
ax1.plot(lat[:],means_ctl_cldlow[:],color="r",lw=2,ls="-" ,label="CFLOW")
#ax1.plot(lat[:],means_exp_cldhgh[:],color="g",lw=2,ls="-",label="NIR down")
#ax1.plot(lat[:],means_ctl_DJF[0,:],color="royalblue",lw=2,ls="-",label="DJF")
#ax1.plot(lat[:],means_ctl_JJA[0,:],color="darkorange",lw=2,ls="-",label="JJA")
#ax1.plot(lat[:],means_exp[0,:],color="k",lw=2,ls=":",label="TSIS")
#ax1.legend(loc="upper left",fontsize=12)
ax1.legend(fontsize=12)
ax1.set_title("Cloud Fraction (CESM2)",fontsize=14)
ax1.set_ylabel(""+units+"",fontsize=14)
#ax1.set_xlabel("Latitude",fontsize=14)
ax1.set_xlim(-90,90)
#ax1.set_ylim(-4,160)
ax1.tick_params(axis="y",labelcolor="k")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#ax1_r=ax1.twinx()
#ax1_r.set_ylabel("CWP ("+units2+")",fontsize=14,c="r")
#ax1_r.plot(lat[:],means_ctl_lwp[:],color="r",lw=2,ls="-" ) #,label="CWP")
#ax1_r.tick_params(axis="y",labelcolor="r")
##ax1_r.set_ylim(-0.6,0.6)

#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)

ax2=fig.add_axes([0.14,0.12,0.7,0.36])
ax2.plot(lat[:],diffs_sig_cldhgh[:],color="k",lw=4,alpha=1.0)
ax2.plot(lat[:],diffs_sig_cldmed[:],color="b",lw=4,alpha=1.0)
ax2.plot(lat[:],diffs_sig_cldlow[:],color="r",lw=4,alpha=1.0)
#ax2.plot(lat[:],diffs_sig_cldhgh[:],color="peachpuff",lw=6,alpha=0.9)
#ax2.plot(lat[:],diffs_sig_cldhgh[0,:],color="yellowgreen",lw=6,alpha=0.9)
#ax2.plot(lat[:],diffs_sig_cldhgh[1,:],color="plum",lw=6,alpha=0.9)
#ax2.plot(lat[:],diffs_sig_net[0,:],color="blue",lw=6,alpha=0.7)
#ax2.plot(lat[:],diffs_sig_net[1,:],color="orange",lw=6,alpha=0.9)

ax2.plot(lat[:],diffs_cldhgh[:],color="k",lw=1 ,label="\u0394CFHGH")
ax2.plot(lat[:],diffs_cldmed[:],color="b",lw=1 ,label="\u0394CFMED")
ax2.plot(lat[:],diffs_cldlow[:],color="r",lw=1 ,label="\u0394CFLOW")
#ax2.plot(lat[:],diffs_cldhgh[1,:],color="r",lw=1,label="\u0394NIR down")
#ax2.plot(lat[:],diffs_cldhgh[0,:],color="g",lw=1,ls="-",label="\u0394UV+VIS up")
#ax2.plot(lat[:],diffs_cldhgh[1,:],color="darkorchid",lw=1,ls="-",label="\u0394NIR up")
#ax2.plot(lat[:],diffs_net[0,:],color="k",lw=2,ls="--",label="\u0394UV+VIS net")
#ax2.plot(lat[:],diffs_net[1,:],color="r",lw=2,ls="--",label="\u0394NIR net")
ax2.plot(lat[:],zeros[:],color="gray",lw=1)
ax2.legend(fontsize=12)
ax2.set_title("Differences (TSIS-1 - CESM2)",fontsize=14) #+var_long_name,fontsize=12)
ax2.set_xlabel("Latitude",fontsize=14)
#ax2.set_ylabel("\u0394TPW ("+units+")",fontsize=14)
ax2.set_ylabel(""+units+"",fontsize=14)
ax2.set_xlim(-90,90)
#ax2.set_ylim(-0.6,0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# add shading 
collection = collections.BrokenBarHCollection.span_where(lat[:], ymin=-1.6, ymax=2.15, \
             where=means_exp_fice >0.1,facecolor='y',alpha=0.3)
ax2.add_collection(collection)

#ax2_r=ax2.twinx()
#ax2_r.set_ylabel("\u0394CWP ("+units2+")",fontsize=14,c="r")
#ax2_r.plot(lat[:],diffs_sig_lwp[:],color="r",lw=4,alpha=1.0)
#ax2_r.plot(lat[:],diffs_lwp[:],color="r",lw=1,ls="-" ) #,label="\u0394CWP")
#ax2_r.tick_params(axis="y",labelcolor="r")
#ax2_r.set_ylim(-2.5,2.5)

#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)

plt.savefig(figure_name+".pdf")
plt.show()

exit()
