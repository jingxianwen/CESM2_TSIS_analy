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

# parameters
from get_parameters import *

# scipy
from scipy import stats


# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"

years=np.arange(2010,2020)
#months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]
var_long_name="Band-by-Band TOA Upward SW"
figure_name="Band_by_Band_TOA_Upward_SW_JJA"
units=r"W/m$^2$"
nlat=96

means_yby_ctl=np.zeros((years.size,3,nlat)) #year by year mean for each variable;
means_yby_exp=np.zeros((years.size,3,nlat)) #year by year mean for each variable
means_ctl=np.zeros((3,nlat)) #multi-year mean for each variable
means_exp=np.zeros((3,nlat)) #multi-year mean for each variable
diffs=np.zeros((3,nlat)) #multi-year exp-ctl diff for each variable
pvals=np.zeros((3,nlat)) #pvalues of ttest
    
for iy in range(0,years.size):    
    # open data file
    fctl=fpath_ctl+ctl_pref+"_JJA_"+str(years[iy])+".nc"
    fexp=fpath_exp+exp_pref+"_JJA_"+str(years[iy])+".nc"
    file_ctl=netcdf_dataset(fctl,"r")
    file_exp=netcdf_dataset(fexp,"r")

    # read lat and lon
    lat=file_ctl.variables["lat"]
    lon=file_ctl.variables["lon"]

    # read required variables
    # variable list: FLUT, FSNTOA, FSNS, FLNS, LHFLX, SHFLX
    #-- TOA
    flut_ctl=file_ctl.variables["FLUT"]
    fsnt_ctl=file_ctl.variables["FSNTOA"]
    flut_exp=file_exp.variables["FLUT"]
    fsnt_exp=file_exp.variables["FSNTOA"]
    #-- Surface
    fsns_ctl=file_ctl.variables["FSNS"]
    flns_ctl=file_ctl.variables["FLNS"]
    lhflx_ctl=file_ctl.variables["LHFLX"]
    shflx_ctl=file_ctl.variables["SHFLX"]
    fsns_exp=file_exp.variables["FSNS"]
    flns_exp=file_exp.variables["FLNS"]
    lhflx_exp=file_exp.variables["LHFLX"]
    shflx_exp=file_exp.variables["SHFLX"]

    # compute zonal mean and change sign 
    # TOA:positive downward
    # Surface: positive upward
    flut_ctl_zm=np.mean(flut_ctl,axis=2)*-1.0
    fsnt_ctl_zm=np.mean(fsnt_ctl,axis=2)
    fsns_ctl_zm=np.mean(fsns_ctl,axis=2)
    flns_ctl_zm=np.mean(flns_ctl,axis=2)*-1.0
    lhflx_ctl_zm=np.mean(lhflx_ctl,axis=2)*-1.0
    shflx_ctl_zm=np.mean(shflx_ctl,axis=2)*-1.0
    flut_exp_zm=np.mean(flut_exp,axis=2)*-1.0
    fsnt_exp_zm=np.mean(fsnt_exp,axis=2)
    fsns_exp_zm=np.mean(fsns_exp,axis=2)
    flns_exp_zm=np.mean(flns_exp,axis=2)*-1.0
    lhflx_exp_zm=np.mean(lhflx_exp,axis=2)*-1.0
    shflx_exp_zm=np.mean(shflx_exp,axis=2)*-1.0
    #print(lhflx_exp_zm)
    # compute NHT(Northward Heat Transport), NHTatm, and NHTnon-atm
    # Kay et al., 2012, JC.
    lat_r=lat[:]/180.*np.pi
    #nlat=np.size(lat_r)
    re=6.371e6  # earth radius in meter

    fntoa_ctl=fsnt_ctl_zm+flut_ctl_zm
    #print(np.mean(flut_ctl_zm))
		    #-np.sum(fsnt_ctl_zm+flut_ctl_zm)/nlat
    fntoa_exp=fsnt_exp_zm+flut_exp_zm
		    #-np.sum(fsnt_exp_zm+flut_exp_zm)/nlat
    fnsfc_ctl=fsns_ctl_zm+flns_ctl_zm+lhflx_ctl_zm+shflx_ctl_zm
		    #-np.sum(fsns_ctl_zm+flns_ctl_zm+lhflx_ctl_zm+shflx_ctl_zm)/nlat
    fnsfc_exp=fsns_exp_zm+flns_exp_zm+lhflx_exp_zm+shflx_exp_zm
		    #-np.sum(fsns_exp_zm+flns_exp_zm+lhflx_exp_zm+shflx_exp_zm)/nlat

   #---------------------------------------------
   # distract the energy imbalance at TOA and surface to guarentee 0 transport at poles.
   # 1. mean imbalance at a unit latitude range
    imbl_toa_ctl=np.sum(fntoa_ctl[0,:]*np.cos(lat_r[:]))*1.0/float(nlat)
    imbl_toa_exp=np.sum(fntoa_exp[0,:]*np.cos(lat_r[:]))*1.0/float(nlat)
    imbl_sfc_ctl=np.sum(fnsfc_ctl[0,:]*np.cos(lat_r[:]))*1.0/float(nlat)
    imbl_sfc_exp=np.sum(fnsfc_exp[0,:]*np.cos(lat_r[:]))*1.0/float(nlat)
   # 2. distract the imbalance from net energy
    fntoa_ctl_w=fntoa_ctl[0,:]*np.cos(lat_r[:])-imbl_toa_ctl
    fntoa_exp_w=fntoa_exp[0,:]*np.cos(lat_r[:])-imbl_toa_exp
    fnsfc_ctl_w=fnsfc_ctl[0,:]*np.cos(lat_r[:])-imbl_sfc_ctl
    fnsfc_exp_w=fnsfc_exp[0,:]*np.cos(lat_r[:])-imbl_sfc_exp
   #---------------------------------------------

    NHT_ctl_tmp1=np.empty((nlat))
    NHTa_ctl_tmp1=np.empty((nlat))
    NHT_exp_tmp1=np.empty((nlat))
    NHTa_exp_tmp1=np.empty((nlat))
    NHT_ctl_tmp2=np.empty((nlat))
    NHTa_ctl_tmp2=np.empty((nlat))
    NHT_exp_tmp2=np.empty((nlat))
    NHTa_exp_tmp2=np.empty((nlat))
    #print(fntoa_exp)
    #exit()
    for il in range(0,nlat):
      # first compute towards north
        NHT_ctl_tmp1[il]=-2*np.pi*re**2*np.sum(fntoa_ctl_w[il:])*1.0/180.*np.pi*1.0e-15
        NHT_exp_tmp1[il]=-2*np.pi*re**2*np.sum(fntoa_exp_w[il:])*1.0/180.*np.pi*1.0e-15
        NHTa_ctl_tmp1[il]=-2*np.pi*re**2*np.sum(fntoa_ctl_w[il:]-fnsfc_ctl_w[il:])*1.0/180.*np.pi*1.0e-15
        NHTa_exp_tmp1[il]=-2*np.pi*re**2*np.sum(fntoa_exp_w[il:]-fnsfc_exp_w[il:])*1.0/180.*np.pi*1.0e-15
    #for il in range(np.int64(nlat/2),nlat):
    #for il in range(0,nlat):
      # #then compute toward south
        #NHT_ctl_tmp2[il]=2*np.pi*re**2*np.sum(fntoa_ctl_w[:il])*1.0/180.*np.pi*1.0e-15
        #NHT_exp_tmp2[il]=2*np.pi*re**2*np.sum(fntoa_exp_w[:il])*1.0/180.*np.pi*1.0e-15
        #NHTa_ctl_tmp2[il]=2*np.pi*re**2*np.sum(fntoa_ctl_w[:il]-fnsfc_ctl_w[:il])*1.0/180.*np.pi*1.0e-15
        #NHTa_exp_tmp2[il]=2*np.pi*re**2*np.sum(fntoa_exp_w[:il]-fnsfc_exp_w[:il])*1.0/180.*np.pi*1.0e-15
     # compute the average
        #NHT_ctl=(NHT_ctl_tmp1+NHT_ctl_tmp2)*0.5
        #NHTa_ctl=(NHTa_ctl_tmp1+NHTa_ctl_tmp2)*0.5
        #NHT_exp=(NHT_exp_tmp1+NHT_exp_tmp2)*0.5
        #NHTa_exp=(NHTa_exp_tmp1+NHTa_exp_tmp2)*0.5

        NHT_ctl=NHT_ctl_tmp1
        NHTa_ctl=NHTa_ctl_tmp1
        NHT_exp=NHT_exp_tmp1
        NHTa_exp=NHTa_exp_tmp1
        NHTna_ctl=NHT_ctl-NHTa_ctl
        NHTna_exp=NHT_exp-NHTa_exp
        
        means_yby_ctl[iy,0,:]=NHT_ctl[:]
        means_yby_ctl[iy,1,:]=NHTa_ctl[:]
        means_yby_ctl[iy,2,:]=NHTna_ctl[:]
        means_yby_exp[iy,0,:]=NHT_exp[:]
        means_yby_exp[iy,1,:]=NHTa_exp[:]
        means_yby_exp[iy,2,:]=NHTna_exp[:]

# compute multi-year mean and ttest
means_ctl=np.mean(means_yby_ctl,axis=0)
means_exp=np.mean(means_yby_exp,axis=0)
diffs=means_exp-means_ctl
ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=ttest.pvalue
print(pvalues)
#print(means_ctl)
#exit()

siglev=0.05
diffs_sig=np.zeros(diffs.shape)
diffs_unsig=np.zeros(diffs.shape)
diffs_sig[:,:]=np.nan
diffs_unsig[:,:]=np.nan

for ih in range(0,3):
   for ip in range(0,nlat):
       if pvalues[ih,ip] < siglev:
           diffs_sig[ih,ip]=diffs[ih,ip]
       else:
           diffs_unsig[ih,ip]=diffs[ih,ip]

### make the plot 
fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.13,0.55,0.78,0.35])
ax2=fig.add_axes([0.13,0.12,0.78,0.35])
line_ctl1=ax1.plot(lat[:],means_ctl[0,:],ls="-",lw=1,c="k",label="Total(CTL)")
line_ctl2=ax1.plot(lat[:],means_ctl[1,:],ls="-",lw=1,c="magenta",label="Atmos(CTL)")
line_ctl3=ax1.plot(lat[:],means_ctl[2,:],ls="-",lw=1,c="aqua",label="Ocean(CTL)")
line_exp1=ax1.plot(lat[:],means_exp[0,:],ls="--",lw=1,c="k",label="Total(TSIS)")
line_exp2=ax1.plot(lat[:],means_exp[1,:],ls="--",lw=1,c="magenta",label="Atmos(TSIS)")
line_exp3=ax1.plot(lat[:],means_exp[2,:],ls="--",lw=1,c="aqua",label="Ocean(TSIS)")

line_dif1=ax2.plot(lat[:],diffs[0,:],ls="-",lw=1,c="k",label="Total")
line_dif2=ax2.plot(lat[:],diffs[1,:],ls="-",lw=1,c="magenta",label="Atmos")
line_dif3=ax2.plot(lat[:],diffs[2,:],ls="-",lw=1,c="aqua",label="Ocean")

line_sig1=ax2.plot(lat[:],diffs_sig[0,:],ls="-",lw=4,c="k")
line_sig2=ax2.plot(lat[:],diffs_sig[1,:],ls="-",lw=4,c="magenta")
line_sig3=ax2.plot(lat[:],diffs_sig[2,:],ls="-",lw=4,c="aqua")

line_zero1=ax1.plot(lat[:],np.zeros((nlat)),ls="-",lw=1,c="gray")
line_zero2=ax2.plot(lat[:],np.zeros((nlat)),ls="-",lw=1,c="gray")

ax1.set_xlim(-90.,90.)
ax2.set_xlim(-90.,90.)
#ax1.set_ylim(-7.9,7.9)
# titles
ax1.set_title("Northward Heat Transport (JJA)",fontsize=12,fontweight='bold')
ax2.set_title("Diff in Northward Heat Transport (JJA)",fontsize=12,fontweight='bold')
#ax1.set_xlabel("Latitude",fontsize=12,fontweight='bold')
ax2.set_xlabel("Latitude",fontsize=12,fontweight='bold')
ax1.set_ylabel("(PW)",fontsize=12,fontweight='bold')
ax2.set_ylabel("(PW)",fontsize=12,fontweight='bold')
# ticks
xtlocs=np.int64(ax1.xaxis.get_ticklocs())
ytlocs=np.int64(ax1.yaxis.get_ticklocs())
ax1.set_xticklabels(xtlocs,fontsize=12,fontweight='bold')
ax1.set_yticklabels(ytlocs,fontsize=12,fontweight='bold')
ytlocs=np.float32(ax2.yaxis.get_ticklocs())
ax2.set_xticklabels(xtlocs,fontsize=12,fontweight='bold')
ax2.set_yticklabels(ytlocs,fontsize=12,fontweight='bold')
# legend
ax1.legend(loc="best",fontsize="small")
ax2.legend(loc="best",fontsize="small")
# adjust panel layout
#fig.subplots_adjust(hspace=0.2)
#save figure as file
#if os.environ["fig_save"]=="True":
#    fname="d3_northw_energy_transport_"+season+"."+os.environ["fig_suffix"]
plt.savefig("NHT_JJA.jpg",dpi=100)
#if os.environ["fig_show"]=="True":
plt.show()
plt.close()

#===============================

#def get_parameters(varnm,season):
#    #list_rad=["FLUT","FLUTC","FLNT","FLNTC","FSNT","FSNTC","FSDS","FSDSC","FSNS","FSNSC"]
#    if varnm == "FLUT":
#        parameters={"units":"W/m2",\
#		   "contour_levs":[120, 140, 160, 180, 200, 220, 240, 260, 280, 300],\
#		   "diff_levs":[-50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50],\
#                   "colormap":"PiYG_r",\
#                   "colormap_diff":"bwr"\
#		   }
#    return parameters
#
#def get_area_mean_range(varnm,lat):
#   # 1. area weighted average 
#    #convert latitude to radians
#    latr=np.deg2rad(lat)
#    #use cosine of latitudes as weights for the mean
#    weights=np.cos(latr)
#    #first calculate zonal mean
#    zonal_mean=varnm.mean(axis=2)
#    #then calculate weighted global mean
#    area_mean=np.average(zonal_mean,axis=1,weights=weights)
#   # 2. min and max
#    minval=varnm.min()
#    maxval=varnm.max()
#    return area_mean,minval,maxval
