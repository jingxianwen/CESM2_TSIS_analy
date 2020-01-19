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

def northw_energy_transport_model_vs_model(season):
    # data path
    ctl_name=os.environ["ctl_name"]
    exp_name=os.environ["exp_name"]
    fpath_ctl=os.environ["fpath_ctl"]+"/"+os.environ["ctl_run_id"]+"_climo_"+season+".nc"
    fpath_exp=os.environ["fpath_exp"]+"/"+os.environ["exp_run_id"]+"_climo_"+season+".nc"
    
    # open data file
    file_ctl=netcdf_dataset(fpath_ctl,"r")
    file_exp=netcdf_dataset(fpath_exp,"r")

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
    nlat=np.size(lat_r)
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
    imbl_toa_ctl=np.sum(fntoa_ctl[0,:]*np.cos(lat_r[:]))*1.0/180.
    imbl_toa_exp=np.sum(fntoa_exp[0,:]*np.cos(lat_r[:]))*1.0/180.
    imbl_sfc_ctl=np.sum(fnsfc_ctl[0,:]*np.cos(lat_r[:]))*1.0/180.
    imbl_sfc_exp=np.sum(fnsfc_exp[0,:]*np.cos(lat_r[:]))*1.0/180.
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
    for il in range(0,nlat):
      # then compute toward south
        NHT_ctl_tmp2[il]=2*np.pi*re**2*np.sum(fntoa_ctl_w[:il])*1.0/180.*np.pi*1.0e-15
        NHT_exp_tmp2[il]=2*np.pi*re**2*np.sum(fntoa_exp_w[:il])*1.0/180.*np.pi*1.0e-15
        NHTa_ctl_tmp2[il]=2*np.pi*re**2*np.sum(fntoa_ctl_w[:il]-fnsfc_ctl_w[:il])*1.0/180.*np.pi*1.0e-15
        NHTa_exp_tmp2[il]=2*np.pi*re**2*np.sum(fntoa_exp_w[:il]-fnsfc_exp_w[:il])*1.0/180.*np.pi*1.0e-15
     # compute the average
        NHT_ctl=(NHT_ctl_tmp1+NHT_ctl_tmp2)*0.5
        NHTa_ctl=(NHTa_ctl_tmp1+NHTa_ctl_tmp2)*0.5
        NHT_exp=(NHT_exp_tmp1+NHT_exp_tmp2)*0.5
        NHTa_exp=(NHTa_exp_tmp1+NHTa_exp_tmp2)*0.5

        #NHT_ctl=NHT_ctl_tmp1
        #NHTa_ctl=NHTa_ctl_tmp1
        #NHT_exp=NHT_exp_tmp1
        #NHTa_exp=NHTa_exp_tmp1
        NHTna_ctl=NHT_ctl-NHTa_ctl
        NHTna_exp=NHT_exp-NHTa_exp

    #print(NHT_ctl)
    #exit()
# plot 
    fig=plt.figure(figsize=(7,8))
    ax1=fig.add_subplot(211)
    ax2=fig.add_subplot(212)
    line_ctl1=ax1.plot(lat[:],NHT_ctl[:],ls="-",lw=1,c="k",label="NHT_tot(standard)")
    line_ctl2=ax1.plot(lat[:],NHTa_ctl[:],ls="-",lw=1,c="magenta",label="NHT_atm(standard)")
    line_ctl3=ax1.plot(lat[:],NHTna_ctl[:],ls="-",lw=1,c="aqua",label="NHT_nonatm(standard)")
    line_exp1=ax1.plot(lat[:],NHT_exp[:],ls="--",lw=1,c="k",label="NHT_tot(mc6_scat)")
    line_exp2=ax1.plot(lat[:],NHTa_exp[:],ls="--",lw=1,c="magenta",label="NHT_atm(mc6_scat)")
    line_exp3=ax1.plot(lat[:],NHTna_exp[:],ls="--",lw=1,c="aqua",label="NHT_nonatm(mc6_scat)")

    line_dif1=ax2.plot(lat[:],NHT_exp[:]-NHT_ctl[:],ls="-",lw=1,c="k",label="NHT_tot")
    line_dif2=ax2.plot(lat[:],NHTa_exp[:]-NHTa_ctl[:],ls="-",lw=1,c="magenta",label="NHT_atm")
    line_dif3=ax2.plot(lat[:],NHTna_exp[:]-NHTna_ctl[:],ls="-",lw=1,c="aqua",label="NHT_nonatm")

    line_zero1=ax1.plot(lat[:],np.zeros((nlat)),ls="-",lw=1,c="gray")
    line_zero2=ax2.plot(lat[:],np.zeros((nlat)),ls="-",lw=1,c="gray")

    ax1.set_xlim(-89.5,89.5)
    ax2.set_xlim(-89.5,89.5)
    #ax1.set_ylim(-7.9,7.9)
    # titles
    ax1.set_title("Northward_Heat_Transport "+season,fontsize=12,fontweight='bold')
    ax2.set_title("Difference "+season,fontsize=12,fontweight='bold')
    #ax1.set_xlabel("Latitude",fontsize=12,fontweight='bold')
    ax2.set_xlabel("Latitude",fontsize=12,fontweight='bold')
    ax1.set_ylabel("NHT(PW)",fontsize=12,fontweight='bold')
    ax2.set_ylabel("NHT(PW)",fontsize=12,fontweight='bold')
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
    fig.subplots_adjust(hspace=0.2)
    #save figure as file
    if os.environ["fig_save"]=="True":
        fname="d3_northw_energy_transport_"+season+"."+os.environ["fig_suffix"]
        plt.savefig(os.environ["OUTDIR"]+"/figures/"+fname,dpi=150)
    if os.environ["fig_show"]=="True":
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
