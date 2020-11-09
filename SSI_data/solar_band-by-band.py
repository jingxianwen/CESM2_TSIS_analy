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


#-- data path
fctl=netcdf_dataset("Solar_avg_CESM_control.nc","r")
fexp=netcdf_dataset("Solar_avg_CESM_TSIS.nc","r")

#-- information for ploting 
var_long_name="Band-by-Band TOA Downward SW"
figure_name="Band_by_Band_downward_TOA_SW_ANN"
units=r"W/m$^2$"

#-- read data: wvl, ssi, tsi 
wvl_ctl=fctl.variables["wvl"]
ssi_ctl=fctl.variables["ssi"][0,:]
tsi_ctl=fctl.variables["tsi"]

wvl_exp=fexp.variables["wvl"]
ssi_exp=fexp.variables["ssi"][0,:]
tsi_exp=fexp.variables["tsi"]

nwvl_ctl=ssi_ctl.size #number of wave length in ssi
nwvl_exp=ssi_exp.size #number of wave length in ssi

#-- decide band width 
bndw_ctl=np.zeros(ssi_ctl.size)
bndw_exp=np.zeros(ssi_exp.size)

for ib in range(0,nwvl_ctl-1):
	bndw_ctl[ib]=wvl_ctl[ib+1]-wvl_ctl[ib]
bndw_ctl[nwvl_ctl-1]=bndw_ctl[nwvl_ctl-2]   
for ib in range(0,nwvl_exp-1):
	bndw_exp[ib]=wvl_exp[ib+1]-wvl_exp[ib]
bndw_exp[nwvl_exp-1]=bndw_exp[nwvl_exp-2]   

#-- flux in each band 
flux_ctl=bndw_ctl*ssi_ctl*0.001
flux_exp=bndw_exp*ssi_exp*0.001


#-- rrtmg broad-band (GCM) intervals and wave lengths:
wvn_gcm_int=np.flip(np.array([820,2600,3250,4000,4650,5100,6150,7700,\
		8050,12850,16000,22650,29000,38000,50000]))  # [cm-1]
wvl_gcm_int=1./wvn_gcm_int*1.0e7 #wave number to wave length [nm]
nbbnd=wvl_gcm_int.size-1

#-- from ssi to broad band flux
bbflux_ctl=np.zeros(nbbnd) 
bbflux_exp=np.zeros(nbbnd)

for ibb in range(0,nbbnd):
     head=np.min(np.where(wvl_ctl>=wvl_gcm_int[ibb]))
     tail=np.max(np.where(wvl_ctl<wvl_gcm_int[ibb+1]))
     bbflux_ctl[ibb]=sum(flux_ctl[head:tail])

for ibb in range(0,nbbnd):
     head=np.min(np.where(wvl_exp>=wvl_gcm_int[ibb]))
     tail=np.max(np.where(wvl_exp<wvl_gcm_int[ibb+1]))
     bbflux_exp[ibb]=sum(flux_exp[head:tail])


#-- absolute fluxes to flux fractions 
bbfrac_ctl=bbflux_ctl/sum(bbflux_ctl)
bbfrac_exp=bbflux_exp/sum(bbflux_exp)

#-- calculate broad band flux with flux fractions above and GCM-derived SOLIN
solin_ctl=340.42
solin_exp=340.43
bbflux_ctl_gcm=solin_ctl*bbfrac_ctl
bbflux_exp_gcm=solin_exp*bbfrac_exp
bbflux_diff_gcm=bbflux_exp_gcm-bbflux_ctl_gcm

#============================
#=== calculation end here ===
#============================

#-- make the plot
fig=plt.figure(figsize=(7,8))
ax1=fig.add_axes([0.13,0.60,0.78,0.33])
ax2=fig.add_axes([0.13,0.12,0.78,0.33])
x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
bands=["0.2-0.26","0.26-0.34","0.34-0.44","0.44-0.63","0.63-0.78","0.78-1.24","1.24-1.3","1.3-1.63","1.63-1.94","1.94-2.15","2.15-2.5","2.5-3.08","3.08-3.85","3.85-12.2"]

ax1.bar(bands,bbflux_ctl_gcm,color="tab:blue")
ax1.set_title(var_long_name+" (CTL)",fontsize=12)
ax1.set_ylabel(units,fontsize=12)
ax1.set_xlabel("Band wave length",fontsize=12)
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.xaxis.grid(color='gray', linestyle=':')
ax1.yaxis.grid(color='gray', linestyle=':')
ax1.set_xticklabels(labels=bands,rotation=-45)

ax2.bar(bands,bbflux_diff_gcm,color="tab:blue",hatch="//",edgecolor="black")
#ax2.bar(bands,diffs_unsig,color="tab:blue")

ax2.set_title("Diff in "+var_long_name+" (TSIS-CTL)",fontsize=12)
ax2.set_ylabel(units,fontsize=12)
ax2.set_xlabel("Band wave length",fontsize=12)
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.xaxis.grid(color='gray', linestyle=':')
ax2.yaxis.grid(color='gray', linestyle=':')
plt.xticks(x,bands,rotation=-45)
plt.savefig(figure_name+".png")
plt.show()

exit()
