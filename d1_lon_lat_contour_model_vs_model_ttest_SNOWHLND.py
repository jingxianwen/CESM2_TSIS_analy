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
from get_parameters import *  #get_area_mean_min_max

# scipy
from scipy import stats

from matplotlib.colors import Normalize
class PiecewiseNorm(Normalize):
    def __init__(self, levels, clip=False):
        # the input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))


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

#---
varnm="SNOWHLND"  #np.array(["FLNS","SOLIN","LHFLX","SHFLX"])
season="ANN"
#figure_name="FSNT_vis_lat_lon_ANN"
#units=r"W/m$^2$"

nlat=96
nlon=144
nlev=32

means_yby_ctl=np.zeros((years.size,nlat,nlon)) #year by year mean for each variable
means_yby_exp=np.zeros((years.size,nlat,nlon)) #year by year mean for each variable
means_ctl=np.zeros((nlat,nlon)) #year by year mean for each variable
means_exp=np.zeros((nlat,nlon)) #year by year mean for each variable
diffs=np.zeros((nlat,nlon)) #multi-year exp-ctl diff for each variable
pvals=np.zeros((nlat,nlon)) #pvalues of ttest

for iy in range(0,years.size):
   # open data file
   fctl=fpath_ctl+ctl_pref+"_"+season+"_"+str(years[iy])+".nc"
   fexp=fpath_exp+exp_pref+"_"+season+"_"+str(years[iy])+".nc"
   file_ctl=netcdf_dataset(fctl,"r")
   file_exp=netcdf_dataset(fexp,"r")
   
   # read lat and lon
   lat=file_ctl.variables["lat"]
   lon=file_ctl.variables["lon"]
   # read data and calculate mean/min/max
   means_yby_ctl[iy,:,:]=file_ctl.variables[varnm][0,:,:]
   means_yby_exp[iy,:,:]=file_exp.variables[varnm][0,:,:]

means_ctl[:,:]=np.mean(means_yby_ctl,axis=0)
means_exp[:,:]=np.mean(means_yby_exp,axis=0)
diffs=means_exp-means_ctl
tmp=np.zeros((nlat,nlon))
#stats_all=np.zeros((3,3))
#stats_now=np.zeros((3))
tmp[:,:]=means_ctl[:,:]
stats_ctl=get_area_mean_min_max(tmp,lat[:])
tmp[:,:]=means_exp[:,:]
stats_exp=get_area_mean_min_max(tmp,lat[:])
tmp[:,:]=diffs[:,:]
stats_dif=get_area_mean_min_max(tmp,lat[:])

#t-test
ttest=stats.ttest_ind(means_yby_ctl,means_yby_exp,axis=0)
pvalues=ttest.pvalue
siglev=0.05
diffs_sig=np.zeros(diffs.shape)
diffs_sig[:,:]=np.nan
plot_sig=False
for ilat in range(0,nlat):
   for ilon in range(0,nlon):
      if pvalues[ilat,ilon] < siglev:
          diffs_sig[ilat,ilon]=diffs[ilat,ilon]
          plot_sig=True

# add cyclic
dtctl=add_cyclic_point(means_ctl[:,:])
dtexp=add_cyclic_point(means_exp[:,:])
dtdif=add_cyclic_point(diffs[:,:])
dtdif_sig=add_cyclic_point(diffs_sig[:,:])
lon=np.append(lon[:],360.)
#print(dtctl.shape)
#print(lon.shape)
#exit()
#print(lon)


# make plot
#parameters=get_parameters(varnm,season)

projection = ccrs.PlateCarree(central_longitude=0)

#fig = plt.figure(figsize=[7.0,11.0],dpi=150.)

fig=plt.figure(figsize=(8,5))
plotTitle = {'fontsize': 14.}
plotSideTitle = {'fontsize': 14.}
plotText = {'fontsize': 8.}
#panel = [(0.1691, 0.6810, 0.6465, 0.2258), \
#         (0.1691, 0.3961, 0.6465, 0.2258), \
#         (0.1691, 0.1112, 0.6465, 0.2258), \
#         ]
panel = [(0.10,0.15,0.70,0.65)]
labels=[ctl_name,exp_name,exp_name+"-"+ctl_name] 

#units=r"Wm$^-$$^2$"
units="m"
contour_levs=np.linspace(0,0.3,7) #[120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
contour_levs=[0.05,0.1,0.15,0.2,0.25,0.3,1.0,10.5]
diff_levs=np.linspace(-1,1,11) #[-50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50]
diff_levs=[-1.2,-0.05,-0.008,-0.006,-0.004,-0.002,0.002,0.004,0.006,0.008,0.05,1.2]
colormap="GnBu"
colormap_diff="RdBu_r"

for i in range(0,1):
   #1. first plot
    levels = None
    norm = None
    cnlevels=diff_levs
    #cnlevels=contour_levs

    ax = fig.add_axes(panel[0],projection=ccrs.PlateCarree(central_longitude=180))

    dtplot=dtdif #[:,:]
    print(stats_dif)
    cmap=colormap_diff
    stats_now=stats_dif

    #dtplot=dtctl
    #print(stats_ctl)
    #cmap=colormap
    #stats_now=stats

    p1 = ax.contourf(lon[:],lat[:],dtplot[:,:],\
                transform=projection,\
                norm=PiecewiseNorm(cnlevels),\
                levels=cnlevels,\
                cmap=cmap,\
                #extend="both",\
        	    )
    ax.set_aspect("auto")
    ax.coastlines(lw=0.3)

    # title
    ax.set_title("Diff in Snow Depth Water Equivalent (TSIS - CESM2, "+season+")",loc="center",fontdict=plotSideTitle)
    #ax.set_title("Snow Depth Water Equivalent (CESM2, "+season+")",loc="center",fontdict=plotSideTitle)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
    ax.set_yticks([ -60, -30, 0, 30, 60 ], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=14.0, direction='out', width=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # color bar
    cbax = fig.add_axes((panel[0][0] + 0.72, panel[0][1]+ 0.025, 0.02, 0.5))
    cbar = fig.colorbar(p1, cax=cbax, ticks=cnlevels)
    cbar.ax.tick_params(labelsize=14.0, length=0)
    fig.text(panel[0][0] + 0.72, panel[0][1] + 0.55, \
             "m",ha='left',fontsize=14)
    # Mean, Min, Max
    #fig.text(panel[0][0] + 0.6635, panel[0][1] + 0.2107,
    #         "Max\nMean\nMin", ha='left', fontdict=plotText)
    #fig.text(panel[0][0] + 0.7835, panel[0][1] + 0.2107, "%.2f\n%.2f\n%.2f" %
    #        (stats_now[2],stats_now[0],stats_now[1]), ha='right', fontdict=plotText)
            #stats_now[[2,0,1]], ha='right', fontdict=plotText)

    p2 = ax.contourf(lon[:],lat[:],dtdif_sig[:,:],\
                transform=projection,\
                norm=PiecewiseNorm(cnlevels),\
                levels=cnlevels,\
	            hatches=['...'], \
                cmap=cmap,\
                #extend="both",\
     	       )
        
#fig.suptitle(varnm, x=0.5, y=0.96, fontsize=14)
#save figure as file
#if os.environ["fig_save"]=="True":
#    fname="d1_lon_lat_contour_"+varnm+"_"+season+"."+os.environ["fig_suffix"]
#    plt.savefig(os.environ["OUTDIR"]+"/figures/"+fname)
#plt.savefig(figure_name+".png")
#if os.environ["fig_show"]=="True":
#    plt.show()
#plt.savefig("./figures/diff_cldtot_"+season+".eps")
plt.show()
plt.close()