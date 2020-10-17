#====================================================
#
#====================================================
# os
import os
#import netCDF4
from netCDF4 import Dataset as netcdf_dataset
# cartopy
import os
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
# matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.path as mpath
import matplotlib.colors as colors
# numpy
import numpy as np
# parameters
from get_parameters import *

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

# data path
ctl_name="CESM2" #os.environ["ctl_name"]
exp_name="TSIS-1" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/climo/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/climo/"

years=np.arange(2010,2020)
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

varnm="SNOWHLND"

pole='N'
season="ANN"
if pole is 'N':
   var_long_name="Snow Depth Water Equivalent ("+season+")"
   figure_name="snow_depth_contour_"+season
elif pole is 'S':
   var_long_name="Snow Depth Water Equivalent ("+season+")"
   figure_name="snow_depth_contour_"+season
units="m" #"Fraction"
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
    #nlat=lat[:].size
    #nlon=lon[:].size
 
    means_yby_ctl[iy,:,:]=file_ctl.variables[varnm][0,:,:]
    means_yby_exp[iy,:,:]=file_exp.variables[varnm][0,:,:]

means_ctl[:,:]=np.mean(means_yby_ctl,axis=0)
means_exp[:,:]=np.mean(means_yby_exp,axis=0)
diffs=means_exp-means_ctl

if pole == "N":
    latbound1=np.min(np.where(lat[:]>55))
    latbound2=nlat
elif pole == "S":
    latbound1=0
    latbound2=np.max(np.where(lat[:]<-55))+1

tmp=np.zeros((nlat,nlon))
tmp[:,:]=means_ctl[:,:]
#print(tmp[:,:,:].shape)
stats_ctl=get_area_mean_min_max(tmp[latbound1:latbound2,:],lat[latbound1:latbound2])
tmp[:,:]=means_exp[:,:]
stats_exp=get_area_mean_min_max(tmp[latbound1:latbound2,:],lat[latbound1:latbound2])
tmp[:,:]=diffs[:,:]
stats_dif=get_area_mean_min_max(tmp[latbound1:latbound2,:],lat[latbound1:latbound2])

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
print(plot_sig)

# add cyclic
dtctl=add_cyclic_point(means_ctl[:,:])
dtexp=add_cyclic_point(means_exp[:,:])
dtdif=add_cyclic_point(diffs[:,:])
dtdif_sig=add_cyclic_point(diffs_sig[:,:])
lon=np.append(lon[:],360.)

# make plot

#  data is originally on PlateCarree projection.
#  cartopy need this to transform projection
data_crs=ccrs.PlateCarree()

#parameters=get_parameters(varnm,season)
#projection = ccrs.PlateCarree(central_longitude=180)
if pole == "N":
    projection = ccrs.NorthPolarStereo(central_longitude=0)
elif pole == "S":
    projection = ccrs.SouthPolarStereo(central_longitude=0)

fig = plt.figure(figsize=[8.0,11.0],dpi=150.)
#fig.set_size_inches(4.5, 6.5, forward=True)
plotTitle = {'fontsize': 13.}
#plotSideTitle = {'fontsize': 9., 'verticalalignment':'center'}
plotSideTitle = {'fontsize': 12.}
plotText = {'fontsize': 10.}
panel = [(0.27, 0.65, 0.3235, 0.25),\
         (0.27, 0.35, 0.3235, 0.25),\
         (0.27, 0.05, 0.3235, 0.25),\
        ]
labels=[ctl_name,exp_name,exp_name+" - "+ctl_name] 

#contour_levs=np.linspace(0,0.3,7) #[120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
#contour_levs=[0.05,0.1,0.15,0.2,0.25,0.3,1.0,10.5]
#diff_levs=np.linspace(-1,1,11) #[-50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50]
#diff_levs=[-1.2,-0.05,-0.008,-0.006,-0.004,-0.002,0.002,0.004,0.006,0.008,0.05,1.2]
colormap="GnBu"
colormap_diff="RdBu_r"

for i in range(0,3):
   #1. first plot
    levels = None
    if i != 2:
        cnlevels= [0.05,0.1,0.15,0.2,0.25,0.3,1.0,10.5]  #parameters["contour_levs"]
        norm = PiecewiseNorm(cnlevels)
    else:
        cnlevels= [-1.2,-0.05,-0.008,-0.006,-0.004,-0.002,0.002,0.004,0.006,0.008,0.05,1.2]
        norm = PiecewiseNorm(cnlevels)
          #np.linspace(-0.08,0.08,9) #parameters["diff_levs"]

    #if len(cnlevels) >0:
    #        levels = [-1.0e8] + cnlevels + [1.0e8]
    #        norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

    ax = fig.add_axes(panel[i],projection=projection,autoscale_on=True)
    ax.set_global()
    if pole == "N":
        ax.gridlines(color="gray",linestyle=":",\
    		xlocs=[0,60,120,180,240,300,360],ylocs=[50,60,70,80,89.5])
    elif pole == "S":
        ax.gridlines(color="gray",linestyle=":",\
    		xlocs=[0,60,120,180,240,300,360],ylocs=[-50,-60,-70,-80,-89.5])
    #ax.grid(c='gray',ls=':')
    
    if pole == "N":
        ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
    elif pole == "S":
        ax.set_extent([-180, 180, -55, -90], crs=ccrs.PlateCarree())

    if i == 0:
        dtplot=dtexp[:,:]
        cmap=colormap
        stats_now=stats_exp
    elif i == 1:
        dtplot=dtctl[:,:]
        cmap=colormap
        stats_now=stats_ctl
    else:
        dtplot=dtdif[:,:]
        cmap=colormap_diff
        stats_now=stats_dif

    p1 = ax.contourf(lon[:],lat[latbound1:latbound2],dtplot[latbound1:latbound2,:],\
                transform=data_crs,\
                norm=norm,\
                levels=cnlevels,\
                cmap=cmap,\
                #extend="both",\
                #autoscale_on=True\
        	    )
    #ax.set_aspect("auto")
    ax.coastlines(lw=0.3)
    #ax.set_autoscale_on()

    theta = np.linspace(0, 2 * np.pi, 100)
    #center, radius = [0.5, 0.5], 0.5
    # correct center location to match latitude circle and contours.
    #center, radius = [0.5, 0.495], 0.50
    center, radius = [0.5, 0.5], 0.50
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ##add longitude and latitude mark
    ##lon
    #transf=data_crs._as_mpl_transform(ax)
    #ax.annotate("0", xy=(-1.85,57.5), xycoords=transf,fontsize=8)
    #ax.annotate("60E", xy=(57,59), xycoords=transf,fontsize=8)
    #ax.annotate("120E", xy=(120,60), xycoords=transf,fontsize=8)
    #ax.annotate("180", xy=(185.2,59.5), xycoords=transf,fontsize=8)
    #ax.annotate("120W", xy=(246, 52.5), xycoords=transf,fontsize=8)
    #ax.annotate("60W", xy=(297, 53), xycoords=transf,fontsize=8)
    ##lat
    #ax.annotate("60N", xy=(-4.5,61.3), xycoords=transf,fontsize=8)
    #ax.annotate("70N", xy=(-8.5,71), xycoords=transf,fontsize=8)
    #ax.annotate("80N", xy=(-17.5,81), xycoords=transf,fontsize=8)

    # title
    ax.set_title(labels[i],loc="center",fontdict=plotSideTitle)
    #ax.set_title(units,loc="right",fontdict=plotSideTitle)

    # color bar
    cbax = fig.add_axes((panel[i][0] + 0.35, panel[i][1] + 0.0354, 0.0326, 0.1792))
    cbar = fig.colorbar(p1, cax=cbax, ticks=cnlevels)
    #w, h = get_ax_size(fig, cbax)
    cbar.ax.tick_params(labelsize=9.0, length=0)

    # Mean, Min, Max
    fig.text(panel[i][0] + 0.35, panel[i][1] + 0.225,
             "Max\nMean\nMin", ha='left', fontdict=plotText)
    fig.text(panel[i][0] + 0.45, panel[i][1] + 0.225, "%.2f\n%.2f\n%.2f" %
             (stats_now[2],stats_now[0],stats_now[1]), ha='right', fontdict=plotText)

    if i==2 and plot_sig:
        p1 = ax.contourf(lon[:],lat[latbound1:latbound2],dtdif_sig[latbound1:latbound2,:],\
                    transform=data_crs,\
                    norm=norm,\
                    levels=cnlevels,\
		    hatches=['...'], \
                    cmap=cmap,\
                    extend="both",\
                    #autoscale_on=True\
            	    )
         
fig.suptitle(var_long_name, x=0.5, y=0.96, fontdict=plotTitle)
#save figure as file
#if os.environ["fig_save"]=="True":
#    fname="d2_polar_contour_"+pole+"_"+varnm+"_"+season+"."+os.environ["fig_suffix"]
plt.savefig(figure_name+".eps")
plt.show()
plt.close()

