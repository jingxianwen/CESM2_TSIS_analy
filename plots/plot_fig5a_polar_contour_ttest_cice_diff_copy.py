#====================================================
# import modules
#====================================================
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
import matplotlib.path as mpath
import matplotlib.colors as colors

# numpy
import numpy as np

# parameters
from get_parameters import *

# scipy
from scipy import stats

# data path
ctl_name="CESM2" #os.environ["ctl_name"]
exp_name="TSIS-1" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/Volumes/WD4T_1/cesm2_solar_exp/"+ctl_pref+"/climo/"
fpath_exp="/Volumes/WD4T_1/cesm2_solar_exp/"+exp_pref+"/climo/"

years=np.arange(2010,2020)
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

varnm="ICEFRAC"

pole='N'
season="ANN"
if pole is 'N':
   var_long_name="Arctic Sea Ice Fraction ("+season+")"
   figure_name="fig5a_Arctic_Sea_Ice_contour_diff"+season
elif pole is 'S':
   var_long_name="Antarctic Sea Ice Fraction ("+season+")"
   figure_name="fig5b_Antarctic_Sea_Ice_contour_diff"+season
units=" " #"Fraction"

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

# add cyclic
dtctl=add_cyclic_point(means_ctl[:,:])
dtexp=add_cyclic_point(means_exp[:,:])
dtdif=add_cyclic_point(diffs[:,:])
dtdif_sig=add_cyclic_point(diffs_sig[:,:])
lon=np.append(lon[:],360.)

#-----------------
# make the plot
#-----------------

#  data is originally on PlateCarree projection.
#  cartopy need this to transform projection
data_crs=ccrs.PlateCarree()
parameters=get_parameters(varnm,season)
if pole == "N":
    projection = ccrs.NorthPolarStereo(central_longitude=0)
elif pole == "S":
    projection = ccrs.SouthPolarStereo(central_longitude=0)

fig = plt.figure(figsize=[5.0,6.0],dpi=150.)

# set text properties
plotTitle = {'fontsize': 13.}
plotSideTitle = {'fontsize': 13.}
plotText = {'fontsize': 11.}
panel = [(0.1,0.1,0.6,0.6)]

label = "Diff in Sea Ice Fraction"

units=parameters["units"]

levels = None
norm = None
cnlevels=np.array([-0.08,-0.06,-0.04,-0.02,-0.01,0.01,0.02,0.04,0.06,0.08])

ax = fig.add_axes(panel[0],projection=projection,autoscale_on=True)
ax.set_global()
if pole == "N":
    ax.gridlines(color="gray",linestyle=":",\
		xlocs=[0,60,120,180,240,300,360],ylocs=[50,60,70,80,89.5])
elif pole == "S":
    ax.gridlines(color="gray",linestyle=":",\
		xlocs=[0,60,120,180,240,300,360],ylocs=[-50,-60,-70,-80,-89.5])

if pole == "N":
    ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
elif pole == "S":
    ax.set_extent([-180, 180, -55, -90], crs=ccrs.PlateCarree())

dtplot=dtdif[:,:]
cmap=parameters["colormap_diff"]
stats_now=stats_dif

p1 = ax.contourf(lon[:],lat[latbound1:latbound2],dtplot[latbound1:latbound2,:],\
            transform=data_crs,\
            #norm=norm,\
            levels=cnlevels,\
            cmap=cmap,\
            extend="both",\
            #autoscale_on=True\
    	    )
ax.coastlines(lw=0.3)

theta = np.linspace(0, 2 * np.pi, 100)
# correct center location to match latitude circle and contours.
center, radius = [0.5, 0.5], 0.50
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)
ax.set_title(label,loc="center",fontdict=plotSideTitle)
# color bar
cbax = fig.add_axes((panel[0][0] + 0.62, panel[0][1] + 0.1, 0.04, 0.35))
cbar = fig.colorbar(p1, cax=cbax, ticks=cnlevels)
cbar.ax.tick_params(labelsize=12.0, length=0)

# Mean, Min, Max
fig.text(panel[0][0] + 0.62, panel[0][1] + 0.48,
         "Max\nMean\nMin", ha='left', fontdict=plotText)
fig.text(panel[0][0] + 0.78, panel[0][1] + 0.48, "%.2f\n%.2f\n%.2f" %
         (stats_now[2],stats_now[0],stats_now[1]), ha='right', fontdict=plotText)

# add ttest hatches
p1 = ax.contourf(lon[:],lat[latbound1:latbound2],dtdif_sig[latbound1:latbound2,:],\
            transform=data_crs,\
            #norm=norm,\
            levels=cnlevels,\
            hatches=['...'], \
            cmap=cmap,\
            extend="both",\
            #autoscale_on=True\
    	    )
     
#plt.savefig("./figures/"+figure_name+".pdf")
#plt.savefig("./figures/"+figure_name+".png",dpi=150)
plt.show()
plt.close()

