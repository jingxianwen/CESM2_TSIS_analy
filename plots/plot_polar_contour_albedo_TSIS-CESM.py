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
import matplotlib.path as mpath
# numpy
import numpy as np
# parameters
from get_parameters import get_area_mean_min_max

#def lon_lat_contour_model_vs_model(varnm,season,scale_ctl,scale_exp,table):
# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
fpath_ctl='/Volumes/WD4T_1/cesm2_solar_exp/solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019/climo/'
#fpath_exp='/Volumes/WD4T_1/cesm2_solar_exp/solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019/climo/'

f1=fpath_ctl+"solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019_climo_ANN.nc"
#f2=fpath_exp+"solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019_climo_ANN.nc"

# open data file
file_ctl=netcdf_dataset(f1,"r")
#file_exp=netcdf_dataset(f2,"r")

# read lat and lon
lat=file_ctl.variables["lat"]
lon=file_ctl.variables["lon"]
lev=file_ctl.variables["lev"]

nlat=len(lat)
nlon=len(lon)
nlev=len(lev)

#varnm="FSSDCLRS14"
varnm="FSSUS10"   # FSSUS10, FSSUS08 
varnm2="FSSDS10"  # FSSDS10, FSSDS08
varnm3="FSSUS08"   # FSSUS10, FSSUS08 
varnm4="FSSDS08"  # FSSDS10, FSSDS08
units=""

dtctl=file_ctl.variables[varnm][:,:,:] / file_ctl.variables[varnm2][:,:,:]
dtexp=file_ctl.variables[varnm3][:,:,:] / file_ctl.variables[varnm4][:,:,:]

dtdif=dtctl[:,:,:]-dtexp[:,:,:]

#exit()

# add cyclic
dtctl=add_cyclic_point(dtctl[:,:,:])
dtexp=add_cyclic_point(dtexp[:,:,:])
dtdif=add_cyclic_point(dtdif[:,:,:])
lon=np.append(lon[:],360.)
#print(lon)

#--------------
# make plot
#--------------

#-- pole related setups --
pole = 'S' 
#ip = ''
ip = 'diff'
if pole is 'N':
    figure_name='fig_Arctic_VIS_albedo_CESM2'
    latbound1=np.min(np.where(lat[:]>55))
    latbound2=nlat
    projection = ccrs.NorthPolarStereo(central_longitude=0)
elif pole is 'S':
    figure_name='fig_Antarctic_VIS_albedo_CESM2'
    latbound1=0
    latbound2=np.max(np.where(lat[:]<-55))+1
    projection = ccrs.SouthPolarStereo(central_longitude=0)
#print(latbound2)
#exit()

#-- calculate area mean --
tmp=np.zeros((nlat,nlon))
tmp[:,:]=dtctl[0,:,:-1]
stats_ctl=get_area_mean_min_max(tmp[latbound1:latbound2,:],lat[latbound1:latbound2])
tmp[:,:]=dtexp[0,:,:-1]
stats_exp=get_area_mean_min_max(tmp[latbound1:latbound2,:],lat[latbound1:latbound2])
tmp[:,:]=dtdif[0,:,:-1]
stats_dif=get_area_mean_min_max(tmp[latbound1:latbound2,:],lat[latbound1:latbound2])
print(stats_dif)

#parameters=get_parameters(varnm,season)
#projection = ccrs.PlateCarree(central_longitude=0)

#fig = plt.figure(figsize=[7.0,11.0],dpi=150.)

fig=plt.figure(figsize=(5,5))
plotTitle = {'fontsize': 13.}
plotSideTitle = {'fontsize': 9.}
plotText = {'fontsize': 12.}
panel = [(0.08, 0.08, 0.7, 0.7), \
         ]
#panel = [(0.1691, 0.6810, 0.6465, 0.2258), \
#         (0.1691, 0.3961, 0.6465, 0.2258), \
#         (0.1691, 0.1112, 0.6465, 0.2258), \
#         ]
labels=["VIS Albedo","\u0394"+"albedo (VIS - NIR)"] 
#units=""
for i in range(0,1):
    print(i)
   #1. first plot
    levels = None
    norm = None
    if ip is 'diff':
        cnlevels=np.arange(-0.30,0.36,0.06)
        dtplot=dtdif[:,:,:]
        #cmap="PiYG_r" #parameters["colormap"]
        cmap="seismic" #parameters["colormap"]
        stats=stats_dif[:]
    else:
        cnlevels=np.arange(0.1,0.9,0.1)
        dtplot=dtctl[:,:,:]
        #cmap="PiYG_r" #parameters["colormap"]
        cmap="rainbow" #parameters["colormap"]
        stats=stats_ctl[:]

    #ax = fig.add_axes(panel[i],projection=ccrs.PlateCarree(central_longitude=180))
    ax = fig.add_axes(panel[0],projection=projection)
    
    #ax.set_global()
    #cmap="PiYG_r" #parameters["colormap"]
    #ax.set_extent([0, 180, -90, 90], crs=ccrs.PlateCarree())
    #p1 = ax.contourf(lon[:],lat[:],dtexp[0,:,:])
    #if i == 0:
    #    dtplot=dtctl[:,:,:]
    #    #cmap="PiYG_r" #parameters["colormap"]
    #    cmap="rainbow" #parameters["colormap"]
    #    stats=stats_ctl[:]
    #elif i == 1:
    #    dtplot=dtdif[:,:,:]
    #    #cmap="PiYG_r" #parameters["colormap"]
    #    cmap="bwr" #parameters["colormap"]
    #    stats=stats_exp[:]
    #else:
    #    dtplot=dtdif[:,:,:]
    #    cmap="seismic" #parameters["colormap_diff"]
    #    #cmap="YlOrRd" #parameters["colormap_diff"]
    #    #stats=stats_dif[:]
    #--- use circle frame ---
    theta = np.linspace(0, 2 * np.pi, 100)
    # correct center location to match latitude circle and contours.
    center, radius = [0.5, 0.5], 0.50
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    #--------------------------
    p1 = ax.contourf(lon[:],lat[latbound1:latbound2],dtplot[0,latbound1:latbound2,:],\
                transform=ccrs.PlateCarree(central_longitude=0),\
                #norm=norm,\
                levels=cnlevels,\
                cmap=cmap,\
                extend="both",\
                alpha=0.9,\
        	    )
    ax.set_aspect("auto")
    ax.coastlines(lw=0.3)
    ax.gridlines(color="gray",linestyle=":",ylocs=[60,70,80,89.5]) 
           #,xlocs=[0,60,120,180,240,300,360],ylocs=[50,60,70,80,89.5])
    # title
    if ip is 'diff':
        ax.set_title(labels[1],loc="center",fontsize=16)
    else:
        ax.set_title(labels[0],loc="center",fontsize=16)
    #ax.set_title("exp",fontdict=plotTitle)
    #ax.set_title(units,loc="right",fontdict=plotSideTitle)
#    ax.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
#    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=8.0, direction='out', width=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # color bar
    cbax = fig.add_axes((panel[i][0] + 0.73, panel[i][1] + 0.04, 0.038, 0.6))
    #cbax = fig.add_axes((panel[i][0] + 0.6635, panel[i][1] + 0.0215, 0.0326, 0.2850))
    cbar = fig.colorbar(p1, cax=cbax, ticks=cnlevels)
    #w, h = get_ax_size(fig, cbax)
    cbar.ax.tick_params(labelsize=14.0, length=0)
    print(stats)
    # Mean, Min, Max
    #fig.text(panel[i][0] + 0.68, panel[i][1] + 0.58,
    #         "Max\nMean\nMin", ha='left', fontdict=plotText)
    #fig.text(panel[i][0] + 0.87, panel[i][1] + 0.58, "%.2f\n%.2f\n%.2f" %
    #         (stats[2],stats[0],stats[1]), ha='right', fontdict=plotText)

#fig.suptitle(varnm, x=0.5, y=0.96, fontsize=14)
#fig.suptitle("July", x=0.5, y=0.96, fontdict=plotTitle)
#save figure as file
#if os.environ["fig_save"]=="True":
#    fname="d1_lon_lat_contour_"+varnm+"_"+season+"."+os.environ["fig_suffix"]
#    plt.savefig(os.environ["OUTDIR"]+"/figures/"+fname)
#plt.savefig("./figures/noEmis_offline_ICEFLAG1_full2/"+figure_name)
#plt.savefig("./"+figure_name)
#if os.environ["fig_show"]=="True":
#    plt.show()
plt.savefig("./figures/"+figure_name+"_"+ip+".pdf")
plt.savefig("./figures/"+figure_name+"_"+ip+".png",dpi=150)
plt.show()
plt.close()
