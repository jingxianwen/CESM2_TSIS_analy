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
from get_parameters import get_area_mean_min_max

#def lon_lat_contour_model_vs_model(varnm,season,scale_ctl,scale_exp,table):
# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
fpath_ctl='/raid00/xianwen/data/cesm211_solar_exp/solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019/climo/'
fpath_exp='/raid00/xianwen/data/cesm211_solar_exp/solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019/climo/'

f1=fpath_ctl+"solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019_climo_ANN.nc"
f2=fpath_exp+"solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019_climo_ANN.nc"

# open data file
file_ctl=netcdf_dataset(f1,"r")
file_exp=netcdf_dataset(f2,"r")

# read lat and lon
lat=file_ctl.variables["lat"]
lon=file_ctl.variables["lon"]
lev=file_ctl.variables["lev"]

#varnm="FSSDCLRS14"
varnm="PS"   #
#varnm_off="FLUTC_OFF"  #offline computation
#units=r"W/m$^2$"
units="hPa"
figure_name="lat_lon_cldlow"

dtctl=file_ctl.variables[varnm][:,:,:] * 0.01 
dtexp=file_exp.variables[varnm][:,:,:] * 0.01

dtdif=dtexp[:,:,:]-dtctl[:,:,:]

#exit()

# add cyclic
#dtctl=add_cyclic_point(dtctl[:,:,:])
#dtexp=add_cyclic_point(dtexp[:,:,:])
#dtdif=add_cyclic_point(dtdif[:,:,:])
#lon=np.append(lon[:],360.)
#print(lon)
# make plot
#parameters=get_parameters(varnm,season)
projection = ccrs.PlateCarree(central_longitude=0)

#fig = plt.figure(figsize=[7.0,11.0],dpi=150.)

fig=plt.figure(figsize=(7,8))
plotTitle = {'fontsize': 13.}
plotSideTitle = {'fontsize': 9.}
plotText = {'fontsize': 8.}
panel = [(0.1691, 0.6810, 0.6465, 0.2258), \
         (0.1691, 0.3961, 0.6465, 0.2258), \
         (0.1691, 0.1112, 0.6465, 0.2258), \
         ]
#labels=[exp_name,ctl_name,varnm+" 500mb ("+exp_name+"-"+ctl_name+")"] 
#labels=[exp_name,ctl_name,exp_name+"-"+ctl_name] 
labels=["PS (CESM2)","PS (TSIS-1)","\u0394"+"PS (TSIS-1 - CESM2)"] 
#labels=[varnm2+"(noScat)",varnm2+"(Scat)","\u0394"+varnm2+"(Scat-noScat)"] 
#units=parameters["units"]
#units="W/m2"
#units="kg/m2"
for i in range(0,3):
    print(i)
   #1. first plot
    levels = None
    norm = None
    if i != 2:
        #cnlevels=np.array([0,10,20,30,40,50,60]) #parameters["contour_levs"]
        #cnlevels=np.arange(145,320,20)
        #cnlevels=np.arange(90,450,30)
        #cnlevels=np.arange(20,150,10)
        #cnlevels=np.arange(10,100,10)
        cnlevels=np.arange(600,1050,50)
    else:
        #cnlevels=np.arange(-0.5,0.6,0.1)
        cnlevels=np.arange(-0.8,1.0,0.2)
        #cnlevels=np.arange(-0.8,1.0,0.2)
        #cnlevels=np.arange(-1.2,1.3,0.3)
        #cnlevels=np.array([-4,-3.5,-3,-2.5,-2,-1.5,-1.,-0.5,0.5,1.,1.5,2.,2.5,3.,3.5,4.]) #parameters["diff_levs"]

    #if len(cnlevels) >0:
    #        levels = [-1.0e8] + cnlevels + [1.0e8]
    #        norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

    ax = fig.add_axes(panel[i],projection=ccrs.PlateCarree(central_longitude=180))
    #ax = fig.add_axes(panel[i],projection=projection)
    #ax.set_global()
    #cmap="PiYG_r" #parameters["colormap"]
    #ax.set_extent([0, 180, -90, 90], crs=ccrs.PlateCarree())
    #p1 = ax.contourf(lon[:],lat[:],dtexp[0,:,:])
    if i == 0:
        dtplot=dtexp[:,:,:]
        #cmap="PiYG_r" #parameters["colormap"]
        cmap="jet" #parameters["colormap"]
        #stats=stats_ctl[:]
    elif i == 1:
        dtplot=dtctl[:,:,:]
        #cmap="PiYG_r" #parameters["colormap"]
        cmap="jet" #parameters["colormap"]
        #stats=stats_exp[:]
    else:
        dtplot=dtdif[:,:,:]
        cmap="seismic" #parameters["colormap_diff"]
        #cmap="YlOrRd" #parameters["colormap_diff"]
        #stats=stats_dif[:]
    p1 = ax.contourf(lon[:],lat[:],dtplot[0,:,:],\
                transform=projection,\
                #norm=norm,\
                levels=cnlevels,\
                cmap=cmap,\
                extend="both",\
        	    )
    ax.set_aspect("auto")
    ax.coastlines(lw=0.3)
    # title
    ax.set_title(labels[i],loc="left",fontdict=plotSideTitle)
    #ax.set_title("exp",fontdict=plotTitle)
    ax.set_title(units,loc="right",fontdict=plotSideTitle)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=8.0, direction='out', width=1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # color bar
    cbax = fig.add_axes((panel[i][0] + 0.6635, panel[i][1] + 0.0215, 0.0326, 0.1792))
    #cbax = fig.add_axes((panel[i][0] + 0.6635, panel[i][1] + 0.0215, 0.0326, 0.2850))
    cbar = fig.colorbar(p1, cax=cbax, ticks=cnlevels)
    #w, h = get_ax_size(fig, cbax)
    cbar.ax.tick_params(labelsize=9.0, length=0)

    # Mean, Min, Max
    #fig.text(panel[i][0] + 0.6635, panel[i][1] + 0.2107,
    #         "Mean\nMin\nMax", ha='left', fontdict=plotText)
    #fig.text(panel[i][0] + 0.7835, panel[i][1] + 0.2107, "%.2f\n%.2f\n%.2f" %
    #         stats[0:3], ha='right', fontdict=plotText)

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
plt.show()
plt.close()
