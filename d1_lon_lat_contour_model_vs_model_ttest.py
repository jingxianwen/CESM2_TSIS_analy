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

# variable group 
#---
#varnm="PRECT"  #np.array(["FLNS","FSNS","LHFLX","SHFLX"])
#season="ANN"
#figure_name="Precipitation_rate_lat_lon_ANN"
#---
#varnm="CLDTOT"  #np.array(["FLNS","FSNS","LHFLX","SHFLX"])
#season="ANN"
#figure_name="Cloud_Fraction_lat_lon_ANN"

#---
#varnm="FSUTOA"  #np.array(["FLNS","FSNS","LHFLX","SHFLX"])
#season="ANN"
#figure_name="FSUTOA_lat_lon_ANN"

#---
varnm="FSNS"  #np.array(["FLNS","FSNS","LHFLX","SHFLX"])
season="ANN"
figure_name="FSNS_lat_lon_ANN"
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
   fctl=fpath_ctl+ctl_pref+"_ANN_"+str(years[iy])+".nc"
   fexp=fpath_exp+exp_pref+"_ANN_"+str(years[iy])+".nc"
   file_ctl=netcdf_dataset(fctl,"r")
   file_exp=netcdf_dataset(fexp,"r")
   
   # read lat and lon
   lat=file_ctl.variables["lat"]
   lon=file_ctl.variables["lon"]
   scale=np.float32(24.*3600.*1000.)
   # read data and calculate mean/min/max
   if varnm=="PRECT":
      means_yby_ctl[iy,:,:]=(file_ctl.variables["PRECC"][0,:,:]+file_ctl.variables["PRECL"][0,:,:])*scale
      means_yby_exp[iy,:,:]=(file_ctl.variables["PRECC"][0,:,:]+file_exp.variables["PRECL"][0,:,:])*scale
   else:
      means_yby_ctl[iy,:,:]=file_ctl.variables[varnm][0,:,:]
      means_yby_exp[iy,:,:]=file_exp.variables[varnm][0,:,:]

means_ctl[:,:]=np.mean(means_yby_ctl,axis=0)
means_exp[:,:]=np.mean(means_yby_exp,axis=0)
diffs=means_exp-means_ctl
tmp=np.zeros((1,nlat,nlon))
#stats_all=np.zeros((3,3))
#stats_now=np.zeros((3))
tmp[0,:,:]=means_ctl[:,:]
stats_ctl=get_area_mean_min_max(tmp,lat[:])
tmp[0,:,:]=means_exp[:,:]
stats_exp=get_area_mean_min_max(tmp,lat[:])
tmp[0,:,:]=diffs[:,:]
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

print(plot_sig)
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
parameters=get_parameters(varnm,season)
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
labels=[ctl_name,exp_name,exp_name+"-"+ctl_name] 
units=parameters["units"]
for i in range(0,3):
   #1. first plot
    levels = None
    norm = None
    if i != 2:
        cnlevels=parameters["contour_levs"]
        #cnlevels=np.arange(125,300,20)
        #cnlevels=np.arange(0,80,8)
    else:
        #cnlevels=np.arange(-9,10,1.5)
        #cnlevels=np.arange(-6,7,1)
        cnlevels=parameters["diff_levs"]

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
        dtplot=dtctl #[:,:]
        cmap=parameters["colormap"]
        stats_now=stats_ctl
        print(stats_now)
    elif i == 1:
        dtplot=dtexp #[:,:]
        cmap=parameters["colormap"]
        stats_now=stats_exp
    else:
        dtplot=dtdif #[:,:]
        cmap=parameters["colormap_diff"]
        stats_now=stats_dif
    #print(lon.shape)
    #print(lat.shape)
    #print(dtplot.shape)
    p1 = ax.contourf(lon[:],lat[:],dtplot[:,:],\
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
  #  print(stats_now.shape)
    fig.text(panel[i][0] + 0.6635, panel[i][1] + 0.2107,
             "Mean\nMin\nMax", ha='left', fontdict=plotText)
    print(stats_now)
    fig.text(panel[i][0] + 0.7835, panel[i][1] + 0.2107, "%.2f\n%.2f\n%.2f" %
            stats_now[0:3], ha='right', fontdict=plotText)


    if i==2 and plot_sig: 
       p2 = ax.contourf(lon[:],lat[:],dtdif_sig[:,:],\
                   transform=projection,\
                   #norm=norm,\
                   levels=cnlevels,\
		   hatches=['......'], \
                   cmap=cmap,\
                   extend="both",\
        	       )
        
#fig.suptitle(varnm, x=0.5, y=0.96, fontsize=14)
fig.suptitle(varnm+" "+season, x=0.5, y=0.96, fontdict=plotTitle)
#save figure as file
#if os.environ["fig_save"]=="True":
#    fname="d1_lon_lat_contour_"+varnm+"_"+season+"."+os.environ["fig_suffix"]
#    plt.savefig(os.environ["OUTDIR"]+"/figures/"+fname)
plt.savefig(figure_name+".png")
#if os.environ["fig_show"]=="True":
#    plt.show()
plt.show()
plt.close()

# write mean values to table    
#    col1=varnm+"["+units+"]"
#    line=col1+" "*(18-len(col1))+f'{stats_out[0]:10.3f}'+" "*5+\
#         f'{stats_out[1]:10.3f}'+" "*5+\
#         f'{stats_out[2]:10.3f}'+" "*5+\
#         f'{stats_out[3]:10.3f}'+"%"
#    table.write(line+"\r\n")
#    return 0
    
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
