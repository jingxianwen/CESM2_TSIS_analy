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
#import numpy.ma as ma 

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
#years=np.arange(2014,2015)
months_all=["01","02","03","04","05","06","07","08","09","10","11","12"]

#---
varnm_1="ICEFRAC"  #np.array(["FLNS","SOLIN","LHFLX","SHFLX"])
varnm_2="TS"     #np.array(["FLNS","SOLIN","LHFLX","SHFLX"])
season="ANN"
#figure_name="FSNT_vis_lat_lon_ANN"
#units=r"W/m$^2$"

nlat=96
nlon=144
nlev=32

means_yby_ctl_cld=np.zeros((years.size,nlat,nlon)) #year by year mean for each variable
means_yby_exp_cld=np.zeros((years.size,nlat,nlon)) #year by year mean for each variable
means_ctl_cld=np.zeros((nlat,nlon)) #year by year mean for each variable
means_exp_cld=np.zeros((nlat,nlon)) #year by year mean for each variable
means_yby_ctl_rad=np.zeros((years.size,nlat,nlon)) #year by year mean for each variable
means_yby_exp_rad=np.zeros((years.size,nlat,nlon)) #year by year mean for each variable
means_ctl_rad=np.zeros((nlat,nlon)) #year by year mean for each variable
means_exp_rad=np.zeros((nlat,nlon)) #year by year mean for each variable
diffs_cld=np.zeros((years.size,nlat,nlon)) #multi-year exp-ctl diff for each variable
diffs_rad=np.zeros((years.size,nlat,nlon)) #multi-year exp-ctl diff for each variable

for iy in range(0,years.size):
   # open data file
   fctl=fpath_ctl+ctl_pref+"_"+season+"_"+str(years[iy])+".nc"
   fexp=fpath_exp+exp_pref+"_"+season+"_"+str(years[iy])+".nc"
   #fctl=fpath_ctl+ctl_pref+"_climo_"+season+".nc"
   #fexp=fpath_exp+exp_pref+"_climo_"+season+".nc"

   file_ctl=netcdf_dataset(fctl,"r")
   file_exp=netcdf_dataset(fexp,"r")
   
   # read lat and lon
   lat=file_ctl.variables["lat"]
   lon=file_ctl.variables["lon"]
   # read data and calculate mean/min/max
   means_yby_ctl_cld[iy,:,:]=file_ctl.variables[varnm_1][0,:,:]*1. #fraction to %
   means_yby_exp_cld[iy,:,:]=file_exp.variables[varnm_1][0,:,:]*1. #fraction to %
   means_yby_ctl_rad[iy,:,:]=file_ctl.variables[varnm_2][0,:,:] #-\
                             #file_ctl.variables[varnm_2[1]][0,:,:]
   means_yby_exp_rad[iy,:,:]=file_exp.variables[varnm_2][0,:,:] #-\
                             #file_exp.variables[varnm_2[1]][0,:,:]
   lndfrac=file_exp.variables["LANDFRAC"][0,:,:]

#means_ctl_cld[:,:]=np.mean(means_yby_ctl_cld,axis=0)
#means_exp_cld[:,:]=np.mean(means_yby_exp_cld,axis=0)
#means_ctl_rad[:,:]=np.mean(means_yby_ctl_rad,axis=0)
#means_exp_rad[:,:]=np.mean(means_yby_exp_rad,axis=0)

   diffs_cld[iy,:,:]=np.where(lndfrac[:,:]<0.01,means_yby_exp_cld[iy,:,:]-means_yby_ctl_cld[iy,:,:],np.nan)
   diffs_rad[iy,:,:]=np.where(lndfrac[:,:]<0.01,means_yby_exp_rad[iy,:,:]-means_yby_ctl_rad[iy,:,:],np.nan)

#diffs_cld=np.where(lndfrac[:,:]<0.01,diffs_cld,np.nan)
#diffs_rad=np.where(lndfrac[:,:]<0.01,diffs_rad,np.nan)

coeff=np.zeros((2))

#i_60S=np.max(np.where(lat[:]<-60))+1
#i_30S=np.max(np.where(lat[:]<-30))+1
#i_0=np.max(np.where(lat[:]<0))+1
#i_30N=np.max(np.where(lat[:]<30))+1
#i_60N=np.max(np.where(lat[:]<60))+1
i_60S=np.max(np.where(lat[:]<-60))+1
i_60N=np.max(np.where(lat[:]<60))+1
i_70S=np.max(np.where(lat[:]<-70))+1
i_70N=np.max(np.where(lat[:]<70))+1
nlat=len(lat[:])

#il_1=[i_0   , i_30N , i_60N, i_30S, i_60S,     0 ]
#il_2=[i_30N , i_60N , nlat , i_0  , i_30S, i_60S ]
il_1=[i_60N,0]
il_2=[nlat,i_60S]
#il_1=[i_60N,i_70S]
#il_2=[i_70N,i_60S]


#titles=["0-30"+u"\xb0"+"N","30"+u"\xb0"+"N-60"+u"\xb0"+"N",r"60"+u"\xb0"+"N-90"+u"\xb0"+"N",\
#        "0-30"+u"\xb0"+"S","30"+u"\xb0"+"S-60"+u"\xb0"+"S","60"+u"\xb0"+"S-90"+u"\xb0"+"S"]

titles=[r"60"+u"\xb0"+"N-90"+u"\xb0"+"N",r"60"+u"\xb0"+"S-90"+u"\xb0"+"S"]
# make plot

fig = plt.figure(figsize=(6,7),constrained_layout=False)
gs=fig.add_gridspec(2,4)
axs_1=fig.add_subplot(gs[0,1:])
axs_2=fig.add_subplot(gs[1,1:],sharex=axs_1,sharey=axs_1)
axs=[axs_1,axs_2]
#fig,axs=plt.subplots(2,1,figsize=(4,7),sharex=True,sharey=True,constrained_layout=False)
#fig,axs=plt.subplots(2,1,figsize=(7,8),sharex=True,sharey=True)
#axs[0].set_autoscale_on(True)
#axs[1].set_autoscale_on(True)
#axs[0] = plt.subplot2grid((2, 10), (0, 1), colspan=8)
#axs[1] = plt.subplot2grid((2, 10), (1, 1), colspan=8)

#fig.subplots_adjust(hspace=0.5)

#panel = [(0.06, 0.15, 0.25, 0.25), \
#         (0.41, 0.15, 0.25, 0.25), \
#         (0.76, 0.15, 0.25, 0.25), \
#         (0.06, 0.6, 0.25, 0.25), \
#         (0.41, 0.6, 0.25, 0.25), \
#         (0.76, 0.6, 0.25, 0.25), \
#         ]

#panel = [(0.20,0.20,0.6,0.6)]
#print(axs.shape)


for ib in range(0,2):
  ix=ib
  iy=0
#---box plot---
  #dice_0=(diffs_cld[il_1[ib]:il_2[ib],:]>-.5)*(diffs_cld[il_1[ib]:il_2[ib],:]<.5) #*(means_exp_cld[il_1[ib]:il_2[ib],:] >10.)
  #box_data=diffs_rad[il_1[ib]:il_2[ib],:][dice_0]
  #bplot=axs[ib].boxplot(box_data,positions=[0],widths=[0.7],vert=True,showmeans=True, showfliers=False,\
  #                patch_artist=True)
  #for patch, color in  zip(bplot['boxes'],'r'):
  #    patch.set_facecolor((0.7,0.4,0.3,0.3))
#-------------
  axs[ib].scatter(np.ravel(diffs_rad[:,il_1[ib]:il_2[ib],:]),\
                  np.ravel(diffs_cld[:,il_1[ib]:il_2[ib],:]),c='k',s=1)
  axs[ib].set_title(titles[ib],loc="center",fontsize=13)
  axs[ib].axhline(y=0,lw=1,c="gray")
  axs[ib].axvline(x=0,lw=1,c="gray")
  #axs[ib].set_xlim(axs[ib].get_xlim()[::-1])
  #axs[ib].set_xlim(-4,8)
  #axs[ib].set_aspect("auto")
  #axs[ib].set_ylabel(" ",fontsize=14)
  if ib ==0:
      axs[ib].tick_params(labelsize=14.0, direction='out', width=1,labelbottom=False)
  if ib ==1:
      axs[ib].tick_params(labelsize=14.0, direction='out', width=1)

  coeff[ib]=np.ma.corrcoef(np.ma.masked_invalid(diffs_rad[:,il_1[ib]:il_2[ib],:].ravel()),\
                        np.ma.masked_invalid(diffs_cld[:,il_1[ib]:il_2[ib],:].ravel()))[0,1]
  print(coeff[ib])

#axs[1].tick_params(labelsize=14.0, direction='out', width=1)
#plt.xticks([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8],["-4","-3","-2","-1","0","1","2","3","4","5","6","7","8"])
#plt.gca().invert_xaxis()
#plt.tight_layout()

#fig.text(0.5,0.94,"Band 0.44-0.63",fontsize=14,va='center',ha='center')
#fig.text(0.03,0.5,"Diff in Surface Temperature (K)",fontsize=14,va='center',ha='center',rotation='vertical')
fig.text(0.2,0.5,"Diff in Sea Ice Fraction",fontsize=14,va='center',ha='center',rotation='vertical')
fig.text(0.62,0.05,"Diff in Surface Temperature (K)",fontsize=14,va='center',ha='center')
fig.text(0.72,0.84,r"R$^2$="+str(round(coeff[0]**2,2)),fontsize=14)
fig.text(0.72,0.42,r"R$^2$="+str(round(coeff[1]**2,2)),fontsize=14)
#plt.savefig("./figures/scat_ts_fsns_polar.png",dpi=150)
plt.show()

'''
plotTitle = {'fontsize': 14.}
plotSideTitle = {'fontsize': 14.}
plotText = {'fontsize': 8.}
#panel = [(0.1691, 0.6810, 0.6465, 0.2258), \
#         (0.1691, 0.3961, 0.6465, 0.2258), \
#         (0.1691, 0.1112, 0.6465, 0.2258), \
#         ]
panel = [(0.10,0.15,0.70,0.65)]
labels=[ctl_name,exp_name,exp_name+"-"+ctl_name] 

units=r"Wm$^-$$^2$"
contour_levs=np.linspace(5,85,11) #[120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
diff_levs=np.linspace(-4,4,9) #[-50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50]
colormap="PiYG_r"
colormap_diff="bwr"

for i in range(0,1):
   #1. first plot
    levels = None
    norm = None
    cnlevels=diff_levs

    ax = fig.add_axes(panel[0],projection=ccrs.PlateCarree(central_longitude=180))

    dtplot=dtdif #[:,:]
    cmap=colormap_diff
    stats_now=stats_dif

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
    ax.set_title("Diff in CLDLOW (TSIS - CESM2)",loc="center",fontdict=plotSideTitle)
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
    cbax = fig.add_axes((panel[0][0] + 0.72, panel[0][1]+ 0.0215, 0.02, 0.6))
    cbar = fig.colorbar(p1, cax=cbax, ticks=cnlevels)
    cbar.ax.tick_params(labelsize=14.0, length=0)

    # Mean, Min, Max
    #fig.text(panel[0][0] + 0.6635, panel[0][1] + 0.2107,
    #         "Max\nMean\nMin", ha='left', fontdict=plotText)
    #fig.text(panel[0][0] + 0.7835, panel[0][1] + 0.2107, "%.2f\n%.2f\n%.2f" %
    #        (stats_now[2],stats_now[0],stats_now[1]), ha='right', fontdict=plotText)
            #stats_now[[2,0,1]], ha='right', fontdict=plotText)

    p2 = ax.contourf(lon[:],lat[:],dtdif_sig[:,:],\
                transform=projection,\
                #norm=norm,\
                levels=cnlevels,\
	    hatches=['...'], \
                cmap=cmap,\
                extend="both",\
     	       )
        
#fig.suptitle(varnm, x=0.5, y=0.96, fontsize=14)
#save figure as file
#if os.environ["fig_save"]=="True":
#    fname="d1_lon_lat_contour_"+varnm+"_"+season+"."+os.environ["fig_suffix"]
#    plt.savefig(os.environ["OUTDIR"]+"/figures/"+fname)
#plt.savefig(figure_name+".png")
#if os.environ["fig_show"]=="True":
#    plt.show()
plt.savefig("./figures/diff_cldlow_ann.eps")
plt.show()
plt.close()
'''
