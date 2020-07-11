#====================================================
#
#====================================================
# os
import os
#glob
import glob
#import netCDF4
from netCDF4 import Dataset as netcdf_dataset
# cartopy
import os
import cartopy.crs as ccrs
#from cartopy.mpl.geoaxes import GeoAxes
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.util import add_cyclic_point
# matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.path as mpath
import matplotlib.colors as colors
# numpy
import numpy as np
# parameters
from get_parameters import *
# scipy
from scipy import stats

#-----------------------
# program starts here.
#-----------------------

# data path
ctl_name="CTL" #os.environ["ctl_name"]
exp_name="TSIS" #os.environ["exp_name"]
ctl_pref="solar_CTL_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"
exp_pref="solar_TSIS_cesm211_VIS_icealb_ETEST-f19_g17-ens_mean_2010-2019"

fpath_ctl="/raid00/xianwen/cesm211_solar/"+ctl_pref+"/monthly/"
fpath_exp="/raid00/xianwen/cesm211_solar/"+exp_pref+"/monthly/"

years=np.arange(2010,2020)
months=np.array(["01","02","03","04","05","06","07","08","09","10","11","12"])
nyr=years.size
nmon=months.size

varnm="PRECL"
varnm2="PRECSL"
pole='N'
season="JJA"

fsample=sorted(glob.glob(fpath_ctl+"*-01.nc"))[0]
fs=netcdf_dataset(fsample,"r")
# read lat and lon
lat=fs.variables["lat"]
lon=fs.variables["lon"]
nlat=len(lat)
nlon=len(lon)

if pole is 'N':
   var_long_name="Arctic Rain "+season
   figure_name="Arctic_Rain_contour_"+season
   latb1=np.min(np.where(lat[:]>65))
   latb2=nlat
elif pole is 'S':
   var_long_name="Antarctic Rain "+season
   figure_name="Antarctic_Rain_contour_"+season
   latb1=0
   latb2=np.max(np.where(lat[:]<-65))+1

units="mm/day" #"Fraction"
#units=r"W/m$^2$"

#nlat=96
#nlon=144
#nlev=32

snow_yby_ctl=np.zeros((nyr,nmon)) #year by year mean for each variable
snow_yby_exp=np.zeros((nyr,nmon)) #year by year mean for each variable
rain_yby_ctl=np.zeros((nyr,nmon)) #year by year mean for each variable
rain_yby_exp=np.zeros((nyr,nmon)) #year by year mean for each variable
snow_ctl=np.zeros((nmon)) #year by year mean for each variable
snow_exp=np.zeros((nmon)) #year by year mean for each variable
rain_ctl=np.zeros((nmon)) #year by year mean for each variable
rain_exp=np.zeros((nmon)) #year by year mean for each variable
diffs_snow=np.zeros((nmon)) #multi-year exp-ctl diff for each variable
pvals_snow=np.zeros((nmon)) #pvalues of ttest
diffs_rain=np.zeros((nmon)) #multi-year exp-ctl diff for each variable
pvals_rain=np.zeros((nmon)) #pvalues of ttest

# Start loop through months 
for im in range(0,nmon):
    fsctl=sorted(glob.glob(fpath_ctl+"*"+months[im]+".nc"))
    fsexp=sorted(glob.glob(fpath_exp+"*"+months[im]+".nc"))

   # compute domain mean for each year
    for iy in range(0,nyr):
        # open data file
        file_ctl=netcdf_dataset(fsctl[iy],"r")
        file_exp=netcdf_dataset(fsexp[iy],"r")

      # rain
        snow_ctl=file_ctl.variables["PRECSL"][0,:,:]
        snow_exp=file_exp.variables["PRECSL"][0,:,:]
      # snow
        rain_ctl=file_ctl.variables["PRECL"][0,:,:]-snow_ctl
        rain_exp=file_exp.variables["PRECL"][0,:,:]-snow_exp
  
      # change units from m/s to mm/day:
        snow_ctl=snow_ctl * 24.*3600.*1000.
        snow_exp=snow_exp * 24.*3600.*1000.
        rain_ctl=rain_ctl * 24.*3600.*1000.
        rain_exp=rain_exp * 24.*3600.*1000.

       # mask land (use ocean only)
        ocnf_ctl=file_ctl.variables["OCNFRAC"][0,:,:]
        ocnf_exp=file_exp.variables["OCNFRAC"][0,:,:]
        snow_ctl=np.ma.masked_array(snow_ctl,ocnf_ctl>0.99)
        snow_exp=np.ma.masked_array(snow_exp,ocnf_exp>0.99)
        rain_ctl=np.ma.masked_array(rain_ctl,ocnf_ctl>0.99)
        rain_exp=np.ma.masked_array(rain_exp,ocnf_exp>0.99)
         
        snow_yby_ctl[iy,im]=get_area_mean_min_max(snow_ctl[latb1:latb2,:],lat[latb1:latb2])[0]
        snow_yby_exp[iy,im]=get_area_mean_min_max(snow_exp[latb1:latb2,:],lat[latb1:latb2])[0]
        rain_yby_ctl[iy,im]=get_area_mean_min_max(rain_ctl[latb1:latb2,:],lat[latb1:latb2])[0]
        rain_yby_exp[iy,im]=get_area_mean_min_max(rain_exp[latb1:latb2,:],lat[latb1:latb2])[0]

# END of month & year loops

# year-by-year to multiannual mean
snow_ctl=np.mean(snow_yby_ctl,axis=0)
snow_exp=np.mean(snow_yby_exp,axis=0)
rain_ctl=np.mean(rain_yby_ctl,axis=0)
rain_exp=np.mean(rain_yby_exp,axis=0)

# rain fraction in total precipitation
rfrac_yby_ctl=rain_yby_ctl/(rain_yby_ctl+snow_yby_ctl)*100.  # rain partition in all precp
rfrac_yby_exp=rain_yby_exp/(rain_yby_exp+snow_yby_exp)*100.  # rain partition in all precp
rfrac_ctl=np.mean(rfrac_yby_ctl,axis=0)
rfrac_exp=np.mean(rfrac_yby_exp,axis=0)

# multiannual mean difference
diffs_snow=snow_exp-snow_ctl
diffs_rain=rain_exp-rain_ctl
diffs_rfrac=rfrac_exp-rfrac_ctl

# t-test
siglev=0.05
ttest=stats.ttest_ind(snow_yby_ctl,snow_yby_exp,axis=0)
pvalues_snow=ttest.pvalue
ttest=stats.ttest_ind(rain_yby_ctl,rain_yby_exp,axis=0)
pvalues_rain=ttest.pvalue
ttest=stats.ttest_ind(rfrac_yby_ctl,rfrac_yby_exp,axis=0)
pvalues_rfrac=ttest.pvalue

diffs_snow_sig=np.zeros(nmon)
diffs_rain_sig=np.zeros(nmon)
diffs_rfrac_sig=np.zeros(nmon)
diffs_snow_sig[:]=np.nan
diffs_rain_sig[:]=np.nan
diffs_rfrac_sig[:]=np.nan

for im in range(0,nmon):
    if pvalues_snow[im] < siglev:
        diffs_snow_sig[im]=diffs_snow[im]
    if pvalues_rain[im] < siglev:
        diffs_rain_sig[im]=diffs_rain[im]
    if pvalues_rfrac[im] < siglev:
        diffs_rfrac_sig[im]=diffs_rfrac[im]

#-------------------
# plot starts here
#-------------------

fig = plt.figure(figsize=[8.0,11.0],dpi=150.)

panel = [(0.2, 0.6, 0.6, 0.3),\
         (0.2, 0.2, 0.6, 0.3)]

ax1 = fig.add_axes(panel[0])
ax2 = fig.add_axes(panel[1])
# draw snow/rain values
ax1.plot(months,snow_ctl[:],color='b',lw=1,ls='-',label='snow (CTL)')
ax1.plot(months,snow_exp[:],color='b',lw=1,ls=':',label='snow (TSIS)')
ax1.plot(months,rain_ctl[:],color='r',lw=1,ls='-',label='rain (CTL)')
ax1.plot(months,rain_exp[:],color='r',lw=1,ls=':',label='rain (TSIS)')
ax1.set_xlim(0,11)

ax1.set_ylabel('Prec (mm/day)',color='k')
# draw differences on top of true values
#ax1_r=ax1.twinx()
#ax1_r.plot(months,diffs_snow[:],color='b',lw=2,ls='--',label='snow (TSIS-CTL)')
#ax1_r.plot(months,diffs_rain[:],color='r',lw=2,ls='--',label='rain (TSIS-CTL)')
ax1.legend(fontsize=8)

# draw fraction
#ax2.plot(months,diffs_snow[:],color='b',lw=2,ls='--',label='snow (TSIS-CTL)')
#ax2.plot(months,diffs_rain[:],color='r',lw=2,ls='--',label='rain (TSIS-CTL)')
#ax2.plot(months,np.zeros((nmon)),color='gray',lw=1,ls='-')
#ax2.plot(months,range(0,nmon),color='gray',lw=1,ls='-')
#ax2_r=ax2.twinx()
ax2.plot(months,diffs_rfrac[:],color='g',lw=2,ls='--',label='rain fraction (TSIS-CTL)')
ax2.plot(months,np.zeros((nmon)),color='g',lw=1,ls=':')
ax2.legend(fontsize=8)
#ax2_r.legend(fontsize=8)

ax2.set_xlim(0,11)
#ax2.set_ylabel('Diff in Prec (mm/day)',color='k')
ax2.set_ylabel('Rain fraction (%)',color='k')
ax2.set_xlabel('Months',color='k')
plt.show()

