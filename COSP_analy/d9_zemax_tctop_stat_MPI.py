#=================================================
# This code compute and plot CFODD for mixed-phase clouds. 
# Required input:
#   1) Radar reflectivity (from 2B-GEOPROF)
#   2) Height (from 2B-GEOPROF)
#   3) Cloud top temperature (from ECWMF-AUX)
#   4) cloud phase and supercooled water layer top (from 2B-CLDCLASS-LIDAR) 
#   5) LWP (from MODIS-5KM-AUX, MODIS-1KM-AUX)
#   6) Cloud top droplet effective radius (from MODIS-5KM-AUX, MODIS-1KM-AUX)
# History:
#   1) JAN 30, 2020, First created. Xianwen.
#=================================================

import os
import glob
import numpy as np
#from hdf_eos_utils import *
from netCDF4 import Dataset as netcdf_dataset

# 
#------------------------
#-- creat Ze vs T bins-- 
#------------------------
num_zebin=25
zebnd=np.linspace(-30.,20.,num_zebin+1)
num_tcbin=20
tcbnd=np.linspace(-40.,0.,num_tcbin+1)

cnt_samp_N1=np.zeros((2,num_tcbin,num_zebin),dtype=np.int64) # counted number of samples
cnt_samp_N2=np.zeros((2,num_tcbin,num_zebin),dtype=np.int64) # counted number of samples
cnt_samp_N3=np.zeros((2,num_tcbin,num_zebin),dtype=np.int64) # counted number of samples
cnt_samp_S1=np.zeros((2,num_tcbin,num_zebin),dtype=np.int64) # counted number of samples
cnt_samp_S2=np.zeros((2,num_tcbin,num_zebin),dtype=np.int64) # counted number of samples
cnt_samp_S3=np.zeros((2,num_tcbin,num_zebin),dtype=np.int64) # counted number of samples

#------------------------
#--output file location--
#------------------------
out_path='./results_ocn_lnd_noprecp_ice95/'
if not os.path.exists(out_path):
   os.mkdir(out_path)
   print('Created output directory: '+out_path)

#------------------------
# set data path and file name prefix
#------------------------
data_path='/glade/scratch/xianwen/archive/cesm211_FHIST-f09_f09_mg17-COSP_runall/atm/hist/'
months=['01','02','03','04','05','06','07','08','09','10','11','12']

#------------------------
# get dimention info from a data file.
#------------------------
fsample=sorted(glob.glob(data_path+"*2001-01-01*.nc"))
ftmp=netcdf_dataset(flist[0],"r")
time = ftmp.variables["time"]
lat  = ftmp.variables["lat"]
lon  = ftmp.variables["lon"]
lev  = ftmp.variables["lev"]
scol = ftmp.variables["cosp_scol"]
ntime = len(time)
nlat  = len(lat)
nlon  = len(lon)
nlev  = len(lev)
nscol = len(scol)
#print(lat[:])
#exit()

#------------------------
# loop throug months and files
#------------------------
for im in months:
    flist=sorted(glob.glob(data_path+"*2001-"+im+"*.nc"))
    out_path_im= out_path+im+'/'
    if not os.path.exists(out_path_im):
       os.mkdir(out_path_im)
       print('Created output directory: '+out_path_im)

    for f in flist:
       print(f)
    # read data
       fnow     = netcdf_dataset(f,"r")  
       tair     = fnow.variables["T"][:,:,:,:]            #[time,lev,lat,lon]
       ctot_ice = fnow.variables["CLDTOT_CAL_ICE"][:,:,:] #[time,lat,lon]
       ctot_liq = fnow.variables["CLDTOT_CAL_LIQ"][:,:,:] #[time,lat,lon]
       landfrac = fnow.variables["LANDFRAC"][:,:,:]       #[time,lat,lon]
       ocnfrac  = fnow.variables["OCNFRAC"][:,:,:]        #[time,lat,lon]
       icefrac  = fnow.variables["ICEFRAC"][:,:,:]        #[time,lat,lon]
       dbze     = fnow.variables["DBZE_CS"][:,:,:,:,:]    #[time,lev,cosp_scol,lat,lon]
       scop     = fnow.variables["SCOPS_OUT"][:,:,:,:,:]  #[time,lev,cosp_scol,lat,lon]
    # sample 
       strcld_flag = scop==1   # stratiform clouds only
       cnvcld_flag = scop==2   # convective clouds only
       mixphs_flag = (ctot_ice > 95.) #* (ctot_liq > 10.)   # mixed phase clouds only
       #print(scop[0,:,0,20,0])
       #print(scop[0,:,1,20,0])
       #print(landfrac[0,:,0])
       #print(ctot_liq[0,:,0])
       #exit()
       #mixphs_flag_exp = np.empty((strcld_flag.shape),dtype=np.ma.core.MaskedArray)
       #for ilev in range(nlev):
       #   for iscol in range(nscol): 
       #      mixphs_flag_exp[:,ilev,iscol,:,:]=mixphs_flag[:,:,:]
       
       #print(type(strcld_flag))
       #print(mixphs_flag_exp.shape)
       #print(type(mixphs_flag_exp))
       #print(mixphs_flag_exp[2,20,5,10:30,10])
       #print(mixphs_flag[2,10:30,10])
    
       for itime in range(ntime):
          for ilat in range(nlat):
             for ilon in range(nlon):
                 if mixphs_flag[itime,ilat,ilon] and strcld_flag[itime,:,:,ilat,ilon].any():
                    #dbze_tmp = np.ma.masked_array(dbze[itime,:,:,ilat,ilon], \
                    #           np.bitwise_not(strcld_flag[itime,:,:,ilat,ilon]), fill_value=99999)
                    dbze_tmp = np.where(cnvcld_flag[itime,:,:,ilat,ilon],-9999.,dbze[itime,:,:,ilat,ilon])
                    dbze_1col=dbze_tmp.max(axis=1)  # use sub-column maximum ze at each layer 
    
               # skip precipitating (surface) cases
                    if dbze_1col[-1] > -15.:  
                        continue
    
               # single layer check (skip non-single-layer):
                    nclays=len(dbze_1col[(-35.<dbze_1col) & (dbze_1col<30.)]) #cld layers
                    if nclays ==0:
                        continue
                    idclays=np.where((-35.<dbze_1col) & (dbze_1col<30.))    #cld location
                    nvext=max(idclays[0])-min(idclays[0])+1 # cld top to base extent
                    if nclays < (max(idclays[0])-min(idclays[0])+1):
                        continue
                
               # surface type (isfc=0, land; 1, ocean)
                    if landfrac[itime,ilat,ilon] > 0.99:
                        isfc=0
                    elif icefrac[itime,ilat,ilon]+ocnfrac[itime,ilat,ilon]>0.99:
                        isfc=1
                    else:
                        continue  # not complete land or ocean
    
               # compute maximum ze and cloud top temperature
                    dbze_max = dbze_1col.max() # maximum ze over all levels
                    for ilev in range(nlev):
                        if dbze_1col[ilev] > -35.: 
                            tctop = tair[itime,ilev,ilat,ilon] - 273.15
    
               # locate ze and tctop in the tctop-zemax table
                    ize = zebnd[zebnd<=dbze_max].size-1
                    itc = tcbnd[tcbnd<=tctop].size-1
                    #print(ize,itc)
                    if ize > num_zebin-1 or ize <  0: # if ze >20. or ze<-30.
                        continue
                    if itc > num_tcbin-1 or itc <  0: # if tc >0. or tc<-40.
                        continue
    
                    #print(isfc,itc,ize)
    
                    if lat[ilat] >= 0. and lat[ilat] <30.:
                        cnt_samp_N1[isfc,itc,ize] = cnt_samp_N1[isfc,itc,ize]+1
                    elif lat[ilat] >= 30. and lat[ilat] <60.:
                        cnt_samp_N2[isfc,itc,ize] = cnt_samp_N2[isfc,itc,ize]+1
                    elif lat[ilat] >= 60.:
                        cnt_samp_N3[isfc,itc,ize] = cnt_samp_N3[isfc,itc,ize]+1
                    elif lat[ilat] >= -30. and lat[ilat] <0.:
                        cnt_samp_S1[isfc,itc,ize] = cnt_samp_S1[isfc,itc,ize]+1
                    elif lat[ilat] >= -60. and lat[ilat] <-30.:
                        cnt_samp_S2[isfc,itc,ize] = cnt_samp_S2[isfc,itc,ize]+1
                    elif lat[ilat] < -60.:
                        cnt_samp_S3[isfc,itc,ize] = cnt_samp_S3[isfc,itc,ize]+1
    
                    #print(dbze_1col.shape)
                    #print(dbze_1col[:])
                    #exit()
    
    #
    #==============================================
    fout=open(out_path_im+'cnt_cld_lnd_NH_0-30.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_N1[0,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_lnd_NH_30-60.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_N2[0,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_lnd_NH_60-90.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_N3[0,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_lnd_SH_0-30.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_S1[0,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_lnd_SH_30-60.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_S2[0,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_lnd_SH_60-90.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_S3[0,il,:])+'\n')
    fout.close()
    
    #--------
    fout=open(out_path_im+'cnt_cld_ocn_NH_0-30.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_N1[1,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_ocn_NH_30-60.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_N2[1,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_ocn_NH_60-90.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_N3[1,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_ocn_SH_0-30.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_S1[1,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_ocn_SH_30-60.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_S2[1,il,:])+'\n')
    fout.close()
    
    fout=open(out_path_im+'cnt_cld_ocn_SH_60-90.txt','w')
    for il in range(num_tcbin):
          fout.write(str(cnt_samp_S3[1,il,:])+'\n')
    fout.close()
