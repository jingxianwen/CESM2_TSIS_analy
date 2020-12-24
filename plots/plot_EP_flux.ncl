;----------------------------------------------------------------------
; epflux_2.ncl
;
; Concepts illustrated:
;   - Reading variables
;   - Using cd_inv_calendar and indexing subscripting to access a time segment
;   - Using the 'epflux' to calculate EP-fluxes
;   - Extracting data from a variable of type 'list'
;   - Plotting vectors and overlaying contours 
;=======================================================================
;                  MAIN: 6.4.0 or later
;=======================================================================
; All needed libraries are automatically loaded
;=======================================================================

;---Set options

  show_accel = 0                 ; plot option
;  dataflag   = "dailyavg"        ; '4xdaily' or 'dailyavg'

  pltDir     = "./figures/"              ; dir for output plot
  pltType    = "x11"             ; getenv("pltType")
  pltRoot    = "epflux"          ; getenv("pltRoot")

;---Set date explicitly or via  environment variables

  years=(/"2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"/)
  season = "ANN"
  ;print(season)
  ;if (season .eq. "DJF") then 
  ;   months=(/"12","01","02"/)
  ;   print(months)
  ;else if (season .eq. "JJA") then 
  ;   months=(/"06","07","08"/)
  ;end if
     months=(/"01","02","03","04","05","06","07","08","09","10","11","12"/)

; open files and get some basic coordinate information.  
; file names; open each; size information  

; CESM2 case

nn = 0.0

do iy=0,9
  do im=0,11

  dir     = "/raid00/xianwen/data/cesm211_solar_exp/solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019/monthly/"                      ; directory with files
  case     = "solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
  infil     = dir+case+".cam.h0."+years(iy)+"-"+months(im)+".nc"

  ;print(infil) 

  inf       = addfile(infil, "r")    ; file references (pointers)

; import the desired variables for the specified time period
                                                      ; (time,level,lat,lon)
  U       = inf->U(:,:,:,:)    ; m/s
  V       = inf->V(:,:,:,:)
  T       = inf->T(:,:,:,:)    ; degK

;---Compute EP-Flux and other quantities

  lat     = U&lat
  plvl    = U&lev 
  sf      =    5.0        ; tofloat(getenv("sc_fact_start"))
                          ; stratospheric magnification factor (plot) 
                          ; NOAA used an environment variable
  opt     = True          
  opt@magf= sf            ; make NCL attribute for use by 'epflux'
  opt@print_var_info = False   ; True (print); False (no Print)

  epf     =  epflux(U,V,T,plvl,lat,opt)   

  Fphi_tmp  = epf[0]          ; extract variables from 'list' for clarity
  Fp_tmp    = epf[1]
  EPdiv_tmp = epf[2]
  dudt_tmp  = epf[3]
  delete(epf)             ; delete list variable; no longer needed
                          ; cursory overview of variable contents

  if (nn .eq. 0.0) then 
    Fphi_1 = Fphi_tmp 
    Fp_1   = Fp_tmp
    EPdiv_1 = EPdiv_tmp
    dudt_1 = dudt_tmp
  else 
    Fphi_1 = Fphi_1 + Fphi_tmp 
    Fp_1   = Fp_1 + Fp_tmp
    EPdiv_1 = EPdiv_1 + EPdiv_tmp
    dudt_1 = dudt_1 + dudt_tmp
  end if

  nn = nn + 1.0

  end do
end do

Fphi_1 = Fphi_1 / nn
Fp_1 = Fp_1 / nn
EPdiv_1 = EPdiv_1 / nn
dudt_1 = dudt_1 / nn

copy_VarMeta(Fphi_tmp,Fphi_1)
copy_VarMeta(Fp_tmp,Fp_1)
copy_VarMeta(EPdiv_tmp,EPdiv_1)
copy_VarMeta(dudt_tmp,dudt_1)

; TSIS case 

nn = 0.0

do iy=0,9
  do im=0,11

  dir     = "/raid00/xianwen/data/cesm211_solar_exp/solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019/monthly/"                      ; directory with files
  case     = "solar_TSIS_cesm211_ETEST-f19_g17-ens_mean_2010-2019"
  infil     = dir+case+".cam.h0."+years(iy)+"-"+months(im)+".nc"

  ;print(infil) 

  inf       = addfile(infil, "r")    ; file references (pointers)

; import the desired variables for the specified time period
                                                      ; (time,level,lat,lon)
  U       = inf->U(:,:,:,:)    ; m/s
  V       = inf->V(:,:,:,:)
  T       = inf->T(:,:,:,:)    ; degK

;---Compute EP-Flux and other quantities

  lat     = U&lat
  plvl    = U&lev 
  sf      =    5.0        ; tofloat(getenv("sc_fact_start"))
                          ; stratospheric magnification factor (plot) 
                          ; NOAA used an environment variable
  opt     = True          
  opt@magf= sf            ; make NCL attribute for use by 'epflux'
  opt@print_var_info = False   ; True (print); False (no Print)

  epf     =  epflux(U,V,T,plvl,lat,opt)   

  Fphi_tmp  = epf[0]          ; extract variables from 'list' for clarity
  Fp_tmp    = epf[1]
  EPdiv_tmp = epf[2]
  dudt_tmp  = epf[3]
  delete(epf)             ; delete list variable; no longer needed
                          ; cursory overview of variable contents

  if (nn .eq. 0.0) then 
    Fphi_2 = Fphi_tmp 
    Fp_2   = Fp_tmp
    EPdiv_2 = EPdiv_tmp
    dudt_2 = dudt_tmp
  else 
    Fphi_2 = Fphi_2 + Fphi_tmp 
    Fp_2   = Fp_2 + Fp_tmp
    EPdiv_2 = EPdiv_2 + EPdiv_tmp
    dudt_2 = dudt_2 + dudt_tmp
  end if

  nn = nn + 1.0

  end do
end do

Fphi_2 = Fphi_2 / nn
Fp_2 = Fp_2 / nn
EPdiv_2 = EPdiv_2 / nn
dudt_2 = dudt_2 / nn

copy_VarMeta(Fphi_tmp,Fphi_2)
copy_VarMeta(Fp_tmp,Fp_2)
copy_VarMeta(EPdiv_tmp,EPdiv_2)
copy_VarMeta(dudt_tmp,dudt_2)

; difference

Fphi = Fphi_2-Fphi_1
Fp = Fp_2-Fp_1
EPdiv = EPdiv_2-EPdiv_1
dudt = dudt_2-dudt_1

Fphi = Fphi_2 ;-Fphi_1
Fp = Fp_2 ;-Fp_1
EPdiv = EPdiv_2 ;-EPdiv_1
dudt = dudt_2 ;-dudt_1

copy_VarMeta(Fphi_tmp,Fphi)
copy_VarMeta(Fp_tmp,Fp)
copy_VarMeta(EPdiv_tmp,EPdiv)
copy_VarMeta(dudt_tmp,dudt)

  ;************************************************
  ; Create Plot 
  ;************************************************
  ; create vector plot resources for pressure-level grid 
  ;************************************************
    res_vec = True
    res_vec@gsnMaximize           = True              ; make ps/eps/pdf large (no effect otherwise)
    res_vec@gsnDraw               = False             ; allows for manual overlaying        
    res_vec@gsnFrame              = False        

    res_vec@vfXArray              = lat               ; use lat for x axis
    res_vec@vfYArray              = plvl              ; use pressure for y axis
    res_vec@trYReverse            = True              ; reverse y-axis
    res_vec@gsnYAxisIrregular2Log = True              ; set y-axis to log scale

    res_vec@tiXAxisString         = "latitude"        ; x-axis label
    res_vec@tiYAxisString         = "pressure (mb)"   ; y-axis label
    res_vec@tiXAxisFontHeightF    = 0.0175
    res_vec@tiYAxisFontHeightF    = 0.0175

    res_vec@vcRefMagnitudeF       = 200               ; add a reference vector
    res_vec@vcRefLengthF          = 0.05              ; what the ref length is
    res_vec@vcMonoLineArrowColor  = False             ; vec's colored by their mag
    res_vec@vcLevelPalette        = "rainbow"         
    res_vec@vcLevelSelectionMode  = "ManualLevels"
    res_vec@vcLevelSpacingF       = 20.0
    res_vec@vcMinLevelValF        = 0.0
    res_vec@vcMaxLevelValF        = 180.0  
    res_vec@vcRefAnnoOn           = False              ; turn off ref wind barb
    res_vec@vcMinDistanceF        = 0.00875            ; trial and error
  
    res_vec@pmLabelBarDisplayMode = "Always"          ; Turn on a label bar.
    res_vec@pmLabelBarWidthF      = 0.08              ; make it thinner
    res_vec@lbPerimOn             = False             ; no box around it

    res_vec@tiMainString          = "TSIS-1 (" + season +")"       ; plot title
    res_vec@tiMainFontHeightF     = 0.0185

    res_vec@tmXBLabelFontHeightF  = 0.0125
    res_vec@tmYLLabelFontHeightF  = 0.0125
    res_vec@tmXBMajorLengthF      = -0.0075           ; minus mean outward face
    res_vec@tmYLMajorLengthF      = -0.0075           ; minus mean outward face

    res_vec@tmYLMode              = "Explicit"        ; Pressure (YL) axis
    res_vec@tmYLValues            = plvl            
    res_vec@tmYLLabels            = tostring(toint(plvl)) 
    res_vec@tmYLLabels(1)         = ""                ; no 925 label
    res_vec@tmYLLabels(2)         = ""                ;    850
    res_vec@tmYLLabels(4)         = ""                ;    600
    res_vec@tmYLLabels(8)         = ""                ;    250
  
    res_vec@vpWidthF              = 0.60              ; shape
    res_vec@vpHeightF             = 0.35

; Create contour plot resources

    res_con                       = True
    res_con@gsnDraw               = False
    res_con@gsnFrame              = False

    res_con@sfXArray              = res_vec@vfXArray   ; =lat 
    res_con@sfYArray              = res_vec@vfYArray   ; =plvl
    res_con@trYReverse            = True               ; reverse y-axis

    res_con@gsnYAxisIrregular2Log = True               ; set y-axis to log scale
    res_con@gsnContourZeroLineThicknessF = 0.0
    res_con@gsnContourPosLineDashPattern = 0
    res_con@gsnContourNegLineDashPattern = 2

    res_con@cnSmoothingOn         = True
    res_con@cnLineColor           = "black"      
    res_con@cnLineThicknessF      = 2.0                ; default is 1.0
   ;res_con@gsnContourLineThicknessesScale  = 0.5
    res_con@cnLineLabelsOn        = False
   ;res_con@cnInfoLabelOn         = False              ; default is True
  
; open file and create graphic

  wks     = gsn_open_wks(pltType,pltDir)                
  Fp(0,:) = Fp@_FillValue                          ; Hide the 1000 mb level
  plotvec = gsn_vector(wks,Fphi,Fp,res_vec)    

  if (show_accel .eq. 1) then 
      dudt@_FillValue = -999.0
      dudt(0,:) = dudt@_FillValue                  ; Hide the 1000 mb level
      res_con@cnLevelSpacingF        = 5.          ; Contour level Spacing
      plotvec2 = gsn_contour(wks,dudt,res_con)     ; Creates plot for du/dt = div(F)/(a*cos(phi))
      overlay(plotvec,plotvec2)
  end if
  if (show_accel .eq. 0) then 
      EPdiv@_FillValue = -999.0
      EPdiv(0,:) = EPdiv@_FillValue                ; Hide the 1000 mb level
      res_con@cnLevelSpacingF        = 200.        ; Contour level Spacing
      plotvec3 = gsn_contour(wks,EPdiv,res_con)    ; Creates plot for div(F)
      overlay(plotvec,plotvec3)
  end if

  draw(plotvec)
  frame(wks)
  
  print ( "done" )
