# Python program to convert monthly model output to multi-year mean climat.
# The climate consistes of ANN, DJF, MAM, JJA, SON, and 01-12 months.
# NCO is used to do the average. 

import os
import numpy as np

#==========================================
# Python function to convert a list 
# to string using join() function 
def listToString(s):  
# initialize an empty string 
    str1 = ""  
# traverse in the string   
    for ele in s:  
        str1 += ele.rstrip()+" "   
# return string   
    return str1  

#==========================================
# The main program starts here.

# Input 
#caseid="E3SM_DECKv1b_H1.ne30"
#monthly_data_path="./E3SM_DECKv1b_H1.ne30/remap_180x360"
caseid=["solar_CTL_cesm211_ETEST-f19_g17-ens0",\
        "solar_CTL_cesm211_ETEST-f19_g17-ens1",\
        "solar_CTL_cesm211_ETEST-f19_g17-ens2",\
        "solar_CTL_cesm211_ETEST-f19_g17-ens3"]
monthly_data_path="/glade/u/home/xianwen/work/archive/" #+caseid+"/atm/hist/"
years=np.arange(2010,2020)
months_to_do=np.array(["01","02","03","04","05","06","07","08","09","10","11","12"])
caseout="solar_CTL_cesm211_ETEST-f19_g17-ens_mean_2010-1019"
out_path=monthly_data_path+caseout+"/"
if not os.path.exists(out_path):
     os.makedirs(out_path)
# create list of all input files
# Monthly climo
for yr in years:
    for mon in months_to_do:
        print("-- doing "+str(yr)+" "+mon+" --")
        list_file="list_"+str(yr)+"-"+mon+".txt"
        if os.path.exists(list_file):
            os.system("rm "+list_file)
        for case in caseid:
           os.system("ls "+monthly_data_path+case+"/atm/hist/*cam.h0."+str(yr)+"-"+mon+".nc|cat >>"+list_file)

        with open(list_file) as f_obj:
            lines=f_obj.readlines()
        lists=listToString(lines)
        
        climo_file=caseout+".cam.h0."+str(yr)+"-"+mon+".nc"
        cmd="ncra "+lists+" "+out_path+"/"+climo_file
        os.system(cmd)
        os.system("mv "+list_file+" "+out_path+"/")
print("----- all finished -----")
	
