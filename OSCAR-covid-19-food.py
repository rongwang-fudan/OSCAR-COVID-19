"""
Copyright: IIASA (International Institute for Applied Systems Analysis), 2016-2018; CEA (Commissariat a L'Energie Atomique) & UVSQ (Universite de Versailles et Saint-Quentin), 2016
Contributor(s): Thomas Gasser (gasser@iiasa.ac.at)

This software is a computer program whose purpose is to simulate the behavior of the Earth system, with a specific but not exclusive focus on anthropogenic climate change.

This software is governed by the CeCILL license under French law and abiding by the rules of distribution of free software.  You can use, modify and/ or redistribute the software under the terms of the CeCILL license as circulated by CEA, CNRS and INRIA at the following URL "http://www.cecill.info". 

As a counterpart to the access to the source code and rights to copy, modify and redistribute granted by the license, users are provided only with a limited warranty and the software's author, the holder of the economic rights, and the successive licensors have only limited liability. 

In this respect, the user's attention is drawn to the risks associated with loading, using, modifying and/or developing or reproducing the software by the user in light of its specific status of free software, that may mean that it is complicated to manipulate, and that also therefore means that it is reserved for developers and experienced professionals having in-depth computer knowledge. Users are therefore encouraged to load and test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or data to be ensured and,  more generally, to use and operate it in the same conditions as regards security. 

The fact that you are presently reading this means that you have had knowledge of the CeCILL license and that you accept its terms.
"""


##################################################
##################################################
##################################################


import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties


##################################################
#   1. OSCAR LITE
##################################################

def OSCAR_lite(p=p,fT=fT,\
               EFF=EFF,ECH4=ECH4,EN2O=EN2O,\
               LUC=LUC,HARV=HARV,SHIFT=SHIFT,\
               EHFC=EHFC,EPFC=EPFC,EODS=EODS,\
               ENOX=ENOX,ECO=ECO,EVOC=EVOC,ESO2=ESO2,ENH3=ENH3,EOC=EOC,EBC=EBC,\
               RFcon=RFcon,RFvolc=RFvolc,RFsolar=RFsolar,\
               force_CO2=False,force_GHG=False,force_halo=False,force_RF=False,force_RFs=False,force_clim=False,\
               var_output=['ELUC','OSNK','LSNK','D_CO2','RF','D_gst','D_EBB_CO2','D_EBB_CH4','D_EBB_N2O','NPP_crop','YieldTot',\
                           'D_EBB_NOX','D_EBB_CO','D_EBB_VOC','D_EBB_SO2','D_EBB_NH3','D_EBB_OC','D_EBB_BC','NPP',\
                           'YieldCot','YieldMai','YieldMai','YieldRic','YieldWhe','YieldRest',\
                           'AreaCot','AreaMai','AreaRub','AreaRic','AreaWhe','AreaRest',\
                           'Cumemission','CumN2O','CumCH4','REFF','EFNPK','EFNPKT','REFFCN','BECCSall','CovidFF','REN2O','RECH4'],\
               plot=[]):

    #===============
    # A. DEFINITIONS
    #===============
    # plot variables
    var_plot = []
    if plot is 'all' or plot is 'CO2' or 'CO2' in plot:
        var_plot += ['D_CO2','OSNK','LSNK','ELUC','D_AREA','D_npp','D_efire','D_fmort','D_rh1','D_fmet','D_rh2','Diff','D_FIN','D_FOUT','D_FCIRC','EFIRE_luc','FMORT_luc','RH1_luc','FMET_luc','RH2_luc','EHWP1_luc','EHWP2_luc','EHWP3_luc','bLSNK','b1ELUC','b2ELUC','b1RH1_luc','b2RH1_luc','b1RH2_luc','b2RH2_luc','b1EFIRE_luc','b2EFIRE_luc','b1EHWP1_luc','b2EHWP1_luc','b1EHWP2_luc','b2EHWP2_luc','b1EHWP3_luc','b2EHWP3_luc']
    if plot is 'all' or plot is 'CH4' or 'CH4' in plot:
        var_plot += ['D_CH4','D_OHSNK_CH4','D_HVSNK_CH4','D_XSNK_CH4','D_EWET','D_EBB_CH4']
    if plot is 'all' or plot is 'N2O' or 'N2O' in plot:
        var_plot += ['D_N2O','D_HVSNK_N2O','D_EBB_N2O']
    if plot is 'all' or plot is 'O3' or 'O3' in plot:
        var_plot += ['D_O3t','D_O3s','D_EESC','D_N2O_lag','D_gst']
    if plot is 'all' or plot is 'AER' or 'AER' in plot:
        var_plot += ['D_SO4','D_POA','D_BC','D_NO3','D_SOA','D_AERh','RF_SO4','RF_POA','RF_BC','RF_NO3','RF_SOA','RF_cloud']
    if plot is 'all' or plot is 'clim' or 'clim' in plot:
        var_plot += ['RF','D_gst','D_gyp','RF_CO2','RF_CH4','RF_H2Os','RF_N2O','RF_halo','RF_O3t','RF_O3s','RF_SO4','RF_POA','RF_BC','RF_NO3','RF_SOA','RF_cloud','RF_BCsnow','RF_LCC']

    # save variables
    var_timeseries = list(set(var_output)|set(var_plot))
    for var in var_timeseries:
        # global variables
        if var in ['D_mld','D_dic','D_pH']\
        or var in ['OSNK','LSNK','D_OHSNK_CH4','D_HVSNK_CH4','D_XSNK_CH4','D_HVSNK_N2O','ProdGlobalCap','EnerDCGlo','ProdDCGloCap','BECCSall','LSNKfo','LSNKcr4','YieldTot','Cumemission','CumN2O','CumCH4','REFF','EFNPK','EFNPKT','REFFCN','CovidFF','REN2O','RECH4']\
        or var in ['D_kOH','D_hv']\
        or var in ['D_O3t','D_EESC','D_O3s','D_SO4','D_POA','D_BC','D_NO3','D_SOA','D_AERh']\
        or var in ['D_CO2','D_CH4','D_CH4_lag','D_N2O','D_N2O_lag']\
        or var in ['RF','RF_warm','RF_atm','RF_CO2','RF_CH4','RF_H2Os','RF_N2O','RF_halo','RF_O3t','RF_O3s','RF_SO4','RF_POA','RF_BC','RF_NO3','RF_SOA','RF_cloud','RF_BCsnow','RF_LCC']\
        or var in ['D_gst','D_sst','D_gyp','D_OHC']:
            exec(var+'_t = np.zeros([ind_final+1],dtype=dty)')
        # (region) variables
        if var in ['ELUC','D_AWET','D_EWET','D_ewet','D_EBB_CO2','D_EBB_CH4','D_EBB_N2O','D_EBB_NOX','D_EBB_CO','D_EBB_VOC','D_EBB_SO2','D_EBB_NH3','D_EBB_OC','D_EBB_BC','D_lst','D_lyp','NPP_crop']:
            exec(var+'_t = np.zeros([ind_final+1,nb_regionI],dtype=dty)')        
        # (region)*(biome) variables
        if var in ['D_AREA','D_npp','D_efire','D_fmort','D_rh1','D_fmet','D_rh2','Diff','D_cveg','D_eharv','D_csoil1','D_csoil2','NPP','bLSNK','b1ELUC','b2ELUC','b1RH1_luc','b2RH1_luc','b1RH2_luc','b2RH2_luc','b1EFIRE_luc','b2EFIRE_luc','b1EHWP1_luc','b2EHWP1_luc','b1EHWP2_luc','b2EHWP2_luc','b1EHWP3_luc','b2EHWP3_luc']:
            exec(var+'_t = np.zeros([ind_final+1,nb_regionI,nb_biome],dtype=dty)')
        # (region)*(biome)*(biome)*(age) variables
        if var in ['EFIRE_luc','FMORT_luc','RH1_luc','FMET_luc','RH2_luc','EHWP1_luc','EHWP2_luc','EHWP3_luc']\
        or var in ['CVEG_luc','CSOIL1_luc','CSOIL2_luc','CHWP1_luc','CHWP2_luc','CHWP3_luc']:
            exec(var+'_t = np.zeros([ind_final+1,nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty)')        
        # (obox) variables
        if var in ['D_FIN','D_FOUT','D_FCIRC','D_CSURF']:
            exec(var+'_t = np.zeros([ind_final+1,nb_obox],dtype=dty)')
        # (species) variables
        if var in ['D_HFC','D_HFC_lag','D_OHSNK_HFC','D_HVSNK_HFC','D_XSNK_HFC']:
            exec(var+'_t = np.zeros([ind_final+1,nb_HFC],dtype=dty)')
        if var in ['D_PFC','D_PFC_lag','D_OHSNK_PFC','D_HVSNK_PFC','D_XSNK_PFC']:
            exec(var+'_t = np.zeros([ind_final+1,nb_PFC],dtype=dty)')
        if var in ['D_ODS','D_ODS_lag','D_OHSNK_ODS','D_HVSNK_ODS','D_XSNK_ODS']:
            exec(var+'_t = np.zeros([ind_final+1,nb_ODS],dtype=dty)')
        # (country) variables
        if var in ['YieldCot','YieldMai','YieldMai','YieldMaiF','YieldMaiC','YieldRic','YieldWhe','YieldRest',\
                   'AreaCot','AreaMai','AreaMaiF','AreaMaiC','AreaRub','AreaRic','AreaWhe','AreaRest',\
                   'ProdCot','ProdMai','ProdMaiF','ProdMaiC','ProdRub','ProdRic','ProdWhe','ProdRest','ProdDC_cou']:
            exec(var+'_t = np.zeros([ind_final+1,171],dtype=dty)') # 171 represent 171 countries
    
        

    # run variables
    # ocean
    D_dic = np.array([0],dtype=dty)
    D_CSURF = np.zeros([nb_obox],dtype=dty)
    # land
    for var in ['D_AREA','D_cveg','D_csoil1','D_csoil2','D_eharv']:
        exec(var+' = np.zeros([nb_regionI,nb_biome],dtype=dty)')
    # land-use
    for var in ['CVEG_luc','CSOIL1_luc','CSOIL2_luc','CHWP1_luc','CHWP2_luc','CHWP3_luc']:
        exec(var+' = np.zeros([nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty)')
    # atmosphere
    for var in ['D_CO2','D_CH4','D_CH4_lag','D_N2O','D_N2O_lag','D_EESC','D_O3s']:
        exec(var+' = np.array([0],dtype=dty)')
    for var in ['D_HFC','D_HFC_lag','D_PFC','D_PFC_lag','D_ODS','D_ODS_lag']:
        exec(var+' = np.zeros([nb_'+var[2:2+3]+'],dtype=dty)')
    # climate
    for var in ['D_gst','D_gst0','D_sst','D_gyp','D_OHC']:
        exec(var+' = np.array([0],dtype=dty)')
    for var in ['D_lst','D_lyp']:
        exec(var+' = np.zeros([nb_regionI],dtype=dty)')        
    #=======
    # B. RUN
    #=======
    #get N input from FAO and fitted    
    newbiome = ['cot','mai','rub','ric','whe']
    #Sensitivity of Nitrogen to npp
    CouID = np.array([line for line in csv.reader(open('data/'+Filein+'/Crop_Area2020.csv','r'))], dtype=dty)[:,1]    
    ## Define
    D_CO2_2018 = np.zeros([1,1],dtype=dty)
    D_lst_2018 = np.zeros([1,10],dtype=dty)
    D_lyp_2018 = np.zeros([1,10],dtype=dty)
    ## N2O and CH4 emissions 
    Cumemission = 0
    EFNPK = 0
    EFNPKT = 0
    REFF = 0
    REFFCN = 0
    CumN2O = 0
    CumCH4 = 0
    REN2O = 0
    RECH4 = 0    
    YieldTot = np.zeros([1])
    Emi_exp = np.zeros([ind_final+1,1],dtype=dty)
    Emi_pet = np.zeros([ind_final+1,1],dtype=dty)
    CovidFF = 0
    Emi_expT = np.zeros([ind_final+1,1],dtype=dty)
    
    ## Crop loss and waste (UN2013)
    lossvl = 0.0
    # Prodduction ration in 2020 from FAO data
    areadic = np.array([line for line in csv.reader(open('data/'+Filein+'/diet.csv','r'))], dtype=dty)[:,2:]   
    # The fraction of crops used for food purposes
    fandic = np.array([line for line in csv.reader(open('data/'+Filein+'/fooduse.csv','r'))], dtype=dty)[:,2:]    
    # The factor for converting the agricultural product produced to the part that is edible
    fcedic = {'Mai':'0.79','Ric':'1','Whe':'0.78'}
    # The caloric content by weight for each crop (kcal kg-1)
    cwcdic = {'Mai':'3622.95','Ric':'3882.05','Whe':'3391.67'}
        
    ## Emission factors from harvested crop products
    harv_0 = np.array([line for line in csv.reader(open('data/'+Filein+'/harv_biome.csv','r'))], dtype=dty)
    ## Population (2020-2100)
    Pop = np.array([line for line in csv.reader(open('data/'+Filein+'/'+IfScen+'_Population.csv','r'))], dtype=dty)[2:,:]
    #Pop = np.array([line for line in csv.reader(open('data/'+Filein+'/UN_Population_H.csv','r'))], dtype=dty)[2:,:]
    ## COVID-19 infection and hospitalizations in 2020
    # Infection
    if mod_Cropdemand == 'COVID-19-Food-Dynamic':        
        ## Reduction of the infection rate relative to the baseline level 
        infection = np.ones([1,171],dtype=dty)*(Infection[infec])     
    elif mod_Cropdemand == 'Population-Driven' and covid == 0:
        infection = np.zeros([1,171],dtype=dty)
    else:
        infection = np.array([line for line in csv.reader(open('data/'+Filein+'/covid_infection_hospitalizations.csv','r'))], dtype=dty)[:,2]
    # Hospitalizations
    ## Cropland Alpha 
    for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
        exec('Alpha'+var+' = np.zeros([nb_regionI],dtype=dty)')
    ## Yeild Ratio
    RatYie = np.zeros([10],dtype=dty)           
    ## npp in FAO,Absolute value
    nppFAO = np.array([line for line in csv.reader(open('data/'+Filein+'/nppAbFAO1961to2020.csv','r'))], dtype=dty)
    nppFAO *= 0.2
    nppcount = 0 
    for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
        exec('nppFAO'+var+' = nppFAO' +'[:,nppcount*10:(nppcount+1)*10]')
        nppcount = nppcount + 1
        
    ## Yield for each countries
    for var in ['Cot','Mai','MaiC','MaiF','Rub','Ric','Whe','Rest']:
        exec('YieldPCou'+var+' = np.zeros([ind_final+1,171],dtype=dty)')
    ## Area harvested for each countries
    for var in ['Cot','Mai','MaiC','MaiF','Rub','Ric','Whe','Rest']:
        exec('AreaPCou'+var+' = np.zeros([ind_final+1,171],dtype=dty)')
    ## Crop production for each regions
    for var in ['Cot','Mai','MaiC','MaiF','Rub','Ric','Whe','Rest']:
        exec('ProdcPCou'+var+' = np.zeros([ind_final+1,171],dtype=dty)') 
    ## Area harvested for each regions
    for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
        exec('AreaReg'+var+' = np.zeros([ind_final+1,nb_regionI],dtype=dty)')      
    ## Crop demand for different scenarios
    for var in ['Cot','Mai','MaiC','MaiF','Rub','Ric','Whe','Rest']:
        exec('CropDemandPCou'+var+' = np.zeros([ind_final+1,171],dtype=dty)')    
    ## LUC from 1700 to 2020 from FAO   
    cropdic = {'Cotton':'Cot','Maize':'Mai','Rubber':'Rub','Rice':'Ric','Wheat':'Whe','Rest':'Rest'}
    for var in ['Cotton','Maize','Rubber','Rice','Wheat','Rest']:
        exec('AreaLUC'+cropdic[var]+' = np.zeros([ind_final+1,9],dtype=dty)')
        exec('AreaLUC'+cropdic[var]+"[0:321,:] = np.array([line for line in csv.reader(open('data/'+Filein+'/LUC_"+var+"_1700_2020.csv','r'))], dtype=dty)")
      
        
    # Area change of non-cereal and cereals
    for t in range(1,ind_final+1):        
        print 'Year:',t+1700        
            
        # Historical land cover change from FAO (1751-2016)
        if t > 50 and t < 317 :
            AreaLUCOri = (np.sum(LUC[t],1) - np.sum(LUC[t],2))
            # calculate the ratio 
            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                exec('Alpha'+var+'[1:10] = AreaLUC'+var+'[t]/AreaLUCOri[1:10,3]')
            # change respective LUC by the ratio
            lucdic = {'Cot':'5','Mai':'6','Rub':'7','Ric':'8','Whe':'9','Rest':'3'}
            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                exec('LUC[t,:,:,'+lucdic[var]+'] = Alpha'+var+'[:,np.newaxis]*LUC[t,:,:,3]')
                exec('LUC[t,:,'+lucdic[var]+',:] = Alpha'+var+'[:,np.newaxis]*LUC[t,:,3,:]')
            # change respective SHIFT by the ratio of SHIFT
            lucdic = {'Cot':'5','Mai':'6','Rub':'7','Ric':'8','Whe':'9','Rest':'3'}
            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                exec('SHIFT[t,:,:,'+lucdic[var]+'] = Alpha'+var+'[:,np.newaxis]*SHIFT[t,:,:,3]')
                exec('SHIFT[t,:,'+lucdic[var]+',:] = Alpha'+var+'[:,np.newaxis]*SHIFT[t,:,3,:]')
                
        if t > 316 and t < 321:
            # Get rid of other disturbation
            LUC[t,1:,1,:] = 0
            LUC[t,1:,:,3] = 0
            LUC[t,1:,3,:] = 0
            lucdic = {'Cot':'5','Mai':'6','Rub':'7','Ric':'8','Whe':'9','Rest':'3'}
            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                exec('LUC[t,1:,1,'+lucdic[var]+'] =  AreaLUC'+var+'[t,:]')
        
        if t == 320:
            # Countries area harvested in 2020
            AreaPCou2020 = np.array([line for line in csv.reader(open('data/'+Filein+'/Crop_Area2020.csv','r'))], dtype=dty)[:,2:]    
            croptype = {'Cot':'0','Mai':'1','Rub':'2','Ric':'3','Whe':'4','Rest':'5'}
            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                exec('AreaPCou'+var+'[t,:] = AreaPCou2020[:,'+croptype[var]+']') 
            for j in range(171):
                for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                    exec('AreaReg'+var+'[t,int(CouID[j])] += AreaPCou'+var+'[t,j]')            
            # Countries yeild in 2020
            YieldPCou2020 = np.array([line for line in csv.reader(open('data/'+Filein+'/Crop_Yield2020.csv','r'))], dtype=dty)[:,2:]    
            croptype = {'Cot':'0','Mai':'1','Rub':'2','Ric':'3','Whe':'4','Rest':'5'}
            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                exec('YieldPCou'+var+'[t,:] = YieldPCou2020[:,'+croptype[var]+']') 
                                        
        for tt in range(p):
            #print 'Loop:',tt+1
            #---------
            # 1. OCEAN
            #---------
            # structure
            D_mld = mld_0 * alpha_mld * (np.exp(gamma_mld*fT*D_sst)-1)
            # fluxes
            D_FIN = p_circ * v_fg * alpha_CO2 * D_CO2
            D_FOUT = p_circ * v_fg * alpha_CO2 * f_pCO2(D_dic,fT*D_sst)
            D_FCIRC = D_CSURF * (1/tau_circ)
            OSNK = np.sum(D_FOUT - D_FIN)
            # stocks
            D_CSURF += (p**-1) * (D_FIN - D_FOUT - D_FCIRC)
            #D_dic = alpha_dic * np.sum(D_CSURF) / (1+D_mld/mld_0) # OSACR 2.2
            D_dic = max(alpha_dic * np.sum(D_CSURF) / (1+D_mld/mld_0),-1*dic_0) # XSQ            
            #--------
            # 2. LAND
            #--------                                                    
            # CO2 et al influence for crops (Maize and other)
            for var in ['D_lyp_cou','D_lyp_Y','D_lyp_couB','D_CO2_B','FT','FT_Ma','FP']:
                exec(var+'= np.zeros([171])')                                    
            # Temperature influence for crops (Maize and other)
            for var in ['Reg','t','s','Reg_Ma','t_Ma','s_Ma']:
                exec('FT'+var+'= np.zeros([10])')
                
            # Yeild in 2021 is equal to 2020 values
            if t == Ys-1700:
                for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                    exec('YieldPCou'+var+'[t,:] = YieldPCou'+var+'[t-1,:]')
                    
            if t == 320:
                # Correct T0 to make T2016-2019 is equal to observed
                # Temperature at 1700
                T0 = np.array([0,18.87,22.15,18.06,22.62,23.28,16.92,20.72,25.00,21.17])    
                Tmod = T0 + np.mean(np.array([D_lst_t[316,:],D_lst_t[317,:],D_lst_t[318,:],D_lst_t[319,:]]),axis=0)
                Tobs=np.array([0,21.56573923,24.2579048,20.19866791,24.99114654,25.52582281,19.85079192,22.94499806,26.94859918,23.12200498])
                Tcorr = Tobs-Tmod
                T0 = T0+Tcorr  
                
            # Get temperature change and yield change relationship from 2022
            if t > Ys-1700:
                D_Ts = D_lst_t[Ys-1700,:]
                D_CO2_Y = D_CO2_t[Ys-1700] # CO2 at the year 2017
                D_lyp_YReg = D_lyp_t[Ys-1700,:] # precipitation at the year 2017                 
                # FCO2, FCO2_Y, FP( FP_Y )
                FCY = (TCO2_A[Tc]*(D_CO2_Y+278)**2 + TCO2_B[Tc]*(D_CO2_Y+278)+ TCO2_C[Tc])/(TCO2_A[Tc]*278**2 + TCO2_B[Tc]*278+ TCO2_C[Tc])
                if D_CO2+278<700:
                    FC = (TCO2_A[Tc]*(D_CO2+278)**2 + TCO2_B[Tc]*(D_CO2+278)+ TCO2_C[Tc])/(TCO2_A[Tc]*278**2 + TCO2_B[Tc]*278+ TCO2_C[Tc])
                elif D_CO2+278>700 and D_CO2+278==700:
                    FC = (TCO2_A[Tc]*700**2 + TCO2_B[Tc]*700+ TCO2_C[Tc])/(TCO2_A[Tc]*278**2 + TCO2_B[Tc]*278+ TCO2_C[Tc])
                if D_CO2 > D_CO2_Y:
                    FC_Ma = FC
                elif D_CO2 <= D_CO2_Y:
                    FC_Ma = FCY                
                #FT
                if TYform == 'Qua':
                    # Original function
                    minv = 0.01
                    tmin = minv*(Para_C[ty] - (Para_B[ty]**2)/(4*Para_A[ty])) # 1% of Ymax,Ymax = c-b^2/4*a
                    #tmin_Ma = 0.01*(Mai_A[0]*(30-8.5)**2 + Mai_B[0]*(30-8.5)+ Mai_C[0])
                    tmin_Ma = minv*(Mai_C[ty] - (Mai_B[ty]**2)/(4*Mai_A[ty]))
                    
                    Xma = -Mai_B[ty]/2/Mai_A[ty]
                    
                    for n in range(10):
                        FTt[n] = max(tmin,Para_A[ty]*(T0[n]+D_lst[n])**2 + Para_B[ty]*(T0[n]+D_lst[n]) + Para_C[ty])
                        FTs[n] = max(tmin,Para_A[ty]*(T0[n]+D_Ts[n])**2 + Para_B[ty]*(T0[n]+D_Ts[n]) + Para_C[ty])
                        FTReg[n] = FTt[n]/FTs[n]
                        if FTs[n] == 0:
                            FTReg[n] = 0
                        # For maize
                        FTt_Ma[n] = max(tmin_Ma,Mai_A[0]*(T0[n]+D_lst[n])**2 + Mai_B[0]*(T0[n]+D_lst[n]) + Mai_C[0])
                        FTs_Ma[n] = max(tmin_Ma,Mai_A[0]*(T0[n]+D_Ts[n])**2 + Mai_B[0]*(T0[n]+D_Ts[n]) + Mai_C[0])
                        if T0[n]+D_lst[n] <Xma:
                            FTt_Ma[n] = Mai_A[0]*(Xma)**2 + Mai_B[0]*(Xma) + Mai_C[0]
                        if T0[n]+D_Ts[n] <Xma:
                            FTs_Ma[n] = Mai_A[0]*(Xma)**2 + Mai_B[0]*(Xma) + Mai_C[0]                       
                        FTReg_Ma[n] = FTt_Ma[n]/FTs_Ma[n]
                        if FTs_Ma[n] == 0:
                            FTReg_Ma[n] = 0                
                # Calculate yield for each country
                CouID = np.array([line for line in csv.reader(open('data/'+Filein+'/Crop_Area2020.csv','r'))], dtype=dty)[:,1]
                for i in range(171):
                    CI = int(CouID[i])
                    D_lyp_Y[i] = D_lyp_YReg[CI]
                    D_lyp_cou[i] = D_lyp[CI]
                    FT[i] = FTReg[CI]; FT_Ma[i] = FTReg_Ma[CI]
                    # Calculate FP
                    FP[i] = np.exp(gamma_yieldP[i]*D_lyp_cou[i])/np.exp(gamma_yieldP[i]*D_lyp_Y[i])
                    # Crops yield for each country
                    funcdic = {'Cot':'cot','Mai':'mai','Rub':'rub','Ric':'ric','Whe':'whe','Rest':'oth'}
                    if mod_Cropdemand == 'Population-Driven' or mod_Cropdemand == 'Constant':
                        for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                            if var in ['Mai','Cot']:
                                exec('YieldPCou'+var+'[t,i] = max(0,YieldPCou'+var+'[320,i] * f_yield'+funcdic[var]+'YStart(FC_Ma,FCY,FT_Ma,FP)[i])')
                            else:
                                exec('YieldPCou'+var+'[t,i] = max(0,YieldPCou'+var+'[320,i] * f_yield'+funcdic[var]+'YStart(FC,FCY,FT,FP)[i])')  
                    else:         
                        for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                            if var in ['Mai','Cot']:
                                exec('YieldPCou'+var+'[t,i] = max(0,YieldPCou'+var+'[320,i] * f_yield'+funcdic[var]+'YStart(FC_Ma,FCY,FT_Ma,FP)[i])')
                            elif var in ['Rub']:
                                exec('YieldPCou'+var+'[t,i] = max(YieldPCou'+var+'[320,i],YieldPCou'+var+'[320,i] * f_yield'+funcdic[var]+'YStart(FC_Ma,FCY,FT_Ma,FP)[i])')
                            else:
                                exec('YieldPCou'+var+'[t,i] = max(0,YieldPCou'+var+'[320,i] * f_yield'+funcdic[var]+'YStart(FC,FCY,FT,FP)[i])')    
            # Simulation of the future crops demand and landcover changes from 2021
            if t > Fy-1700:

                # Get rid cropdemand data for each loop                                                         
                for var in ['Cot','Mai','MaiC','MaiF','Rub','Ric','Whe','Rest']:                        
                    exec('CropDemandPCou'+var+'[t,:] = 0')
                    
                # Cropland demand scenarios : Crop demand with COVID-19 influences 
                if mod_Cropdemand == 'COVID-19-Food-Dynamic-Constant':
                    # Per capita 1700 kacal d-1 (Minimum calorie requirement OSCAR:mod_Cropdemand == 'Constant')
                    Cademand = 2000
                    # Cropland expansion from 1700 + expt
                    expt = 321
                    # COVID-19 infection rate
                    Infcov = '0.01'
                    # Medical demand for crops material in 2020: Cotton (Mask),Mazie (Ethanol),Rubber (Gloves).
                    cparam = {'Cot':'SAR_Cot_C','MaiC':'SAR_Mai_C','Rub':'SAR_Rub_C'}
                    for var in ['Cot','MaiC','Rub']:                       
                        if var=='Cot':
                            exec('COVID'+var+' = np.ones([1,171],dtype=dty)*f_'+var+'demand('+Infcov+','+cparam[var]+',SAR_Non_C)')                            
                        elif var=='MaiC':
                            exec('COVID'+var+' = np.ones([1,171],dtype=dty)*f_Maidemand('+Infcov+','+cparam[var]+')')
                        else:
                            exec('COVID'+var+' = np.ones([1,171],dtype=dty)*f_'+var+'demand('+Infcov+','+cparam[var]+')')   
                    # Cotton, Mazie and Rubber demand for each countries by COVID-19 influences
                    # Crop loss
                    lossdic = {'Cot':'1.7420','MaiC':'1.6482','Rub':'1.6348'}
                    # Crops demand for COVID-19
                    for var in ['Cot','MaiC','Rub']:
                        exec('CropDemandPCou'+var+'[t,:] = COVID'+var+'*Pop[t-320,:]*10**(-12)*'+lossdic[var])
                        exec('CropDemandPCou'+var+'[320,:] = COVID'+var+'*Pop[0,:]*10**(-12)*'+lossdic[var])
                    # Crops for food
                    cindex = {'Ric':'0','Mai':'1','Whe':'2'}
                    if t <= expt:
                        for var in ['Ric','Mai','Whe']:                               
                            if var == 'Mai':
                                exec('CropDemandPCou'+var+'F[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                                exec('CropDemandPCou'+var+'Fs = (max(0,(Cademand-1700))*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                                exec('AreaPCou'+var+'F[t,:] = AreaPCou'+var+'[Fy-1700,:]*1.01')
                            else:
                                exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                                exec('CropDemandPCou'+var+'s = (max(0,(Cademand-1700))*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[Fy-1700,:]*1.01')

                    elif t < expt + 31:
                        for var in ['Ric','Mai','Whe']:                               
                            if var == 'Mai':
                                exec('CropDemandPCou'+var+'F[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')
                                exec('AreaPCou'+var+'F[t,:] = AreaPCou'+var+'F[t-1,:] + (CropDemandPCou'+var+'F[t,:] - CropDemandPCou'+var+'F[t-1,:]+CropDemandPCou'+var+'Fs/30.0)/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'F[t,:] = np.max(AreaPCou'+var+'F, axis=0)') 
                            else:
                                exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')                                               
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:]+CropDemandPCou'+var+'s/30.0)/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')           
                                                                                
                    else:
                        # Cereals calories demand from 2021
                        for var in ['Mai','Ric','Whe']:
                            # Crops for food use 
                            if var == 'Mai':
                                exec('CropDemandPCou'+var+'F[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')
                                exec('AreaPCou'+var+'F[t,:] = AreaPCou'+var+'F[t-1,:] + (CropDemandPCou'+var+'F[t,:] - CropDemandPCou'+var+'F[t-1,:])/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'F[t,:] = np.max(AreaPCou'+var+'F, axis=0)') 
                            else:
                                exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')                                               
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:])/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')                     
                    # Area harvested: meet basic requirements in 2030
                    if t < Fy-1700 + 11:
                        for var in ['Cot','Mai','Rub']:
                            if var == 'Mai':
                                exec('AreaPCou'+var+'C[t,:] = AreaPCou'+var+'C[t-1,:] + (CropDemandPCou'+var+'C[t,:] - CropDemandPCou'+var+'C[t-1,:] + CropDemandPCou'+var+'C[320,:]/10.0)/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'C[t,:] = np.max(AreaPCou'+var+'C, axis=0)') 
                            else:
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:] + CropDemandPCou'+var+'[320,:]/10.0)/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')                                   
                    # COVID-19 stops in 2050
                    elif t < Fy-1700 + 31:
                        for var in ['Cot','Mai','Rub']:
                            if var == 'Mai':
                                exec('AreaPCou'+var+'C[t,:] = AreaPCou'+var+'C[t-1,:] + (CropDemandPCou'+var+'C[t,:] - CropDemandPCou'+var+'C[t-1,:])/YieldPCou'+var+'[t-1,:]')
                                exec('AreaPCou'+var+'C[t,:] = np.max(AreaPCou'+var+'C, axis=0)')                         
                            else:
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:])/YieldPCou'+var+'[t-1,:]')
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')
                    else:                        
                        for var in ['Cot','Mai','Rub']:
                            if var == 'Mai':
                                exec('AreaPCou'+var+'C[t,:] = np.max(AreaPCou'+var+'C, axis=0)')                         
                            else:
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')                                                                            
                    # Maize                                
                    AreaPCouMai[t,:] = AreaPCouMaiC[t,:] + AreaPCouMaiF[t,:]
                    AreaPCouMai[t,:] = np.max(AreaPCouMai, axis=0)
                    # Other Crops        
                    for var in ['Rest']:
                        exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[Fy-1700,:]')  
                
                ## Cropland demand scenarios : Crop demand with COVID-19 influences 
                if mod_Cropdemand == 'COVID-19-Food-Dynamic':
                    # Per capita 1700 kacal d-1 (Minimum calorie requirement OSCAR:mod_Cropdemand == 'Constant') 
                    Cademand = Calories[cal]
                    # Cropland expansion from 1700+expt
                    expt = 321
                    # Medical demand for crops material in 2020: Cotton (Mask),Mazie (Ethanol),Rubber (Gloves).
                    cparam = {'Cot':'SAR_Cot_C','MaiC':'SAR_Mai_C','Rub':'SAR_Rub_C'}
                    for var in ['Cot','MaiC','Rub']:  
                        if (var=='Cot'):
                            exec('COVID'+var+' = np.ones([1,171],dtype=dty)*f_'+var+'demand(Infection[infec],'+cparam[var]+',SAR_Non_C)')
                        elif var=='MaiC':
                            exec('COVID'+var+' = np.ones([1,171],dtype=dty)*f_Maidemand(Infection[infec],'+cparam[var]+')')
                        else:
                            exec('COVID'+var+' = np.ones([1,171],dtype=dty)*f_'+var+'demand(Infection[infec],'+cparam[var]+')')
                    # Cotton, Mazie and Rubber demand for each countries by COVID-19 influences
                    # Crop loss
                    lossdic = {'Cot':'1.7420','MaiC':'1.6482','Rub':'1.6348'}
                    # Crops demand for COVID-19
                    for var in ['Cot','MaiC','Rub']:
                        exec('CropDemandPCou'+var+'[t,:] = COVID'+var+'*Pop[t-320,:]*10**(-12)*'+lossdic[var])
                        exec('CropDemandPCou'+var+'[320,:] = COVID'+var+'*Pop[0,:]*10**(-12)*'+lossdic[var])
                    # Crops for food                        
                    cindex = {'Ric':'0','Mai':'1','Whe':'2'}
                    if t <= expt:
                        for var in ['Ric','Mai','Whe']:                               
                            if var == 'Mai':
                                exec('CropDemandPCou'+var+'F[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                                exec('CropDemandPCou'+var+'Fs = (max(0,(Cademand-1700))*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                                exec('AreaPCou'+var+'F[t,:] = AreaPCou'+var+'[Fy-1700,:]*1.01')
                            else:
                                exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                                exec('CropDemandPCou'+var+'s = (max(0,(Cademand-1700))*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[Fy-1700,:]*1.01')

                    elif t < expt + 31:
                        for var in ['Ric','Mai','Whe']:                               
                            if var == 'Mai':
                                exec('CropDemandPCou'+var+'F[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')
                                exec('AreaPCou'+var+'F[t,:] = AreaPCou'+var+'F[t-1,:] + (CropDemandPCou'+var+'F[t,:] - CropDemandPCou'+var+'F[t-1,:]+CropDemandPCou'+var+'Fs/30.0)/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'F[t,:] = np.max(AreaPCou'+var+'F, axis=0)') 
                            else:
                                exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')                                               
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:]+CropDemandPCou'+var+'s/30.0)/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')           
                                                                                
                    else:
                        # Cereals calories demand from 2021
                        for var in ['Mai','Ric','Whe']:
                            # Crops for food use 
                            if var == 'Mai':
                                exec('CropDemandPCou'+var+'F[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')
                                exec('AreaPCou'+var+'F[t,:] = AreaPCou'+var+'F[t-1,:] + (CropDemandPCou'+var+'F[t,:] - CropDemandPCou'+var+'F[t-1,:])/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'F[t,:] = np.max(AreaPCou'+var+'F, axis=0)') 
                            else:
                                exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')                                               
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:])/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')                     
                                                                                                                                                                      
                    # Area harvested: meet basic requirements in 2030
                    if t < Fy-1700 + 11:
                        for var in ['Cot','Mai','Rub']:
                            if var == 'Mai':
                                exec('AreaPCou'+var+'C[t,:] = AreaPCou'+var+'C[t-1,:] + (CropDemandPCou'+var+'C[t,:] - CropDemandPCou'+var+'C[t-1,:] + CropDemandPCou'+var+'C[320,:]/10.0)/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'C[t,:] = np.max(AreaPCou'+var+'C, axis=0)') 
                            else:
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:] + CropDemandPCou'+var+'[320,:]/10.0)/YieldPCou'+var+'[t-1,:]') 
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')   
                    # COVID-19 stops in 2050
                    elif t < Fy-1700 + 31:
                        for var in ['Cot','Mai','Rub']:
                            if var == 'Mai':
                                exec('AreaPCou'+var+'C[t,:] = AreaPCou'+var+'C[t-1,:] + (CropDemandPCou'+var+'C[t,:] - CropDemandPCou'+var+'C[t-1,:])/YieldPCou'+var+'[t-1,:]')
                                exec('AreaPCou'+var+'C[t,:] = np.max(AreaPCou'+var+'C, axis=0)')                         
                            else:
                                exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:])/YieldPCou'+var+'[t-1,:]')
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')                                
                    else:                        
                        for var in ['Cot','Mai','Rub']:
                            if var == 'Mai':
                                exec('AreaPCou'+var+'C[t,:] = np.max(AreaPCou'+var+'C, axis=0)')                         
                            else:
                                exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')
                    # Maize                                
                    AreaPCouMai[t,:] = AreaPCouMaiC[t,:] + AreaPCouMaiF[t,:]
                    AreaPCouMai[t,:] = np.max(AreaPCouMai, axis=0)
                    # Other Crops        
                    for var in ['Rest']:
                        exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[Fy-1700,:]')  
                                                    
                ## Cropland demand scenarios : Population increase
                if mod_Cropdemand == 'Population-Driven':                    
                    # Per capita 1700 kacal d-1 (Minimum calorie requirement OSCAR:mod_Cropdemand == 'Constant')  
                    Cademand = 2000
                    # Cropland expansion from 1700+expt
                    expt = 321
                    cindex = {'Ric':'0','Mai':'1','Whe':'2'}
                    if t <= expt:
                        for var in ['Ric','Mai','Whe']:                               
                            exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                            exec('CropDemandPCou'+var+'s = (max(0,(Cademand-1700))*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365')
                            exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[Fy-1700,:]*1.01')
                    elif t < expt + 31:
                        for var in ['Ric','Mai','Whe']:                               
                            exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')                                               
                            exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:]+CropDemandPCou'+var+'s/30.0)/YieldPCou'+var+'[t-1,:]') 
                            exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')                                                                                
                    else:
                        # Cereals calories demand from 2021
                        for var in ['Mai','Ric','Whe']:
                            # Crops for food use 
                            exec('CropDemandPCou'+var+'[t,:] = (Cademand*areadic[:,'+cindex[var]+']*Pop[t-320,:]/'+cwcdic[var]+'/'+fcedic[var]+')*10**(-12)*365*(1+lossvl)')                                               
                            exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[t-1,:] + (CropDemandPCou'+var+'[t,:] - CropDemandPCou'+var+'[t-1,:])/YieldPCou'+var+'[t-1,:]') 
                            exec('AreaPCou'+var+'[t,:] = np.max(AreaPCou'+var+', axis=0)')  
                    # Other Crops        
                    for var in ['Cot','Rub','Rest']:
                        # Per capita 1400 kacal d-1 (Minimum calorie requirement)
                        exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[Fy-1700,:]')   
    
                ## Cropland demand scenarios : No COVID-19 influences    
                if mod_Cropdemand == 'Constant':
                    for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                        exec('AreaPCou'+var+'[t,:] = AreaPCou'+var+'[Fy-1700,:]')
                
                # Get rid areaReg data for each loop                                                         
                for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:                        
                    exec('AreaReg'+var+'[t,:] = 0')                
                # Area harvested for each regions                
                CouID = np.array([line for line in csv.reader(open('data/'+Filein+'/Crop_Area2020.csv','r'))], dtype=dty)[:,1] 
                for j in range(171):
                    for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:                        
                        exec('AreaReg'+var+'[t,int(CouID[j])] += AreaPCou'+var+'[t,j]')
                # LUC
                for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                    for reg in range(9):                       
                        exec('AreaLUC'+var+'[t,reg] = 36.0/21.0*((AreaReg'+var+'[t,reg+1]-AreaReg'+var+'[t-1,reg+1])-15.0/36.0*AreaLUC'+var+'[t-1,reg])')
                
                # Get rid of other disturbation 
                LUC[t,1:,1,:] = 0
                LUC[t,1:,:,3] = 0
                LUC[t,1:,3,:] = 0
                
                # First marginal land, then forest
                if ExpSeq == 'Mar1For2':
                    # Marginal land area
                    exp_area = 0
                    for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                        exec('exp_area += (np.sum(AreaReg'+var+'[t,:])-np.sum(AreaReg'+var+'[320,:]))')                  
                    lucdic = {'Cot':'5','Mai':'6','Rub':'7','Ric':'8','Whe':'9','Rest':'3'}
                    # First marginal land, then forest
                    if exp_area <= MarArea:
                        # First marginal land                                        
                        for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                            exec('LUC[t,1:,0,'+lucdic[var]+'] = AreaLUC'+var+'[t,:]')
                    else:
                        if t == expt:
                            m = MarArea/exp_area; n = 1-m
                            #print m,n,exp_area,expt
                            # Marginal land (m)                                       
                            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                                exec('LUC[t,1:,0,'+lucdic[var]+'] = m*AreaLUC'+var+'[t,:]')
                            # Forest (n)
                            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                                exec('LUC[t,1:,1,'+lucdic[var]+'] = n*AreaLUC'+var+'[t,:]')
                        # Then forest       
                        else:
                            for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                                exec('LUC[t,1:,1,'+lucdic[var]+'] = AreaLUC'+var+'[t,:]')  
                # Forest           
                if ExpSeq == 'For1Mar2':
                    lucdic = {'Cot':'5','Mai':'6','Rub':'7','Ric':'8','Whe':'9','Rest':'3'}
                    for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                        exec('LUC[t,1:,1,'+lucdic[var]+'] = AreaLUC'+var+'[t,:]')                      
                                                                                                                                                
            D_AREA += (p**-1) * (np.sum(LUC[t],1) - np.sum(LUC[t],2)) 
            
            D_AWET = AWET_0*(gamma_wetT*fT*D_lst + gamma_wetP*fT*D_lyp + gamma_wetC*fT*D_CO2)
            # factors
            D_k_igni = gamma_igniT*fT*D_lst[:,np.newaxis]+gamma_igniP*fT*D_lyp[:,np.newaxis]+gamma_igniC*fT*D_CO2
            D_k_rho = f_rho(fT*D_lst[:,np.newaxis],fT*D_lyp[:,np.newaxis])            
            # fluxes
            D_npp = npp_0 * f_npp(D_CO2,fT*D_lst[:,np.newaxis],fT*D_lyp[:,np.newaxis]) 
            
            # Historical npp from FAO (1961-2021) 
            if t > 260 and t < Ys-1700: 
                cropnpp = {'Cot':'5','Mai':'6','Rub':'7','Ric':'8','Whe':'9','Rest':'3'}
                for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                    exec('D_npp[:,'+lucdic[var]+'] = nppFAO'+var+'[t-261,:] - npp_0[:,'+lucdic[var]+']')
            if t == 321:
                cropnpp = {'Cot':'5','Mai':'6','Rub':'7','Ric':'8','Whe':'9','Rest':'3'}
                for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                    exec('D_npp[:,'+lucdic[var]+'] = nppFAO'+var+'[320-261,:] - npp_0[:,'+lucdic[var]+']')                
            
            # npp from 2021        
            if t > Ys-1700:
                D_TY = D_lst- D_lst_t[Ys-1700,:]  #the difference between temperature at t and at 2017
                D_CO2_Y = D_CO2_t[Ys-1700] # CO2 at the year 2017
                D_lyp_Y = D_lyp_t[Ys-1700,:] # precipitation at the year 2017                                
                FP = np.exp(gamma_nppP*D_lyp[:,np.newaxis])/np.exp(gamma_nppP*D_lyp_Y[:,np.newaxis])
                # nnp from yeild
                for i in range(10):
                    lucdic = {'Cot':'5','Mai':'6','Rub':'7','Ric':'8','Whe':'9','Rest':'3'}
                    if mod_Cropdemand == 'Population-Driven' or mod_Cropdemand == 'Constant':
                        for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                            if var in ['Mai','Cot']:
                                exec('D_npp[i,'+lucdic[var]+'] = max(-1*npp_0[i,'+lucdic[var]+'],(nppFAO'+var+'[Fy-1961,i]* f_nppcrYStart(FC_Ma,FCY,FTReg_Ma[:,np.newaxis],FP)[i,'+lucdic[var]+']-npp_0[i,'+lucdic[var]+']))')
                            else:
                                exec('D_npp[i,'+lucdic[var]+'] = max(-1*npp_0[i,'+lucdic[var]+'],(nppFAO'+var+'[Fy-1961,i]* f_nppcrYStart(FC,FCY,FTReg[:,np.newaxis],FP)[i,'+lucdic[var]+']-npp_0[i,'+lucdic[var]+']))')
                    else:
                        for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                            if var in ['Mai','Cot']:
                                exec('D_npp[i,'+lucdic[var]+'] = max(-1*npp_0[i,'+lucdic[var]+'],(nppFAO'+var+'[Fy-1961,i]* f_nppcrYStart(FC_Ma,FCY,FTReg_Ma[:,np.newaxis],FP)[i,'+lucdic[var]+']-npp_0[i,'+lucdic[var]+']))')
                            elif var in ['Rub']:
                                exec('D_npp[i,'+lucdic[var]+'] = max(-1*npp_0[i,'+lucdic[var]+'],nppFAO'+var+'[Fy-1961,i]-npp_0[i,'+lucdic[var]+'])')
                            else:
                                exec('D_npp[i,'+lucdic[var]+'] = max(-1*npp_0[i,'+lucdic[var]+'],(nppFAO'+var+'[Fy-1961,i]* f_nppcrYStart(FC,FCY,FTReg[:,np.newaxis],FP)[i,'+lucdic[var]+']-npp_0[i,'+lucdic[var]+']))')
                           
            # fluxes            
            D_efire = igni_0 * ((1+D_k_igni)*(cveg_0 + D_cveg) - cveg_0)
            D_fmort = mu_0 * D_cveg
            D_rh1 = rho1_0 * ((1+D_k_rho)*(csoil1_0 + D_csoil1) - csoil1_0)
            D_fmet = k_met * D_rh1
            D_rh2 = rho2_0 * ((1+D_k_rho)*(csoil2_0 + D_csoil2) - csoil2_0)
            D_ewet = ewet_0 * np.nan_to_num(np.sum(p_wet * D_csoil1,1) / np.sum(p_wet * csoil1_0,1))     
            D_eharv = harv_0 * D_cveg
            LSNK = np.sum((D_rh1 + D_rh2 + D_efire + D_eharv - D_npp)*(AREA_0 + D_AREA))
            
            D_EWET = ewet_0*D_AWET + D_ewet*AWET_0 + D_ewet*D_AWET             
            # stocks
            D_cveg += (p**-1) * (D_npp - D_fmort - D_efire - D_eharv)
            D_csoil1 += (p**-1) * (D_fmort - D_fmet - D_rh1)
            D_csoil2 += (p**-1) * (D_fmet - D_rh2)            
            Diff = abs(D_rh1) + abs(D_rh2) - abs(D_npp)
            
            # calculate production in model
            NPP_crop = (D_npp[:,3]+npp_0[:,3])*(D_AREA[:,3]+AREA_0[:,3])   
            YieldPCouMaiC[t,:] = YieldPCouMai[t,:]  
            YieldPCouMaiF[t,:] = YieldPCouMai[t,:]
            for var in ['Cot','Mai','MaiC','MaiF','Rub','Ric','Whe','Rest']:
                exec('ProdcPCou'+var+'[t,:] = YieldPCou'+var+'[t,:]*AreaPCou'+var+'[t,:]')
            
            # Emissions from fertilizers in the agricultural sector (N, P2O5 and K2O)
            if t > Fy-1700:
                if mod_fertilizers == 'Tian': 
                    AreaTotal =np.sum(AreaPCouCot[t,:]+AreaPCouMai[t,:]+AreaPCouRub[t,:]+AreaPCouRic[t,:]+AreaPCouWhe[t,:]+AreaPCouRest[t,:])
                    AreaTotal2020 =np.sum(AreaPCouCot[320,:]+AreaPCouMai[320,:]+AreaPCouRub[320,:]+AreaPCouRic[320,:]+AreaPCouWhe[320,:]+AreaPCouRest[320,:])
                    Emi_exp[t,0] = ENPK*127*10**(-3)*(AreaTotal/AreaTotal2020-1) # GtC 
                if mod_fertilizers == 'Xing':
                    Prodtotal = 0
                    ProdtotalT = 0
                    for var in ['Cot','Mai','Rub','Ric','Whe','Rest']:
                        exec('Prodtotal += (AreaPCou'+var+'[t,:]-AreaPCou'+var+'[320,:])*YieldPCou'+var+'[320,:]')
                        exec('ProdtotalT += (AreaPCou'+var+'[t,:])*YieldPCou'+var+'[320,:]')  
                    Emi_exp[t,0] = ENPK*np.sum(Prodtotal)*12/44.0 # GtC
                    Emi_expT[t,0] = ENPK*np.sum(ProdtotalT)*12/44.0 # GtC

            #  Emissions from producing face masks, hand sanitizer and gloves
            if  t > Fy-1700 and t < Fy-1700 + 31:
                if mod_Cropdemand == 'COVID-19-Food-Dynamic-Constant' or mod_Cropdemand == 'COVID-19-Food-Dynamic':
                    cparam = {'Cot':'SAR_Cot_C','Non':'SAR_Cot_C','Mai':'SAR_Mai_C','Rub':'SAR_Rub_C'}
                    for var in ['Cot','Mai','Rub','Non']:  
                        if (var=='Non'):
                            exec('COVID'+var+' = np.ones([1,171],dtype=dty)*p_'+var+'demand('+Infcov+','+cparam[var]+',SAR_Non_C)')
                        else:
                            exec('COVID'+var+' = np.ones([1,171],dtype=dty)*p_'+var+'demand('+Infcov+','+cparam[var]+')')
                    Emi_pet[t,0] = (np.sum(COVIDCot*Pop[t-320,:])*0.111*0.07*10**(-12)+np.sum(COVIDNon*Pop[t-320,:])*15*0.07*10**(-12)\
                                    +np.sum(COVIDMai*Pop[t-320,:])*0.891*0.09*10**(-12)+np.sum(COVIDRub*Pop[t-320,:])*0.179*0.01*10**(-12))*12/44.0
                                
            ## Total yield
            if t > Fy-1700:
                YieldTot = np.sum(ProdcPCouCot[t,:]+ProdcPCouMai[t,:]+ProdcPCouRub[t,:]+ProdcPCouRic[t,:]+ProdcPCouWhe[t,:]+ProdcPCouRest[t,:])\
                /np.sum(AreaPCouCot[t,:]+AreaPCouMai[t,:]+AreaPCouRub[t,:]+AreaPCouRic[t,:]+AreaPCouWhe[t,:]+AreaPCouRest[t,:])
            ##Calculate production per capita per day at country level, note that population is constant at 2020***
            ProdDC_cou = np.zeros([171])
            if t > Fy-1700:
                ProdDC_cou = (ProdcPCouMai[t,:]+ProdcPCouRic[t,:]+ProdcPCouWhe[t,:])/Pop[0]/365
                
            # Save crops informations
            typedic ={'Yield':'YieldPCou','Area':'AreaPCou','Prod':'ProdcPCou'}
            for crp in ['Cot','Mai','MaiC','MaiF','Rub','Ric','Whe','Rest']:
                for var in ['Yield','Area','Prod']:
                    exec(var+crp+ ' = '+ typedic[var]+crp+'[t,:]') 
            
            # 3. LAND-USE
            #------------            
            # initialization
            # land-use change
            for b1 in range(nb_biome):
                for b2 in range(nb_biome):
                    CVEG_luc[:,b1,b2,t] +=  (p**-1) * -(cveg_0+D_cveg)[:,b2] * LUC[t,:,b1,b2] 
                    CSOIL1_luc[:,b1,b2,t] += (p**-1) * ((csoil1_0+D_csoil1)[:,b1] - (csoil1_0+D_csoil1)[:,b2]) * LUC[t,:,b1,b2]
                    CSOIL2_luc[:,b1,b2,t] += (p**-1) * ((csoil2_0+D_csoil2)[:,b1] - (csoil2_0+D_csoil2)[:,b2]) * LUC[t,:,b1,b2]
                    CSOIL1_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (p_AGB[:,b1]*p_HWP0[:,b1]+(1-p_AGB[:,b1])) * LUC[t,:,b1,b2]
                    CHWP1_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * p_AGB[:,b1]*p_HWP1[:,b1] * LUC[t,:,b1,b2]
                    CHWP2_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * p_AGB[:,b1]*p_HWP2[:,b1] * LUC[t,:,b1,b2]
                    CHWP3_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * p_AGB[:,b1]*p_HWP3[:,b1] * LUC[t,:,b1,b2]
            # harvest
            for b in range(nb_biome):
                CVEG_luc[:,b,b,t] += (p**-1) * -HARV[t,:,b]
                CSOIL1_luc[:,b,b,t] += (p**-1) * p_HWP0[:,b] * HARV[t,:,b]
                CHWP1_luc[:,b,b,t] += (p**-1) * p_HWP1[:,b] * HARV[t,:,b]
                CHWP2_luc[:,b,b,t] += (p**-1) * p_HWP2[:,b] * HARV[t,:,b]
                CHWP3_luc[:,b,b,t] += (p**-1) * p_HWP3[:,b] * HARV[t,:,b]
            # shifting cultivation
            for b1 in range(nb_biome):
                for b2 in range(b1,nb_biome):
                    CVEG_luc[:,b1,b2,t] += (p**-1) * -(cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * SHIFT[t,:,b1,b2]
                    CSOIL1_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * (p_AGB[:,b1]*p_HWP0[:,b1]+(1-p_AGB[:,b1])) * SHIFT[t,:,b1,b2]
                    CHWP1_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * p_AGB[:,b1]*p_HWP1[:,b1] * SHIFT[t,:,b1,b2]
                    CHWP2_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * p_AGB[:,b1]*p_HWP2[:,b1] * SHIFT[t,:,b1,b2]
                    CHWP3_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * p_AGB[:,b1]*p_HWP3[:,b1] * SHIFT[t,:,b1,b2]
                    CVEG_luc[:,b2,b1,t] += (p**-1) * -(cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * SHIFT[t,:,b1,b2]
                    CSOIL1_luc[:,b2,b1,t] += (p**-1) * (cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * (p_AGB[:,b2]*p_HWP0[:,b2]+(1-p_AGB[:,b2])) * SHIFT[t,:,b1,b2]
                    CHWP1_luc[:,b2,b1,t] += (p**-1) * (cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * p_AGB[:,b2]*p_HWP1[:,b2] * SHIFT[t,:,b1,b2]
                    CHWP2_luc[:,b2,b1,t] += (p**-1) * (cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * p_AGB[:,b2]*p_HWP2[:,b2] * SHIFT[t,:,b1,b2]
                    CHWP3_luc[:,b2,b1,t] += (p**-1) * (cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * p_AGB[:,b2]*p_HWP3[:,b2] * SHIFT[t,:,b1,b2]

            # fluxes
            # book-keeping model
            NPP_luc = 0*CVEG_luc
            EFIRE_luc = (igni_0*(1+D_k_igni))[:,np.newaxis,:,np.newaxis] * CVEG_luc
            FMORT_luc = mu_0[:,np.newaxis,:,np.newaxis] * CVEG_luc
            
            RH1_luc = (rho1_0*(1+D_k_rho))[:,np.newaxis,:,np.newaxis] * CSOIL1_luc
            FMET_luc = k_met * RH1_luc
            RH2_luc = (rho2_0*(1+D_k_rho))[:,np.newaxis,:,np.newaxis] * CSOIL2_luc
            
            EHWP1_luc = np.zeros([nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty)
            EHWP1_luc[:,:,:,:t+1] = (r_HWP1*(1-r_HWP1**tt))[np.newaxis,np.newaxis,np.newaxis,t::-1] * CHWP1_luc[:,:,:,:t+1]
            EHWP2_luc = np.zeros([nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty)
            EHWP2_luc[:,:,:,:t+1] = (r_HWP2*(1-r_HWP2**tt))[np.newaxis,np.newaxis,np.newaxis,t::-1] * CHWP2_luc[:,:,:,:t+1]
            EHWP3_luc = np.zeros([nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty) 
            EHWP3_luc[:,:,:,:t+1] = (r_HWP3*(1-r_HWP3**tt))[np.newaxis,np.newaxis,np.newaxis,t::-1] * CHWP3_luc[:,:,:,:t+1]
            ELUC = np.sum(np.sum(np.sum( RH1_luc + RH2_luc + EFIRE_luc + EHWP1_luc + EHWP2_luc + EHWP3_luc ,3),2),1)
            
            # Each biome
            bLSNK = D_rh1 + D_rh2 + D_efire - D_npp
            bLSNK[:,3] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.34*D_npp)[:,3]
            bLSNK[:,5] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.34*D_npp)[:,5]
            bLSNK[:,6] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.37*D_npp)[:,6]
            bLSNK[:,7] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.35*D_npp)[:,7]
            bLSNK[:,8] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.35*D_npp)[:,8]
            bLSNK[:,9] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.35*D_npp)[:,9]

            
            b1ELUC = np.sum(np.sum( RH1_luc + RH2_luc + EFIRE_luc + EHWP1_luc + EHWP2_luc + EHWP3_luc ,3),2)
            b2ELUC = np.sum(np.sum( RH1_luc + RH2_luc + EFIRE_luc + EHWP1_luc + EHWP2_luc + EHWP3_luc ,3),1)
            b1RH1_luc =  np.sum(np.sum( RH1_luc,3),2)
            b2RH1_luc =  np.sum(np.sum( RH1_luc,3),1)
            b1RH2_luc =  np.sum(np.sum( RH2_luc,3),2)
            b2RH2_luc =  np.sum(np.sum( RH2_luc,3),1)
            b1EFIRE_luc = np.sum(np.sum( EFIRE_luc,3),2)
            b2EFIRE_luc = np.sum(np.sum( EFIRE_luc,3),1)
            b1EHWP1_luc = np.sum(np.sum( EHWP1_luc,3),2)
            b2EHWP1_luc = np.sum(np.sum( EHWP1_luc,3),1)
            b1EHWP2_luc = np.sum(np.sum( EHWP2_luc,3),2)
            b2EHWP2_luc = np.sum(np.sum( EHWP2_luc,3),1)
            b1EHWP3_luc = np.sum(np.sum( EHWP3_luc,3),2)
            b2EHWP3_luc = np.sum(np.sum( EHWP3_luc,3),1)

            #biomass burning
            for VAR in ['CO2','CH4','N2O','NOX','CO','VOC','SO2','NH3','OC','BC']:
                exec('D_EBB_'+VAR+' = np.sum( alpha_BB_'+VAR+'*(igni_0*cveg_0*D_AREA + D_efire*AREA_0 + D_efire*D_AREA) ,1)')
                exec('D_EBB_'+VAR+' += np.sum(np.sum(np.sum( alpha_BB_'+VAR+'[:,:,np.newaxis,np.newaxis]*EHWP1_luc ,3),2),1)')
                exec('D_EBB_'+VAR+' += np.sum(np.sum(np.sum( alpha_BB_'+VAR+'[:,np.newaxis,:,np.newaxis]*EFIRE_luc ,3),2),1)')
                           
            # stocks
            CVEG_luc += (p**-1) * (NPP_luc - FMORT_luc - EFIRE_luc)
            CSOIL1_luc += (p**-1) * (FMORT_luc - FMET_luc - RH1_luc)
            CSOIL2_luc += (p**-1) * (FMET_luc - RH2_luc)
            CHWP1_luc += (p**-1) * -EHWP1_luc
            CHWP2_luc += (p**-1) * -EHWP2_luc
            CHWP3_luc += (p**-1) * -EHWP3_luc
            #-------------
            # 4. CHEMISTRY
            #-------------

            # factors
            D_kOH = f_kOH(D_CH4,D_O3s,fT*D_gst,np.sum(ENOX[t]+D_EBB_NOX),np.sum(ECO[t]+D_EBB_CO),np.sum(EVOC[t]+D_EBB_VOC))
            D_hv = f_hv(D_N2O_lag,D_EESC,fT*D_gst)
            # fluxes
            D_OHSNK_CH4 = -alpha_CH4/tau_CH4_OH * (CH4_0*D_kOH + D_CH4 + D_kOH*D_CH4)
            D_HVSNK_CH4 = -alpha_CH4/tau_CH4_hv * (CH4_0*D_hv + D_CH4_lag + D_hv*D_CH4_lag)
            D_XSNK_CH4 = -alpha_CH4*(1/tau_CH4_soil + 1/tau_CH4_ocean) * D_CH4
            D_HVSNK_N2O = -alpha_N2O/tau_N2O_hv * (N2O_0*D_hv + D_N2O_lag + D_hv*D_N2O_lag)
            for VAR in ['HFC','PFC','ODS']:
                exec('D_OHSNK_'+VAR+' = -alpha_'+VAR+'/tau_'+VAR+'_OH * ('+VAR+'_0*D_kOH + D_'+VAR+' + D_kOH*D_'+VAR+')')
                exec('D_HVSNK_'+VAR+' = -alpha_'+VAR+'/tau_'+VAR+'_hv * ('+VAR+'_0*D_hv + D_'+VAR+'_lag + D_hv*D_'+VAR+'_lag)')
                exec('D_XSNK_'+VAR+' = -alpha_'+VAR+'/tau_'+VAR+'_othr * D_'+VAR)
            # stocks
            D_O3t = chi_O3t_CH4*np.log(1+D_CH4/CH4_0) + Gamma_O3t*fT*D_gst
            D_O3t += chi_O3t_NOX*np.sum(w_reg_NOX*np.sum(p_reg4*(ENOX[t]+D_EBB_NOX)[:,np.newaxis],0))
            D_O3t += chi_O3t_CO*np.sum(w_reg_CO*np.sum(p_reg4*(ECO[t]+D_EBB_CO)[:,np.newaxis],0))
            D_O3t += chi_O3t_VOC*np.sum(w_reg_VOC*np.sum(p_reg4*(EVOC[t]+D_EBB_VOC)[:,np.newaxis],0))
            D_EESC = np.sum(f_fracrel(tau_lag) * (n_Cl+alpha_Br*n_Br) * D_ODS_lag)
            D_O3s = chi_O3s_EESC*D_EESC + chi_O3s_N2O*D_N2O_lag * (1-D_EESC/EESC_x) + Gamma_O3s*fT*D_gst
            D_SO4 = alpha_SO4*tau_SO2*np.sum(w_reg_SO2*np.sum(p_reg4*(ESO2[t]+D_EBB_SO2)[:,np.newaxis],0)) + alpha_SO4*tau_DMS*0 + Gamma_SO4*fT*D_gst
            D_POA = tau_OMff*alpha_POM*np.sum(w_reg_OC*np.sum(p_reg4*(EOC[t])[:,np.newaxis],0)) + tau_OMbb*alpha_POM*np.sum(D_EBB_OC) + Gamma_POA*fT*D_gst
            D_BC = tau_BCff*np.sum(w_reg_BC*np.sum(p_reg4*(EBC[t])[:,np.newaxis],0)) + tau_BCbb*np.sum(D_EBB_BC) + Gamma_BC*fT*D_gst
            D_NO3 = alpha_NO3*tau_NOX*np.sum(ENOX[t]+D_EBB_NOX) + alpha_NO3*tau_NH3*np.sum(ENH3[t]+D_EBB_NH3) + Gamma_NO3*fT*D_gst
            D_SOA = tau_VOC*np.sum(EVOC[t]+D_EBB_VOC) + tau_BVOC*0 + Gamma_SOA*fT*D_gst
            D_DUST = 0*( tau_DUST*0 + Gamma_DUST*fT*D_gst )
            D_SALT = 0*( tau_SALT*0 + Gamma_SALT*fT*D_gst )
            D_AERh = solub_SO4*D_SO4 + solub_POA*D_POA + solub_BC*D_BC + solub_NO3*D_NO3 + solub_SOA*D_SOA + solub_DUST*D_DUST + solub_SALT*D_SALT
                        
            #--------------
            # 5. ATMOSPHERE
            #--------------
            # stocks            
            # Covid-19                         
            if covid== 1:           
                if t > 319 and t < effcovid + 319:
                    D_CO2 += (p**-1) * (1/alpha_CO2) * (np.sum(EFF[t]) + RedEFF[t-effcovid-320] + np.sum(ELUC) + LSNK + OSNK + Emi_exp[t] + Emi_pet[t])                        
                    Cumemission = np.sum(EFF[t]) + RedEFF[t-effcovid-320] + np.sum(ELUC) + Emi_exp[t] + Emi_pet[t]
                    REFF = np.sum(EFF[t]) + RedEFF[t-effcovid-320]
                    EFNPK = Emi_exp[t]
                    EFNPKT = Emi_expT[t]
                    CovidFF = Emi_pet[t]
                else:
                    D_CO2 += (p**-1) * (1/alpha_CO2) * (np.sum(EFF[t]) + np.sum(ELUC) + LSNK + OSNK + Emi_exp[t] + Emi_pet[t])                        
                    Cumemission = np.sum(EFF[t]) + np.sum(ELUC) + Emi_exp[t] + Emi_pet[t]
                    REFF = np.sum(EFF[t])
                    EFNPK = Emi_exp[t]
                    EFNPKT = Emi_expT[t]
                    CovidFF = Emi_pet[t]
            else:
                D_CO2 += (p**-1) * (1/alpha_CO2) * (np.sum(EFF[t]) + np.sum(ELUC) + LSNK + OSNK + Emi_exp[t])
                Cumemission = np.sum(EFF[t]) + np.sum(ELUC) + Emi_exp[t] 
                REFF = np.sum(EFF[t])
                EFNPK = Emi_exp[t]
                EFNPKT = Emi_expT[t]

            
            # CH4
            D_CH4 += (p**-1) * (1/alpha_CH4) * (np.sum(ECH4[t]) + np.sum(D_EBB_CH4) + np.sum(D_EWET) + D_OHSNK_CH4 + D_HVSNK_CH4 + D_XSNK_CH4)
            CumCH4 = np.sum(ECH4[t]) + np.sum(D_EBB_CH4) + np.sum(D_EWET) + D_OHSNK_CH4 + D_HVSNK_CH4 + D_XSNK_CH4
            # N2O
            D_N2O += (p**-1) * (1/alpha_N2O) * (np.sum(EN2O[t]) + np.sum(D_EBB_N2O) + D_HVSNK_N2O) 
            CumN2O =  np.sum(EN2O[t]) + np.sum(D_EBB_N2O) + D_HVSNK_N2O
            
            D_HFC += (p**-1) * (1/alpha_HFC) * (np.sum(EHFC[t],0) + D_OHSNK_HFC + D_HVSNK_HFC + D_XSNK_HFC)
            D_PFC += (p**-1) * (1/alpha_PFC) * (np.sum(EPFC[t],0) + D_OHSNK_PFC + D_HVSNK_PFC + D_XSNK_PFC)
            D_ODS += (p**-1) * (1/alpha_ODS) * (np.sum(EODS[t],0) + D_OHSNK_ODS + D_HVSNK_ODS + D_XSNK_ODS)
            # Carbon reduction policy:EFF and ELUC
            if mod_Carbonreduction == 'Carbon-Neutrality':
                # Action time 
                # peak carbon dioxide emissions time
                pt = Actime - 1700
                # Carbon neutrality time
                cnt = pt + 30             
                x1 = [pt,cnt]; y1 = [np.sum(EFF[pt]) + np.sum(ELUC),0]
                RCO2 = np.zeros([ind_final+1,1])
                # Fossil fuel emmissions
                y2 = [np.sum(EFF[pt]) + np.sum(ELUC),0]
                Ry2 = np.zeros([ind_final+1,1])
                for i in range(pt,cnt):
                    RCO2[i,0] = np.interp(i,x1,y1)  
                    Ry2[i,0] = np.interp(i,x1,y2)
                if t > pt:
                    D_CO2 = D_CO2 - (p**-1) * (1/alpha_CO2) * (np.sum(EFF[t]) + np.sum(ELUC) - RCO2[t])
                    Cumemission = Cumemission - (np.sum(EFF[t]) + np.sum(ELUC) - RCO2[t])
                    REFF = np.sum(EFF[t]) - (np.sum(EFF[t]) + np.sum(ELUC) - RCO2[t])
                    REFFCN = - (np.sum(EFF[t]) - Ry2[t])
                    
            else:
                pass
            
            if mod_Carbonreduction == 'BECCS':

                # Fossil fuel
                EFFSSP70 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP7.0_EFF2200.csv','r'))], dtype=dty)[:,0]
                EFFSSP60 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP6.0_EFF2200.csv','r'))], dtype=dty)[:,0]
                EFFSSP45 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP4.5_EFF2200.csv','r'))], dtype=dty)[:,0]
                EFFSSP26 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP2.6_EFF2200.csv','r'))], dtype=dty)[:,0]
                # CH4
                CH4SSP70 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP7.0_CH42200.csv','r'))], dtype=dty)[:,0]
                CH4SSP60 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP6.0_CH42200.csv','r'))], dtype=dty)[:,0]
                CH4SSP45 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP4.5_CH42200.csv','r'))], dtype=dty)[:,0]
                CH4SSP26 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP2.6_CH42200.csv','r'))], dtype=dty)[:,0]
                # N2O
                N2OSSP70 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP7.0_N2O2200.csv','r'))], dtype=dty)[:,0]
                N2OSSP60 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP6.0_N2O2200.csv','r'))], dtype=dty)[:,0]
                N2OSSP45 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP4.5_N2O2200.csv','r'))], dtype=dty)[:,0]
                N2OSSP26 = np.array([line for line in csv.reader(open('data/'+Filein+'/SSP2.6_N2O2200.csv','r'))], dtype=dty)[:,0]

                pt = Actime - 1700
                cnt = pt + 10
                x1 = [pt,cnt];
                y1_eff = [EFFSSP70[pt],EFFSSP45[cnt]]
                y1_ch4 = [CH4SSP70[pt],CH4SSP26[cnt]]
                y1_n2o = [N2OSSP70[pt],N2OSSP26[cnt]]
                           
                RCO2 = np.zeros([ind_final+1,1])
                RCH4 = np.zeros([ind_final+1,1])
                RN2O = np.zeros([ind_final+1,1])
                
                for i in range(pt,cnt):
                    RCO2[i,0] = np.interp(i,x1,y1_eff)  
                    RCH4[i,0] = np.interp(i,x1,y1_ch4)
                    RN2O[i,0] = np.interp(i,x1,y1_n2o)                            
                                          
                ## ratio of biomass transport and treatment               
                RatTT = 0.092
                ## Add BECCS
                BECCSall = np.zeros([1],dtype=dty)								
                ''' BECCS, choose to open or close''' 											
                if t > Actime -1700-1:
                    ## consider only the first part multiply water content
                    procot = AreaPCouCot[t,:]*YieldPCouCot[t,:]; promai = AreaPCouMai[t,:]*YieldPCouMai[t,:]
                    prorub = AreaPCouRub[t,:]*YieldPCouRub[t,:]; proric = AreaPCouRic[t,:]*YieldPCouRic[t,:]
                    prowhe = AreaPCouRic[t,:]*YieldPCouRic[t,:]; prorest = AreaPCouRest[t,:]*YieldPCouRest[t,:] 
                    BECCSall = np.sum(2.2324*procot[:,np.newaxis]+0.7565*promai[:,np.newaxis]+0.7677*prorub[:,np.newaxis]\
                                      +0.7312*proric[:,np.newaxis]+0.7677*prowhe[:,np.newaxis]+0.4472*prorest[:,np.newaxis])                                             
                    BECCSall *= 1-RatTT

                    if t < cnt:
                        # EFF
                        D_CO2 = D_CO2 - (p**-1) * (1/alpha_CO2) * (BECCSall + min(EFFSSP70[t]-RCO2[t], EFFSSP70[t]-EFFSSP45[t]))
                        Cumemission = Cumemission - (BECCSall + min(EFFSSP70[t]-RCO2[t], EFFSSP70[t]-EFFSSP45[t]))       
                        REFF = np.sum(EFF[t]) - (BECCSall + min(EFFSSP70[t]-RCO2[t], EFFSSP70[t]-EFFSSP45[t]))

                        # CH4
                        D_CH4=  D_CH4 - (p**-1) * (1/alpha_CH4) * (min(CH4SSP70[t]-RCH4[t], CH4SSP70[t]-CH4SSP26[t]))
                        CumCH4 = CumCH4 - (min(CH4SSP70[t]-RCH4[t], CH4SSP70[t]-CH4SSP26[t]))
                        RECH4 = - (min(CH4SSP70[t]-RCH4[t], CH4SSP70[t]-CH4SSP26[t]))
                        # N2O
                        D_N2O = D_N2O - (p**-1) * (1/alpha_N2O) * (min(N2OSSP70[t]-RN2O[t], N2OSSP70[t]-N2OSSP26[t]))
                        CumN2O = CumN2O - (min(N2OSSP70[t]-RN2O[t], N2OSSP70[t]-N2OSSP26[t]))
                        REN2O = - (min(N2OSSP70[t]-RN2O[t], N2OSSP70[t]-N2OSSP26[t]))

                    else:
                        # EFF
                        D_CO2 = D_CO2 - (p**-1) * (1/alpha_CO2) * (BECCSall + EFFSSP70[t] - EFFSSP45[t])
                        Cumemission = Cumemission - (BECCSall + EFFSSP70[t] - EFFSSP45[t])       
                        REFF = np.sum(EFF[t]) - (BECCSall + EFFSSP70[t] - EFFSSP45[t])

                        # CH4
                        D_CH4=  D_CH4 - (p**-1) * (1/alpha_CH4) * (CH4SSP70[t]-CH4SSP26[t])
                        CumCH4 = CumCH4 - (CH4SSP70[t]-CH4SSP26[t])
                        RECH4 = - (CH4SSP70[t]-CH4SSP26[t])
                        # N2O
                        D_N2O = D_N2O - (p**-1) * (1/alpha_N2O) * (N2OSSP70[t]-N2OSSP26[t])
                        CumN2O = CumN2O - (N2OSSP70[t]-N2OSSP26[t])
                        REN2O = - (N2OSSP70[t]-N2OSSP26[t])

                else:
                    pass
            
            #Cumemission += CumN2O*127*(10**-3)    

            for VAR in ['CH4','N2O','HFC','PFC','ODS']:
                exec('D_'+VAR+'_lag += (p**-1) * ((1/tau_lag)*D_'+VAR+' - (1/tau_lag)*D_'+VAR+'_lag)')

            # FORCE
            if force_CO2:
                D_CO2 = D_CO2_force[t]
            
            if force_GHG:
                D_CO2 = D_CO2_force[t]
                D_CH4 = D_CH4_force[t]
                D_N2O = D_N2O_force[t]
            
            if force_halo:
                D_HFC[:] = D_HFC_force[t]
                D_PFC[:] = D_PFC_force[t]
                D_ODS[:] = D_ODS_force[t]
            
            #-----------
            # 6. CLIMATE
            #-----------

            # fluxes
            # per component
            RF_CO2 = f_RF_CO2(D_CO2)
            RF_CH4 = f_RF_CH4(D_CH4)-(f_RF_overlap(D_CH4,D_N2O)-f_RF_overlap(0,D_N2O))
            RF_H2Os = f_RF_H2Os(D_CH4_lag)
            RF_N2O = f_RF_N2O(D_N2O)-(f_RF_overlap(D_CH4,D_N2O)-f_RF_overlap(D_CH4,0))
            RF_halo = np.sum(radeff_HFC*D_HFC) + np.sum(radeff_PFC*D_PFC) + np.sum(radeff_ODS*D_ODS)
            for VAR in ['O3t','O3s','SO4','POA','BC','NO3','SOA','DUST','SALT']:
                exec('RF_'+VAR+' = radeff_'+VAR+'*D_'+VAR)
            RF_cloud = k_BC_adjust*RF_BC + Phi_0*np.log(1+max(-0.9,D_AERh/AERh_0))
            RF_BCsnow = radeff_BCsnow*np.sum(w_reg_BCsnow*np.sum(p_reg9*(EBC[t]+D_EBB_BC)[:,np.newaxis],0))
            RF_LCC = np.sum(alpha_LCC*D_AREA)
            if t > 317:
                RF_LCC = np.sum(alpha_LCC*D_AREA_t[317])
                
            # FORCE
            if force_RFs:
                for VAR in ['CO2','CH4','H2Os','N2O','halo']+['O3t','O3s','SO4','POA','BC','NO3','SOA','DUST','SALT']+['cloud','BCsnow','LCC']:
                    exec('RF_'+VAR+' = RF_'+VAR+'_force[t]')

            # totals
            RF = RF_CO2 + RF_CH4 + RF_H2Os + RF_N2O + RF_halo + RF_O3t + RF_O3s + RF_SO4 + RF_POA + RF_BC + RF_NO3 + RF_SOA + RF_DUST + RF_SALT + RF_cloud + RF_BCsnow + RF_LCC + RFcon[t] + RFvolc[t] + RFsolar[t]
            RF_warm = RF_CO2 + RF_CH4 + RF_H2Os + RF_N2O + RF_halo + RF_O3t + RF_O3s + RF_SO4 + RF_POA + RF_BC + RF_NO3 + RF_SOA + RF_DUST + RF_SALT + RF_cloud + warmeff_BCsnow*RF_BCsnow + warmeff_LCC*RF_LCC + RFcon[t] + warmeff_volc*RFvolc[t] + RFsolar[t]
            RF_atm = p_atm_CO2*RF_CO2 + p_atm_noCO2*(RF_CH4+RF_N2O+RF_halo) + p_atm_O3t*RF_O3t + p_atm_strat*(RF_O3s+RF_H2Os) + p_atm_scatter*(RF_SO4+RF_POA+RF_NO3+RF_SOA+RF_DUST+RF_SALT+RFvolc[t]) + p_atm_absorb*RF_BC + p_atm_cloud*(RF_cloud+RFcon[t]) + p_atm_alb*(RF_BCsnow+RF_LCC) + p_atm_solar*RFsolar[t]

            # FORCE
            if force_RF:
                RF_warm = RF_force[t] * (RF_warm/RF)
                RF_atm = RF_force[t] * (RF_atm/RF)
                RF = RF_force[t]

            # stocks
            # temperatures
            D_gst += (p**-1) * (1/tau_gst) * (lambda_0*RF_warm - D_gst - theta_0*(D_gst-D_gst0))
            D_gst0 += (p**-1) * (1/tau_gst0) * theta_0*(D_gst-D_gst0)
            D_sst = w_reg_sst*D_gst
            D_lst = w_reg_lst*D_gst
            # precipitations
            D_gyp = alpha_gyp*D_gst + beta_gyp*RF_atm
            D_lyp = w_reg_lyp*D_gyp
            # ocean
            D_OHC += (p**-1) * p_OHC * alpha_OHC * (RF - D_gst/lambda_0)
            D_pH = f_pH(D_CO2)

            # FORCE
            if force_clim:
                D_gst = D_gst_force[t]
                D_sst = D_sst_force[t]
                D_lst = D_lst_force[t]
                D_lyp = D_lyp_force[t]

            #-----------
            # Y. SAVE
            #-----------
            for var in var_timeseries:
                exec(var+'_t[t] += (p**-1) * '+var)

            #---------
            # Z. TESTS
            #---------        
            if np.isnan(np.sum(D_CO2)):
                print 'D_CO2 = NaN at t = '+str(t)+' and tt = '+str(tt)
                print 'OSNK = '+str(np.sum(OSNK))
                print 'LSNK = '+str(np.sum(LSNK))
                print 'ELUC = '+str(np.sum(ELUC))
                break
            if np.isnan(np.sum(D_CH4)):
                print 'D_CH4 = NaN at t = '+str(t)+' and tt = '+str(tt)              
                print 'D_EWET = '+str(np.sum(D_EWET))
                print 'D_OHSNK = '+str(np.sum(D_OHSNK_CH4))
                print 'D_HVSNK = '+str(np.sum(D_HVSNK_CH4))
                break
            if np.isnan(np.sum(D_gst)):
                print 'D_gst = NaN at t = '+str(t)+' and tt = '+str(tt)
                print 'RF_CO2 = '+str(np.sum(RF_CO2))
                print 'RF_CH4 = '+str(np.sum(RF_CH4))
                print 'RF_H2Os = '+str(np.sum(RF_H2Os))
                print 'RF_N2O = '+str(np.sum(RF_N2O))
                print 'RF_halo = '+str(np.sum(RF_halo))
                print 'RF_O3t = '+str(np.sum(RF_O3t))
                print 'RF_O3s = '+str(np.sum(RF_O3s))
                print 'RF_SO4 = '+str(np.sum(RF_SO4))
                print 'RF_POA = '+str(np.sum(RF_POA))
                print 'RF_BC = '+str(np.sum(RF_BC))
                print 'RF_NO3 = '+str(np.sum(RF_NO3))
                print 'RF_SOA = '+str(np.sum(RF_SOA))
                print 'RF_DUST = '+str(np.sum(RF_DUST))
                print 'RF_SALT = '+str(np.sum(RF_SALT))
                print 'RF_cloud = '+str(np.sum(RF_cloud))
                print 'RF_BCsnow = '+str(np.sum(RF_BCsnow))
                print 'RF_LCC = '+str(np.sum(RF_LCC))
                break

        if np.isnan(np.sum(D_CO2))|np.isnan(np.sum(D_CH4))|np.isnan(np.sum(D_gst)):
            for var in var_timeseries:
                if (t < ind_final):
                    exec(var+'_t[t+1:] = np.nan')
            break

    #===========
    # C. FIGURES
    #===========

    if plot is 'all' or plot is 'CO2' or 'CO2' in plot:
        plot_CO2(D_CO2_t,OSNK_t,LSNK_t,ELUC_t,EFF,D_AREA_t,D_npp_t,D_efire_t,D_fmort_t,D_rh1_t,D_fmet_t,D_rh2_t,D_FIN_t,D_FOUT_t,D_FCIRC_t,EFIRE_luc_t,FMORT_luc_t,RH1_luc_t,RH2_luc_t,EHWP1_luc_t,EHWP2_luc_t,EHWP3_luc_t)
    if plot is 'all' or plot is 'CH4' or 'CH4' in plot:
        plot_CH4(D_CH4_t,D_OHSNK_CH4_t,D_HVSNK_CH4_t,D_XSNK_CH4_t,D_EWET_t,D_EBB_CH4_t,ECH4)
    if plot is 'all' or plot is 'N2O' or 'N2O' in plot:
        plot_N2O(D_N2O_t,D_HVSNK_N2O_t,D_EBB_N2O_t,EN2O)
    if plot is 'all' or plot is 'O3' or 'O3' in plot:
        plot_O3(D_O3t_t,D_O3s_t,D_EESC_t,D_N2O_lag_t,D_gst_t)
    if plot is 'all' or plot is 'AER' or 'AER' in plot:
        plot_AER(D_SO4_t,D_POA_t,D_BC_t,D_NO3_t,D_SOA_t,D_AERh_t,RF_SO4_t,RF_POA_t,RF_BC_t,RF_NO3_t,RF_SOA_t,RF_cloud_t)
    if plot is 'all' or plot is 'clim' or 'clim' in plot:
        plot_clim(RF_t,D_gst_t,D_gyp_t,RF_CO2_t,RF_CH4_t,RF_H2Os_t,RF_N2O_t,RF_halo_t,RF_O3t_t,RF_O3s_t,RF_SO4_t,RF_POA_t,RF_BC_t,RF_NO3_t,RF_SOA_t,RF_cloud_t,RF_BCsnow_t,RF_LCC_t,RFcon,RFvolc,RFsolar)

    #===========
    # D. OUTPUTS
    #===========

    output = []
    for var in var_output:
        exec('output.append('+var+'_t)')      

    return output


##################################################
#   2. CONTROL PLOTS
##################################################

#=========
# 2.1. CO2
#=========

def plot_CO2(D_CO2,OSNK,LSNK,ELUC,EFF,D_AREA,D_npp,D_efire,D_fmort,D_rh1,D_fmet,D_rh2,D_FIN,D_FOUT,D_FCIRC,D_MORT_luc,D_EFIRE_luc,D_RH1_luc,D_RH2_luc,EHWP1_luc,EHWP2_luc,EHWP3_luc):
    plt.figure()

    # atmospheric CO2
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_CO2,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(CO2_ipcc)),CO2_ipcc-CO2_0,color='r',lw=2,ls='--',label='IPCC')    
    if (ind_final > ind_cdiac):
        plt.plot(1700+np.arange(min(len(CO2_rcp),ind_final+1)),CO2_rcp[:min(len(CO2_rcp),ind_final+1),0]-CO2_0,color='0.8',lw=2,ls=':',label='RCP2.6')
        plt.plot(1700+np.arange(min(len(CO2_rcp),ind_final+1)),CO2_rcp[:min(len(CO2_rcp),ind_final+1),1]-CO2_0,color='0.6',lw=2,ls=':',label='RCP4.5')
        plt.plot(1700+np.arange(min(len(CO2_rcp),ind_final+1)),CO2_rcp[:min(len(CO2_rcp),ind_final+1),2]-CO2_0,color='0.4',lw=2,ls=':',label='RCP6.0')
        plt.plot(1700+np.arange(min(len(CO2_rcp),ind_final+1)),CO2_rcp[:min(len(CO2_rcp),ind_final+1),3]-CO2_0,color='0.2',lw=2,ls=':',label='RCP8.5')
    plt.title('$\Delta$CO2 (ppm)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # budget fluxes
    ax = plt.subplot(2,3,2)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),np.sum(EFF,1),color='#666666',lw=2,label='EFF')
    plt.plot(1700+np.arange(ind_final+1),np.sum(ELUC,1),color='#993300',lw=2,label='ELUC')
    plt.plot(1700+np.arange(ind_final+1),OSNK,color='#000099',lw=2,label='OSNK')
    plt.plot(1700+np.arange(ind_final+1),LSNK,color='#009900',lw=2,label='LSNK')
    plt.plot(1700+np.arange(ind_final)+1,alpha_CO2*(D_CO2[1:]-D_CO2[:-1]),color='#FFCC00',lw=2,label='d_CO2')
    plt.plot(1700+np.arange(len(EFF_gcp)),EFF_gcp,color='#666666',ls='--')
    plt.plot(1700+np.arange(len(ELUC_gcp)),ELUC_gcp,color='#CC3300',ls='--')
    plt.plot(1700+np.arange(len(OSNK_gcp)),OSNK_gcp,color='#000099',ls='--')
    plt.plot(1700+np.arange(len(LSNK_gcp)),LSNK_gcp,color='#009900',ls='--')
    plt.plot(1700+np.arange(len(d_CO2_gcp)),d_CO2_gcp,color='#FFCC00',ls='--')
    plt.plot([1700,1700],[0,0],'k--',label='GCP')
    plt.title('CO2 fluxes (GtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # airborne fraction
    ax = plt.subplot(2,3,3)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final)+1,alpha_CO2*(D_CO2[1:]-D_CO2[:-1])/np.sum(EFF+ELUC,1)[1:],color='#FFCC00',lw=1,label='AF')
    plt.plot(1700+np.arange(ind_final+1),-OSNK/np.sum(EFF+ELUC,1),color='#000099',lw=1,label='OF')
    plt.plot(1700+np.arange(ind_final+1),-LSNK/np.sum(EFF+ELUC,1),color='#009900',lw=1,label='LF')    
    plt.plot(np.arange(1959,1700+ind_cdiac+1),np.ones([ind_cdiac-259+1])*np.mean((alpha_CO2*(D_CO2[1:]-D_CO2[:-1])/np.sum(EFF+ELUC,1)[1:])[259-1:ind_cdiac]),color='k',lw=2,label='OSCAR')
    plt.plot(np.arange(1959,1700+ind_cdiac+1),np.ones([ind_cdiac-259+1])*np.mean((d_CO2_gcp/(EFF_gcp+ELUC_gcp))[259:ind_cdiac+1]),color='r',lw=2,ls='--',label='GCP')
    plt.title('airborne fraction',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))             
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])
    ax.set_ylim([-0.2,1.2])

    # ELUC details
    ax = plt.subplot(2,3,4)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),np.sum(ELUC,1),color='k',ls='-.',lw=2,label='ELUC')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(D_EFIRE_luc+D_RH1_luc+D_RH2_luc,4),3),2),1),color='#009900',lw=2,label='ELUC_bio')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(EHWP1_luc+EHWP2_luc+EHWP3_luc,4),3),2),1),color='#993300',lw=2,label='ELUC_hwp')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(EHWP1_luc,4),3),2),1),color='#FF3300',lw=1,label='EHWP1')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(EHWP2_luc,4),3),2),1),color='#CC9900',lw=1,label='EHWP2')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(EHWP3_luc,4),3),2),1),color='#663300',lw=1,label='EHWP3')    
    plt.title('ELUC fluxes (GtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))

    # LSNK details
    ax = plt.subplot(2,3,5)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),-LSNK,color='k',lw=2,ls='-.',label='$-$LSNK')    
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(D_npp*(AREA_0+D_AREA),2),1),color='#009900',lw=2,label='D_NPP')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(D_efire*(AREA_0+D_AREA),2),1),color='#FF3300',lw=2,label='D_EFIRE')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(D_fmort*(AREA_0+D_AREA),2),1),color='#336633',lw=2,label='D_FMORT')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum((D_rh1+D_rh2)*(AREA_0+D_AREA),2),1),color='#663300',lw=2,label='D_RH')
    plt.title('LSNK fluxes (GtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))

    # OSNK details
    ax = plt.subplot(2,3,6)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),-OSNK,color='k',lw=2,ls='-.',label='$-$OSNK')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_FIN,1),color='#000099',lw=2,label='D_FIN')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_FOUT,1),color='#0099FF',lw=2,label='D_FOUT')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_FCIRC,1),color='#663399',lw=2,label='D_FCIRC')             
    plt.title('OSNK fluxes (GtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))

#=========
# 2.2. CH4
#=========

def plot_CH4(D_CH4,D_OHSNK_CH4,D_HVSNK_CH4,D_XSNK_CH4,D_EWET,D_EBB_CH4,ECH4):
    plt.figure()

    # atmospheric CH4
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_CH4,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(CH4_ipcc)),CH4_ipcc-CH4_0,color='r',lw=2,ls='--',label='IPCC')    
    if (ind_final > ind_cdiac):
        plt.plot(1700+np.arange(min(len(CH4_rcp),ind_final+1)),CH4_rcp[:min(len(CH4_rcp),ind_final+1),0]-CH4_0,color='0.8',lw=2,ls=':',label='RCP2.6')
        plt.plot(1700+np.arange(min(len(CH4_rcp),ind_final+1)),CH4_rcp[:min(len(CH4_rcp),ind_final+1),1]-CH4_0,color='0.6',lw=2,ls=':',label='RCP4.5')
        plt.plot(1700+np.arange(min(len(CH4_rcp),ind_final+1)),CH4_rcp[:min(len(CH4_rcp),ind_final+1),2]-CH4_0,color='0.4',lw=2,ls=':',label='RCP6.0')
        plt.plot(1700+np.arange(min(len(CH4_rcp),ind_final+1)),CH4_rcp[:min(len(CH4_rcp),ind_final+1),3]-CH4_0,color='0.2',lw=2,ls=':',label='RCP8.5')
    plt.title('$\Delta$CH4 (ppb)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # budget fluxes
    ax = plt.subplot(2,3,2)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),np.sum(ECH4,1),color='#666666',lw=2,label='ECH4')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_EBB_CH4,1),color='#993300',lw=2,label='D_EBB')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_EWET,1),color='#006666',lw=2,label='D_EWET')
    plt.plot(1700+np.arange(ind_final+1),(D_OHSNK_CH4+D_HVSNK_CH4+D_XSNK_CH4),color='#990066',lw=2,label='D_SNK')
    plt.plot(1700+np.arange(ind_final)+1,alpha_CH4*(D_CH4[1:]-D_CH4[:-1]),color='#FFCC00',lw=2,label='d_CH4')
    plt.title('CH4 fluxes (MtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # lifetime
    ax = plt.subplot(2,3,3)
    plt.plot(1700+np.arange(ind_final+1),alpha_CH4*(CH4_0+D_CH4)/(alpha_CH4*CH4_0*(1/tau_CH4_OH+1/tau_CH4_hv+1/tau_CH4_soil+1/tau_CH4_ocean)-D_OHSNK_CH4-D_HVSNK_CH4-D_XSNK_CH4),color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(ind_final+1),alpha_CH4*(CH4_0+D_CH4)/(alpha_CH4*CH4_0/tau_CH4_OH-D_OHSNK_CH4),color='k',lw=1,label='OH only')
    plt.title('CH4 lifetime (yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # wetlands
    ax = plt.subplot(2,3,4)
    plt.title('wetlands',fontsize='medium')
 
    # biomass burning
    ax = plt.subplot(2,3,5)
    plt.title('biomass burning',fontsize='medium')


#=========
# 2.3. N2O
#=========

def plot_N2O(D_N2O,D_HVSNK_N2O,D_EBB_N2O,EN2O):
    plt.figure()

    # atmospheric N2O
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_N2O,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(N2O_ipcc)),N2O_ipcc-N2O_0,color='r',lw=2,ls='--',label='IPCC')    
    if (ind_final > ind_cdiac):
        plt.plot(1700+np.arange(min(len(N2O_rcp),ind_final+1)),N2O_rcp[:min(len(N2O_rcp),ind_final+1),0]-N2O_0,color='0.8',lw=2,ls=':',label='RCP2.6')
        plt.plot(1700+np.arange(min(len(N2O_rcp),ind_final+1)),N2O_rcp[:min(len(N2O_rcp),ind_final+1),1]-N2O_0,color='0.6',lw=2,ls=':',label='RCP4.5')
        plt.plot(1700+np.arange(min(len(N2O_rcp),ind_final+1)),N2O_rcp[:min(len(N2O_rcp),ind_final+1),2]-N2O_0,color='0.4',lw=2,ls=':',label='RCP6.0')
        plt.plot(1700+np.arange(min(len(N2O_rcp),ind_final+1)),N2O_rcp[:min(len(N2O_rcp),ind_final+1),3]-N2O_0,color='0.2',lw=2,ls=':',label='RCP8.5')
    plt.title('$\Delta$N2O (ppb)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # budget fluxes
    ax = plt.subplot(2,3,2)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),np.sum(EN2O,1),color='#666666',lw=2,label='EN2O')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_EBB_N2O,1),color='#993300',lw=2,label='D_EBB')
    plt.plot(1700+np.arange(ind_final+1),D_HVSNK_N2O,color='#990066',lw=2,label='D_SNK')
    plt.plot(1700+np.arange(ind_final)+1,alpha_N2O*(D_N2O[1:]-D_N2O[:-1]),color='#FFCC00',lw=2,label='d_N2O')
    plt.title('N2O fluxes (MtN/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # lifetime
    ax = plt.subplot(2,3,3)
    plt.plot(1700+np.arange(ind_final+1),alpha_N2O*(N2O_0+D_N2O)/(alpha_N2O*N2O_0/tau_N2O_hv-D_HVSNK_N2O),color='k',lw=2,label='OSCAR')
    plt.title('N2O lifetime (yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

#========
# 2.4. O3
#========

def plot_O3(D_O3t,D_O3s,D_EESC,D_N2O_lag,D_gst):
    plt.figure()

    # tropospheric O3
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_O3t,color='k',lw=2,label='OSCAR')
    plt.title('$\Delta$O3 trop. (DU)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # stratospheric O3
    ax = plt.subplot(2,3,2)
    plt.plot(1700+np.arange(ind_final+1),D_O3s,color='k',lw=2,label='OSCAR')
    plt.title('$\Delta$O3 strat. (DU)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # EESC
    ax = plt.subplot(2,3,3)
    plt.plot(1700+np.arange(ind_final+1),D_EESC,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(ind_final+1),(chi_O3s_N2O*D_N2O_lag*(1-D_EESC/EESC_x)/chi_O3s_EESC),color='k',lw=1,label='N2O effect')    
    plt.title('$\Delta$EESC (ppt)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # age-of-air
    ax = plt.subplot(2,3,4)
    plt.plot(1700+np.arange(ind_final+1),tau_lag/(1+gamma_age*D_gst),color='k',lw=2,label='OSCAR')
    plt.title('mean age-of-air (yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])    

#==============
# 2.5. Aerosols
#==============

def plot_AER(D_SO4,D_POA,D_BC,D_NO3,D_SOA,D_AERh,RF_SO4,RF_POA,RF_BC,RF_NO3,RF_SOA,RF_cloud):
    plt.figure()

    # atmospheric burden
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_SO4,color='b',lw=2,label='D_SO4')
    plt.plot(1700+np.arange(ind_final+1),D_POA,color='m',lw=2,label='D_POA')
    plt.plot(1700+np.arange(ind_final+1),D_BC,color='r',lw=2,label='D_BC')
    plt.plot(1700+np.arange(ind_final+1),D_NO3,color='g',lw=2,label='D_NO3')
    plt.plot(1700+np.arange(ind_final+1),D_SOA,color='y',lw=2,label='D_SOA')    
    plt.plot(1700+np.arange(ind_final+1),D_AERh,color='c',lw=2,label='D_AERh') 
    plt.title('$\Delta$ burdens (Tg)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # radiative forcing
    ax = plt.subplot(2,3,4)
    plt.plot([1700,1700+ind_final],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),RF_SO4,color='b',lw=2,label='RF_SO4')
    plt.errorbar([2010],[-0.40],yerr=[[0.20],[0.20]],marker='o',mfc='b',color='k')
    plt.plot(1700+np.arange(ind_final+1),RF_POA,color='m',lw=2,label='RF_POA')
    plt.errorbar([2010],[-0.29],yerr=[[-0.29*0.63],[-0.29*0.72]],marker='o',mfc='m',color='k')
    plt.plot(1700+np.arange(ind_final+1),RF_BC,color='r',lw=2,label='RF_BC')
    plt.errorbar([2010],[+0.60],yerr=[[+0.60*0.61],[+0.60*0.70]],marker='o',mfc='r',color='k')
    plt.plot(1700+np.arange(ind_final+1),RF_NO3,color='g',lw=2,label='RF_NO3')
    plt.errorbar([2010],[-0.11],yerr=[[0.19],[0.08]],marker='o',mfc='g',color='k')
    plt.plot(1700+np.arange(ind_final+1),RF_SOA,color='y',lw=2,label='RF_SOA')
    plt.errorbar([2010],[-0.03],yerr=[[0.24],[0.23]],marker='o',mfc='y',color='k') 
    plt.plot(1700+np.arange(ind_final+1),RF_cloud,color='c',lw=2,label='RF_cloud') 
    plt.errorbar([2010],[-0.45],yerr=[[0.75],[0.45]],marker='o',mfc='c',color='k')
    #plt.errorbar([2010],[-0.10],yerr=[[0.20],[0.20]],marker='o',mfc='0.5',color='k')
    plt.title('RF (W/m2)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,max(1700+ind_final,2010+10)])

#=============
# 2.6. Climate
#=============

def plot_clim(RF,D_gst,D_gyp,RF_CO2,RF_CH4,RF_H2Os,RF_N2O,RF_halo,RF_O3t,RF_O3s,RF_SO4,RF_POA,RF_BC,RF_NO3,RF_SOA,RF_cloud,RF_BCsnow,RF_LCC,RFcon,RFvolc,RFsolar):
    plt.figure()

    # radiative forcing
    ax = plt.subplot(2,3,1)
    plt.plot([1700,1700+ind_final],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),RF,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(RF_ipcc)),RF_ipcc,color='r',lw=2,ls='--',label='IPCC')
    plt.title('RF (W/m2)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # global temperature
    ax = plt.subplot(2,3,2)
    plt.plot([1700,1700+ind_final],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),D_gst-np.mean(D_gst[200:230]),color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(gst_giss)),gst_giss-np.mean(gst_giss[200:230]),color='b',ls='--',label='GISS')
    plt.plot(1700+np.arange(len(gst_had)),gst_had-np.mean(gst_had[200:230]),color='g',ls='--',label='Hadley')
    plt.plot(1700+np.arange(len(gst_ncdc)),gst_ncdc-np.mean(gst_ncdc[200:230]),color='m',ls='--',label='NCDC')
    plt.title('$\Delta$ temp. (K)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # global precipitations
    ax = plt.subplot(2,3,3)
    plt.plot(1700+np.arange(ind_final+1),D_gyp,color='k',lw=2,label='OSCAR')
    plt.title('$\Delta$ precip. (mm)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # RF details
    ax = plt.subplot(2,3,4)
    plt.plot([1700,1700+ind_final],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),RF_CO2+RF_CH4+RF_N2O+RF_halo+RF_H2Os,color='r',lw=2,label='WMGHG')
    plt.plot(1700+np.arange(len(RF_WMGHG_ipcc)),RF_WMGHG_ipcc,color='r',ls='--')
    plt.plot(1700+np.arange(ind_final+1),RF_O3t+RF_O3s,color='y',lw=2,label='O3')
    plt.plot(1700+np.arange(len(RF_O3_ipcc)),RF_O3_ipcc,color='y',ls='--')    
    plt.plot(1700+np.arange(ind_final+1),RF_SO4+RF_POA+RF_BC+RF_NO3+RF_SOA+RF_cloud,color='b',lw=2,label='AER')
    plt.plot(1700+np.arange(len(RF_AER_ipcc)),RF_AER_ipcc,color='b',ls='--')
    plt.plot(1700+np.arange(ind_final+1),RF_BCsnow+RF_LCC,color='g',lw=2,label='Alb.')
    plt.plot(1700+np.arange(len(RF_Alb_ipcc)),RF_Alb_ipcc,color='g',ls='--')
    plt.plot(1700+np.arange(ind_final+1),RFcon,color='k',ls='--',label='Ant.')
    plt.plot(1700+np.arange(ind_final+1),RFvolc+RFsolar,color='0.5',ls='--',label='Nat.')
    plt.title('RF (W/m2)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])
