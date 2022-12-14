import os
import math

from scipy.stats import *
from matplotlib.font_manager import FontProperties
import pandas as pd

##################################################
#   1. HISTORICAL SIMULATIONS
##################################################

#=============
# 1.1. LOADING
#=============

# load all
execfile('OSCAR.py')

# create forced GHG
for VAR in ['CO2','CH4','N2O']:
    exec('D_'+VAR+'_force = np.zeros([ind_final+1],dtype=dty)')
    exec('D_'+VAR+'_force[300:ind_final+1] = '+VAR+'_rcp[300:ind_final+1,3] - '+VAR+'_rcp[300,3]')
for VAR in ['HFC','PFC','ODS']:
    exec('D_'+VAR+'_force = np.zeros([ind_final+1,nb_'+VAR+'],dtype=dty)')
    exec('D_'+VAR+'_force[65:305+1] = '+VAR+'_cmip5[65:] - '+VAR+'_0')
    exec('D_'+VAR+'_force[:,:] *= (('+VAR+'_ipcc[305] - '+VAR+'_0)/D_'+VAR+'_force[305])[np.newaxis,:]')
    exec('D_'+VAR+'_force[305:312] = '+VAR+'_ipcc[305:ind_final+1] - '+VAR+'_0')
    exec('D_'+VAR+'_force[np.isnan(D_'+VAR+'_force)] = 0')

# create forced RF
# variables
for VAR in ['CO2','CH4','H2Os','N2O','halo','O3t','O3s','SO4','POA','BC','NO3','SOA','cloud','DUST','SALT','BCsnow','LCC']:
    exec('RF_'+VAR+'_force = np.zeros([ind_final+1],dtype=dty)')
# load CMIP5 RF
rf_dico = {'CH4':'CH4','N2O':'N2O','FGAS':'halo','ODS':'halo','OC':'POA','BC':'BC','SOX':'SO4','NOX':'NO3','BBAER':'SOA','DUST':'DUST','CLOUD':'cloud'}
for VAR in rf_dico.values():
    exec('RF_'+VAR+'_tmp = np.zeros([ind_final+1],dtype=dty)')
TMP = np.array([line for line in csv.reader(open('data/Historic_CMIP5/#DATA.Historic_CMIP5.1765-2005_(19for).RF.csv','r'))][1:],dtype=dty)
lgd = [line for line in csv.reader(open('data/Historic_CMIP5/#DATA.Historic_CMIP5.1765-2005_(19for).RF.csv','r'))][0]
for n in range(len(lgd)):
    if lgd[n] in rf_dico.keys():
        exec('RF_'+rf_dico[lgd[n]]+'_tmp[65:305+1] += TMP[:,n]')
# extend CMIP5 RF
for VAR in rf_dico.values():
    for t in range(50,66):
        exec('RF_'+VAR+'_tmp[t] = (t-50)/(66-50.) * RF_'+VAR+'_tmp[66]')
    for t in range(305+1,310+1):
        exec('RF_'+VAR+'_tmp[t] = RF_'+VAR+'_tmp[305] + (t-305.)/(310-305.) * (RF_'+VAR+'_tmp[305]-RF_'+VAR+'_tmp[300])')
# rescale CMIP5 RF
RF_CH4_tmp *= 0.48/RF_CH4_tmp[310]
RF_N2O_tmp *= 0.17/RF_N2O_tmp[310]
RF_halo_tmp *= 0.36/RF_halo_tmp[310]
RF_SO4_tmp *= -0.40/RF_SO4_tmp[310] * 0.35/0.32
RF_POA_tmp *= -0.29/RF_POA_tmp[310] * 0.35/0.32
RF_BC_tmp *= 0.60/RF_BC_tmp[310] * 0.35/0.32
RF_NO3_tmp *= -0.11/RF_NO3_tmp[310] * 0.35/0.32
RF_SOA_tmp *= -0.03/RF_SOA_tmp[310] * 0.35/0.32
RF_DUST_tmp *= -0.10/RF_DUST_tmp[310] * 0.35/0.32
RF_cloud_tmp *= -0.55/RF_cloud_tmp[310]
# reallocate IPCC RF
TMP = np.array([line for line in csv.reader(open('data/Historic_IPCC-AR5/#DATA.Historic_IPCC-AR5.1750-2011_(11for).RF.csv','r'))][1:], dtype=dty)
lgd = [line for line in csv.reader(open('data/Historic_IPCC-AR5/#DATA.Historic_IPCC-AR5.1750-2011_(11for).RF.csv','r'))][0]
RF_CO2_force[51:313] = TMP[:ind_final+1-51,lgd.index('CO2')]
RF_CH4_force[51:313] = (RF_CH4_tmp/(RF_CH4_tmp+RF_N2O_tmp+RF_halo_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('GHG Other')]
RF_N2O_force[51:313] = (RF_N2O_tmp/(RF_CH4_tmp+RF_N2O_tmp+RF_halo_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('GHG Other')]
RF_halo_force[51:313] = (RF_halo_tmp/(RF_CH4_tmp+RF_N2O_tmp+RF_halo_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('GHG Other')]
RF_O3t_force[51:313] = TMP[:ind_final+1-51,lgd.index('O3 (Trop)')]
RF_O3s_force[51:313] = TMP[:ind_final+1-51,lgd.index('O3 (Strat)')]
RF_SO4_force[51:313] = (RF_SO4_tmp/(RF_SO4_tmp+RF_POA_tmp+RF_BC_tmp+RF_NO3_tmp+RF_SOA_tmp+RF_DUST_tmp+RF_cloud_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('Aerosol (Total)')]
RF_POA_force[51:313] = (RF_POA_tmp/(RF_SO4_tmp+RF_POA_tmp+RF_BC_tmp+RF_NO3_tmp+RF_SOA_tmp+RF_DUST_tmp+RF_cloud_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('Aerosol (Total)')]
RF_BC_force[51:313] = (RF_BC_tmp/(RF_SO4_tmp+RF_POA_tmp+RF_BC_tmp+RF_NO3_tmp+RF_SOA_tmp+RF_DUST_tmp+RF_cloud_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('Aerosol (Total)')]
RF_NO3_force[51:313] = (RF_NO3_tmp/(RF_SO4_tmp+RF_POA_tmp+RF_BC_tmp+RF_NO3_tmp+RF_SOA_tmp+RF_DUST_tmp+RF_cloud_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('Aerosol (Total)')]
RF_SOA_force[51:313] = (RF_SOA_tmp/(RF_SO4_tmp+RF_POA_tmp+RF_BC_tmp+RF_NO3_tmp+RF_SOA_tmp+RF_DUST_tmp+RF_cloud_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('Aerosol (Total)')]
RF_DUST_force[51:313] = 0.5*(RF_DUST_tmp/(RF_SO4_tmp+RF_POA_tmp+RF_BC_tmp+RF_NO3_tmp+RF_SOA_tmp+RF_DUST_tmp+RF_cloud_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('Aerosol (Total)')]
RF_SALT_force[51:313] = 0.5*(RF_DUST_tmp/(RF_SO4_tmp+RF_POA_tmp+RF_BC_tmp+RF_NO3_tmp+RF_SOA_tmp+RF_DUST_tmp+RF_cloud_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('Aerosol (Total)')]
RF_cloud_force[51:313] = (RF_cloud_tmp/(RF_SO4_tmp+RF_POA_tmp+RF_BC_tmp+RF_NO3_tmp+RF_SOA_tmp+RF_DUST_tmp+RF_cloud_tmp))[51:313] * TMP[:ind_final+1-51,lgd.index('Aerosol (Total)')]
RF_LCC_force[51:313] = TMP[:ind_final+1-51,lgd.index('LUC')]
RF_H2Os_force[51:313] = TMP[:ind_final+1-51,lgd.index('H2O (Strat)')]
RF_BCsnow_force[51:313] = TMP[:ind_final+1-51,lgd.index('BC Snow')]

# create forced climate
# from Hadley center only (for consistency)
# global
D_gst_force = np.zeros([ind_final+1],dtype=dty)
D_gst_force[201:315] = gst_had[201:ind_final+1] - np.mean(gst_had[201:231])
# sea
D_sst_force = np.zeros([ind_final+1],dtype=dty)
D_sst_force[201:315] = sst_had[201:ind_final+1] - np.mean(sst_had[201:231])
# land
D_lst_force = np.zeros([ind_final+1,nb_regionI],dtype=dty)
D_lst_force[201:315] = lst_cru[201:ind_final+1] - np.mean(lst_cru[201:231],0)[np.newaxis,:]
D_lyp_force = np.zeros([ind_final+1,nb_regionI],dtype=dty)
D_lyp_force[201:315] = lyp_cru[201:ind_final+1] - np.mean(lyp_cru[201:231],0)[np.newaxis,:]

#=============
# 1.2. RUNNING
#=============
# list variables
list_global = ['OSNK','LSNK','D_CO2','D_OHSNK_CH4','D_HVSNK_CH4','D_XSNK_CH4','D_CH4','D_HVSNK_N2O','D_N2O',\
               'D_EESC','D_O3t','D_O3s','D_SO4','D_POA','D_BC','D_NO3','D_SOA','D_AERh',\
               'RF_CO2','RF_CH4','RF_H2Os','RF_N2O','RF_halo','RF_O3t','RF_O3s','RF_SO4','RF_POA','RF_BC','RF_NO3',\
               'RF_SOA','RF_cloud','RF_BCsnow','RF_LCC','RF','RF_warm','RF_atm','D_gst','D_sst','D_gyp','D_OHC','YieldTot',\
               'Cumemission','CumN2O','CumCH4','REFF','EFNPK','EFNPKT','REFFCN','BECCSall','CovidFF','REN2O','RECH4']

list_regional = ['ELUC','D_EBB_CO2','D_EBB_CH4','D_EBB_N2O','D_EBB_BC','D_EBB_OC','D_EBB_NOX','D_EBB_SO2','D_EWET','D_lst','D_lyp','NPP_crop','D_npp']

list_land = ['D_AREA','D_cveg','D_eharv','D_csoil1','D_csoil2','D_rh1','D_rh2','Diff','D_efire','D_npp',\
             'bLSNK','b1ELUC','b2ELUC','b1RH1_luc','b2RH1_luc','b1RH2_luc','b2RH2_luc',\
             'b1EFIRE_luc','b2EFIRE_luc','b1EHWP1_luc','b2EHWP1_luc','b1EHWP2_luc','b2EHWP2_luc','b1EHWP3_luc','b2EHWP3_luc']

list_special = ['D_HFC','D_PFC','D_ODS','D_OHSNK_HFC','D_OHSNK_PFC','D_OHSNK_ODS','D_HVSNK_HFC',\
                'D_HVSNK_PFC','D_HVSNK_ODS','D_XSNK_HFC','D_XSNK_PFC','D_XSNK_ODS']

list_drivers = ['EFF','ECH4','EN2O','EBC','ESO2','ENOX','EOC']

list_country = ['YieldCot','YieldMai','YieldMai','YieldMaiF','YieldMaiC','YieldRic','YieldWhe','YieldRest',\
                   'AreaCot','AreaMai','AreaMaiF','AreaMaiC','AreaRub','AreaRic','AreaWhe','AreaRest',\
                   'ProdCot','ProdMai','ProdMaiF','ProdMaiC','ProdRub','ProdRic','ProdWhe','ProdRest','ProdDC_cou']

# create save variables
for simu in ['online']:
    for VAR in list_global:
        exec(VAR+'_'+simu+' = np.zeros([1,ind_final+1],dtype=dty)')
    for VAR in list_regional+list_drivers:
        exec(VAR+'_'+simu+' = np.zeros([1,ind_final+1,nb_regionI],dtype=dty)')
    
    for VAR in list_country:
        exec(VAR+'_'+simu+' = np.zeros([1,ind_final+1,171],dtype=dty)')
        
    for VAR in list_land:
        exec(VAR+'_'+simu+' = np.zeros([1,ind_final+1,nb_regionI,nb_biome],dtype=dty)')
    for VAR in list_special:
        if 'HFC' in VAR:
            exec(VAR+'_'+simu+' = np.zeros([1,ind_final+1,nb_HFC],dtype=dty)')
        elif 'PFC' in VAR:
            exec(VAR+'_'+simu+' = np.zeros([1,ind_final+1,nb_PFC],dtype=dty)')
        elif 'ODS' in VAR:
            exec(VAR+'_'+simu+' = np.zeros([1,ind_final+1,nb_ODS],dtype=dty)')

# RELOAD
execfile('OSCAR-loadD.py')
execfile('OSCAR-loadP.py')
execfile('OSCAR-format.py')
execfile('OSCAR-covid-19-food.py')
# RUN (online)
print 'online'
OUT = OSCAR_lite(var_output=list_global+list_regional+list_land+list_special+list_country)

nvar = -1
for VAR in list_global+list_regional+list_land+list_special+list_country:
    nvar += 1
    exec(VAR+'_online[...] = OUT[nvar][...]')
for VAR in list_drivers:
    exec(VAR+'_online[...] = '+VAR+'[...]')


calories = '2.0Mcal'
# Export data
# CO2.. 
# 'ELUC','LSNK','OSNK','Cumemission','CumN2O','CumCH4'
#   ['AreaCot','AreaMai','AreaRub','AreaRic','AreaWhe','AreaRest','REFF','EFNPK','EFNPKT','REFFCN','ELUC','LSNK','OSNK','Cumemission','CumN2O','CumCH4','BECCSall']
for var in ['REFF','EFNPK','EFNPKT','REFFCN','ELUC','LSNK','OSNK','Cumemission','CumN2O','CumCH4','BECCSall','CovidFF','D_lst','EFF','ECH4','EN2O','REN2O','RECH4','D_CO2']:
    exec(var+' = '+var+'_online[0,...]')
    exec('data_'+var+' =  pd.DataFrame('+var+')')
    exec('data_'+var+".to_excel('results/online/'+scen_EFF+'_'+covidinf[covid]+'_'+ExpSeq+'_'+mod_Cropdemand+'_'+mod_Cropeqs+'_"+var+str(ind_final+1700)+'_Actime'+str(Actime)+'_'+calories+".xlsx',header=None,index=None)") 

# Crops
# Global Area and Production 
for var in ['Cot','Mai','MaiC','MaiF','Rub','Ric','Whe','Rest']:
    exec('Prod'+var+' = Prod'+var+'_online[0,...]')
    exec('Area'+var+' = Area'+var+'_online[0,...]')
    
    exec('Data_CProd'+var+' = pd.DataFrame(Prod'+var+')')
    exec('Data_CArea'+var+' = pd.DataFrame(Area'+var+')')    
    exec('Data_CProd'+var+".to_excel('results/online/'+scen_EFF+'_'+covidinf[covid]+'_'+ExpSeq+'_'+mod_Cropdemand+'_'+mod_Cropeqs+'_Prod"+var+str(ind_final+1700)+'_Actime'+str(Actime)+'_'+calories+".xlsx',header=None,index=None)")
    exec('Data_CArea'+var+".to_excel('results/online/'+scen_EFF+'_'+covidinf[covid]+'_'+ExpSeq+'_'+mod_Cropdemand+'_'+mod_Cropeqs+'_Area"+var+str(ind_final+1700)+'_Actime'+str(Actime)+'_'+calories+".xlsx',header=None,index=None)") 
    
    exec('Data_Prod'+var+' = np.sum(pd.DataFrame(Prod'+var+'),1)')
    exec('Data_Area'+var+' = np.sum(pd.DataFrame(Area'+var+'),1)')    
    exec('Data_Prod'+var+".to_excel('results/online/'+scen_EFF+'_'+covidinf[covid]+'_'+ExpSeq+'_'+mod_Cropdemand+'_'+mod_Cropeqs+'_TotalProd"+var+str(ind_final+1700)+'_Actime'+str(Actime)+'_'+calories+".xlsx',header=None,index=None)")
    exec('Data_Area'+var+".to_excel('results/online/'+scen_EFF+'_'+covidinf[covid]+'_'+ExpSeq+'_'+mod_Cropdemand+'_'+mod_Cropeqs+'_TotalArea"+var+str(ind_final+1700)+'_Actime'+str(Actime)+'_'+calories+".xlsx',header=None,index=None)")   


Area_con = np.zeros([401,171],dtype=dty)
Area_con= AreaCot + AreaMai + AreaRic + AreaWhe + AreaRest + AreaRub
Areacon = pd.DataFrame(Area_con)
exec('Areacon'+".to_excel('results/online/'+scen_EFF+'_'+covidinf[covid]+'_'+ExpSeq+'_'+mod_Cropdemand+'_'+mod_Cropeqs+'_Areacon"+str(ind_final+1700)+'_Actime'+str(Actime)+'_'+calories+".xlsx',header=None,index=None)")

# D_gst
D_gst_Result = pd.DataFrame(np.zeros([ind_final-150+1,2]),columns = ['Year','D_gst'])
D_gst_Result.loc[:,'Year'] = np.arange(1850,ind_final+1701) # ind_final+1701
D_gst_Result.loc[:,'D_gst'] = D_gst_online[:,150:].transpose(1,0)-0.6893
D_gst_Result.to_excel('results/online/'+scen_EFF+'_'+covidinf[covid]+'_'+ExpSeq+'_'+mod_Cropdemand+'_'+mod_Cropeqs+'_Gst'+str(ind_final+1700)+'_Actime'+str(Actime)+'_'+calories+'.xlsx',index=False)

