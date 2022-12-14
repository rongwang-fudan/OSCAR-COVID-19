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


import os
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin,fsolve
from scipy.special import gammainc
from matplotlib.font_manager import FontProperties


##################################################
#   1. OPTIONS
##################################################

p = 6                                   # time step (p-th year; must be >4)
fC = 1                                  # carbon feedback (0 or 1)
fT = 1                                  # climate feedback (0 or 1)
dty = np.float32                        # precision (float32 or float64)

PI_1750 = True                         # if False simulates the 1700-1750 period
ind_final = 400                         # ending year of run (+1700)
ind_attrib = 0                          # starting year of attribution

attrib_DRIVERS = 'producers'            # [deprecated]
attrib_FEEDBACKS = 'emitters'           # [deprecated]
attrib_ELUCdelta = 'causal'             # [deprecated]
attrib_ELUCampli = 'causal'             # [deprecated]

mod_regionI = 'Houghton'                # SRES4 | SRES11 | RECCAP* | Raupach* | Houghton | IMACLIM | Kyoto | RCP5 | RCP10*
mod_regionJ = 'RCP5'                    # SRES4 | SRES11 | RECCAP* | Raupach* | Houghton | IMACLIM | Kyoto | RCP5 | RCP10*
mod_sector = ''                         # '' | Time | TimeRCP
mod_kindFF = 'one'                      # one | CDIAC
mod_kindLUC = 'one'                     # one | all
mod_kindGHG = 'one'                     # one | RCP
mod_kindCHI = 'one'                     # one | all
mod_kindAER = 'one'                     # one | all
mod_kindRF = 'one'                      # one | two | all
mod_kindGE = ''                         # '' | PUP

mod_biomeSHR = 'w/GRA'                  # SHR | w/GRA | w/FOR
mod_biomeURB = 'w/DES'                  # URB | w/DES

data_EFF = 'CDIAC'                      # CDIAC | EDGAR
data_LULCC = 'LUH1'                     # LUH1
data_ECH4 = 'EDGAR'                     # EDGAR | ACCMIP | EPA
data_EN2O = 'EDGAR'                     # EDGAR | EPA
data_Ehalo = 'EDGAR'                    # EDGAR
data_ENOX = 'EDGAR'                     # EDGAR | ACCMIP
data_ECO = 'EDGAR'                      # EDGAR | ACCMIP
data_EVOC = 'EDGAR'                     # EDGAR | ACCMIP
data_ESO2 = 'EDGAR'                     # EDGAR | ACCMIP
data_ENH3 = 'EDGAR'                     # EDGAR | ACCMIP
data_EOC = 'ACCMIP'                     # ACCMIP
data_EBC = 'ACCMIP'                     # ACCMIP
data_RFant = 'IPCC-AR5'                 # '' | IPCC-AR5
data_RFnat = 'IPCC-AR5'                 # '' | IPCC-AR5

mod_DATAscen = 'trends'                 # raw | offset | smoothX (X in yr) | trends

IfScen = 'SSP7.0'
scen_EFF = IfScen                       # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_LULCC = 'stop'                     # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ECH4 = IfScen                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_EN2O = IfScen                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_Ehalo = 'stop'                     # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ENOX = IfScen                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ECO = IfScen                       # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_EVOC = IfScen                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ESO2 = IfScen                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ENH3 = IfScen                      # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_EOC = IfScen                       # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_EBC = IfScen                       # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_RFant = 'stop'                     # stop | cst
scen_RFnat = 'stop'                     # stop | cst
scen_Other = 'RCP6.0'
                        
mod_OSNKstruct = 'HILDA'                # HILDA | BD-model | 2D-model | 3D-model
mod_OSNKchem = 'CO2SysPower'            # CO2SysPade | CO2SysPower
mod_OSNKtrans = 'mean-CMIP5'            # mean-CMIP5 | CESM1-BGC | IPSL-CM5A-LR | MPI-ESM-LR

mod_LSNKnpp = 'log'                     # log | hyp
mod_LSNKrho = 'exp'                     # exp | gauss
mod_LSNKpreind = 'mean-TRENDYv2'        # mean-TRENDYv2 | CLM-45 | JSBACH | JULES | LPJ | LPJ-GUESS | LPX-Bern | OCN | ORCHIDEE | VISIT
mod_LSNKtrans = 'mean-CMIP5'            # mean-CMIP5 | BCC-CSM-11 | CESM1-BGC | CanESM2 | HadGEM2-ES | IPSL-CM5A-LR | MPI-ESM-LR | NorESM1-ME
mod_LSNKcover = 'mean-TRENDYv2'         # ESA-CCI | MODIS | Ramankutty1999 | Levavasseur2012 | mean-TRENDYv2 | CLM-45 | JSBACH | JULES | LPJ | LPJ-GUESS | LPX-Bern | OCN | ORCHIDEE | VISIT

mod_EFIREpreind = 'mean-TRENDYv2'       # '' | mean-TRENDYv2 | CLM-45 | JSBACH | LPJ | LPJ-GUESS | ORCHIDEE | VISIT
mod_EFIREtrans = 'mean-CMIP5'           # '' | mean-CMIP5 | CESM1-BGC | IPSL-CM5A-LR | MPI-ESM-LR | NorESM1-ME

mod_ELUCagb = 'mean-TRENDYv2'           # mean-TRENDYv2 | CLM-45 | LPJ-GUESS | ORCHIDEE
mod_EHWPbb = 'high'                     # high | low
mod_EHWPtau = 'Earles2012'              # Houghton2001 | Earles2012
mod_EHWPfct = 'gamma'                   # gamma | lin | exp

mod_OHSNKtau = 'Prather2012'            # Prather2012 | CESM-CAM-superfast | CICERO-OsloCTM2 | CMAM | EMAC | GEOSCCM | GDFL-AM3 | GISS-E2-R | GISS-E2-R-TOMAS | HadGEM2 | LMDzORINCA | MIROC-CHEM | MOCAGE | NCAR-CAM-35 | STOC-HadAM3 | TM5 | UM-CAM
mod_OHSNKfct = 'lin'                    # lin | log
mod_OHSNKtrans = 'Holmes2013'           # mean-OxComp | Holmes2013 | GEOS-Chem | Oslo-CTM3 | UCI-CTM

mod_EWETpreind = 'mean-WETCHIMP'        # '' | mean-WETCHIMP | CLM-4Me | DLEM | IAP-RAS | LPJ-Bern | LPJ-WSL | ORCHIDEE | SDGVM
mod_AWETtrans = 'mean-WETCHIMP'         # '' | mean-WETCHIMP | CLM-4Me | DLEM | LPJ-Bern | ORCHIDEE | SDGVM | UVic-ESCM

mod_HVSNKtau = 'Prather2015'            # Prather2015 | GMI | GEOSCCM | G2d-M | G2d | Oslo-c29 | Oslo-c36 | UCI-c29 | UCI-c36
mod_HVSNKtrans = 'Prather2015'          # Prather2012 | Prather2015 | G2d | Oslo-c29 | UCI-c29
mod_HVSNKcirc = 'mean-CCMVal2'          # mean-CCMVal2 | AMTRAC | CAM-35 | CMAM | Niwa-SOCOL | SOCOL | ULAQ | UMUKCA-UCAM

mod_O3Tregsat = 'mean-HTAP'             # '' | mean-HTAP | CAMCHEM | FRSGCUCI | GISS-modelE | GMI | INCA | LLNL-IMPACT | MOZART-GFDL | MOZECH | STOC-HadAM3 | TM5-JRC | UM-CAM
mod_O3Temis = 'mean-ACCMIP'             # mean-OxComp | mean-ACCMIP | CICERO-OsloCTM2 | NCAR-CAM-35 | STOC-HadAM3 | UM-CAM
mod_O3Tclim = 'mean-ACCMIP'             # '' | mean-ACCMIP | CESM-CAM-superfast | GFDL-AM3 | GISS-E2-R | MIROC-CHEM | MOCAGE | NCAR-CAM-35 | STOC-HadAM3 | UM-CAM
mod_O3Tradeff = 'IPCC-AR5'              # IPCC-AR5 | IPCC-AR4 | mean-ACCMIP | CESM-CAM-superfast | CICERO-OsloCTM2 | CMAM | EMAC | GEOSCCM | GFDL-AM3 | GISS-E2-R | HadGEM2 | LMDzORINCA | MIROC-CHEM | MOCAGE | NCAR-CAM-35 | STOC-HadAM3 | UM-CAM | TM5

mod_O3Sfracrel = 'Newman2006'           # Newman2006 | Laube2013-HL | Laube2013-ML
mod_O3Strans = 'mean-CCMVal2'           # mean-CCMVal2 | AMTRAC | CCSR-NIES | CMAM | CNRM-ACM | LMDZrepro | MRI | Niwa-SOCOL | SOCOL | ULAQ | UMSLIMCAT | UMUKCA-UCAM
mod_O3Snitrous = 'Daniel2010-sat'       # '' | Daniel2010-sat | Daniel2010-lin
mod_O3Sradeff = 'IPCC-AR4'              # IPCC-AR4 | mean-ACCENT | ULAQ | DLR-E39C | NCAR-MACCM | CHASER

mod_SO4regsat = 'mean-HTAP'             # '' | mean-HTAP | CAMCHEM | GISS-PUCCINI | GMI | GOCART | INCA2 | LLNL-IMPACT | SPRINTARS
mod_SO4load = 'mean-ACCMIP'             # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_SO4radeff = 'mean-AeroCom2'         # mean-AeroCom2 | BCC | CAM4-Oslo | CAM-51 | GEOS-CHEM | GISS-MATRIX | GISS-modelE | GMI | GOCART | HadGEM2 | IMPACT-Umich | INCA | MPIHAM | NCAR-CAM-35 | OsloCTM2 | SPRINTARS

mod_POAconv = 'default'                 # default | GFDL | CSIRO
mod_POAregsat = 'mean-HTAP'             # '' | mean-HTAP | CAMCHEM | GISS-PUCCINI | GMI | GOCART | INCA2 | LLNL-IMPACT | SPRINTARS
mod_POAload = 'mean-ACCMIP'             # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_POAradeff = 'mean-AeroCom2'         # mean-AeroCom2 | BCC | CAM4-Oslo | CAM-51 | GEOS-CHEM | GISS-MATRIX | GISS-modelE | GMI | GOCART | HadGEM2 | IMPACT-Umich | INCA | MPIHAM | NCAR-CAM-35 | OsloCTM2 | SPRINTARS

mod_BCregsat = 'mean-HTAP'              # '' | mean-HTAP | CAMCHEM | GISS-PUCCINI | GMI | GOCART | INCA2 | LLNL-IMPACT | SPRINTARS
mod_BCload = 'mean-ACCMIP'              # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_BCradeff = 'mean-AeroCom2'          # mean-AeroCom2 | BCC | CAM4-Oslo | CAM-51 | GEOS-CHEM | GISS-MATRIX | GISS-modelE | GMI | GOCART | HadGEM2 | IMPACT-Umich | INCA | MPIHAM | NCAR-CAM-35 | OsloCTM2 | SPRINTARS
mod_BCadjust = 'Boucher2013'            # Boucher2013 | CSIRO | GISS | HadGEM2 | ECHAM5 | ECMWF

mod_NO3load = 'Bellouin2011'            # Bellouin2011 | Hauglustaine2014
mod_NO3radeff = 'mean-AeroCom2'         # mean-AeroCom2 | GEOS-CHEM | GISS-MATRIX | GMI | HadGEM2 | IMPACT-Umich | INCA | NCAR-CAM-35 | OsloCTM2

mod_SOAload = 'mean-ACCMIP'             # '' | mean-ACCMIP | GFDL-AM3 | GISS-E2-R
mod_SOAradeff = 'mean-AeroCom2'         # mean-oCom2 | CAM-51 | GEOS-CHEM | IMPACT-Umich | MPIHAM | OsloCTM2

mod_DUSTload = 'mean-ACCMIP'            # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_DUSTradeff = ''                     # [no value for now]

mod_SALTload = 'mean-ACCMIP'            # mean-ACCMIP | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_SALTradeff = ''                     # [no value for now]

mod_CLOUDsolub = 'Lamarque2011'         # Hansen2005 | Lamarque2011
mod_CLOUDerf = 'mean-ACCMIP'            # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | HadGEM2 | LMDzORINCA | MIROC-CHEM | NCAR-CAM-51
mod_CLOUDpreind = 'median'              # low | median | high

mod_ALBBCreg = 'Reddy2007'              # Reddy2007
mod_ALBBCrf = 'mean-ACCMIP'             # mean-ACCMIP | CICERO-OsloCTM2 | GFDL-AM3 | GISS-E2-R | GISS-E2-R-TOMAS | HadGEM2 | MIROC-CHEM | NCAR-CAM-35 | NCAR-CAM-51
mod_ALBBCwarm = 'median'                # low | median | high 

mod_ALBLCflux = 'CERES'                 # CERES | GEWEX | MERRA
mod_ALBLCalb = 'GlobAlbedo'             # GlobAlbedo | MODIS
mod_ALBLCcover = 'ESA-CCI'              # ESA-CCI | MODIS
mod_ALBLCwarm = 'Jones2013'             # Hansen2005 | Davin2007 | Davin2010 | Jones2013

mod_TEMPresp = 'mean-CMIP5'             # mean-CMIP5 | ACCESS-10 | ACCESS-13 | BCC-CSM-11 | BCC-CSM-11m | CanESM2 | CCSM4 | CNRM-CM5 | CNRM-CM5-2 | CSIRO-Mk360 | GFDL-CM3 | GFDL-ESM2G | GFDL-ESM2M | GISS-E2-H | GISS-E2-R | HadGEM2-ES | IPSL-CM5A-LR | IPSL-CM5A-MR | IPSL-CM5B-LR | MIROC5 | MIROC-ESM | MPI-ESM-LR | MPI-ESM-MR | MPI-ESM-P | MRI-CGCM3 | NorESM1-M
mod_TEMPpattern = 'hist&RCPs'           # 4xCO2 | hist&RCPs

mod_PRECresp = 'mean-CMIP5'             # mean-CMIP5 | ACCESS-10 | ACCESS-13 | BCC-CSM-11 | BCC-CSM-11m | CanESM2 | CCSM4 | CNRM-CM5 | CNRM-CM5-2 | CSIRO-Mk360 | GFDL-CM3 | GFDL-ESM2G | GFDL-ESM2M | GISS-E2-H | GISS-E2-R | HadGEM2-ES | IPSL-CM5A-LR | IPSL-CM5A-MR | IPSL-CM5B-LR | MIROC5 | MIROC-ESM | MPI-ESM-LR | MPI-ESM-MR | MPI-ESM-P | MRI-CGCM3 | NorESM1-M
mod_PRECradfact = 'Andrews2010'         # Andrews2010 | Kvalevag2013
mod_PRECpattern = 'hist&RCPs'           # 4xCO2 | hist&RCPs

mod_ACIDsurf = 'Bernie2010'             # Tans2009 | Bernie2010

mod_SLR = ''                            # [no value for now]

##################################################
#   2. COVID-19 and Food influneces
##################################################
# ------------------------------------
#   2.1 crops parameters
# ------------------------------------  
# Choose input files
Filein = 'Crop_COVID_19'
# choose Yc-Tc types
TYform = 'Qua'  # Qua / Log / LogCub / Log2.5
ty = 0  
# Quadratic function
# For quadratic equation: 
# Other Crops
Para_A = [-0.0067]
Para_B = [0.1214]
Para_C = [2.077]
# Maize
Mai_A = [-0.0104] 
Mai_B = [0.4011]
Mai_C = [-2.1896]
#choose Yc-CO2 types
Tc = 0
TCO2_A = [-0.000002382]
TCO2_B = [0.003335]
TCO2_C = [0.124615]
# Lastest FAO data 
Fy = 2020
# To predict npp and yield
Ys = 2021
# From year YsAr to predict area                            
YsAr = 2021                         
## Cm Cn, N2O concentration D_N2O
Cm = 2
Cn = 0
## N2O function, Pf - peak value; Tf - final time to reach peak value
Pf = 150
Tf = 350

# --------------------------------------------
#   2.2 COVID-19 influence carbon emissions
# --------------------------------------------
# COVID-19 medical demands scanary
# Lockdonw influce
effcovid = 31
# GtC/yr for COVID-19 to 2030
RedEFF1 = [-0.71, -0.24,-0.22,-0.19,-0.16,-0.13,-0.11,-0.08,-0.05,-0.03,0]
# GtC/yr for COVID-19 to 2025 
RedEFF2 = [-0.71, -0.24,-0.18,-0.12,-0.06,0]
# GtC/yr for COVID-19 to 2050 
RedEFF3 = [-0.71, -0.69,-0.66,-0.64,-0.62,-0.59,-0.57,-0.54,-0.52,-0.50,-0.47,-0.45,-0.43,-0.40,-0.38,\
           -0.36,-0.33,-0.31,-0.28,-0.26,-0.24,-0.21,-0.19,-0.17,-0.14,-0.12,-0.09,-0.07,-0.05,-0.02,0] 
# GtC/yr for COVID-19 to 2050
RedEFF4 = [-0.71, -0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,\
           -0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,-0.71,0] 
# Without Reduction for COVID-19
RedEFF5 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0]
# GtC/yr for COVID-19 to 2022 
RedEFF6 = [-0.71,-0.35,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0] 

# --------------------------------------------
#   2.3 Land cover change scenarios
# --------------------------------------------
# COVID-19-Food-Dynamic| COVID-19-Food-Dynamic-Constant | Population-Driven | Constant
mod_Cropdemand = 'COVID-19-Food-Dynamic-Constant'
# COVID-19 lockdown pathway
if (mod_Cropdemand == 'Population-Driven'):
    RedEFF = RedEFF5      
if (mod_Cropdemand == 'COVID-19-Food-Dynamic-Constant' or mod_Cropdemand == 'COVID-19-Food-Dynamic'):
    RedEFF = RedEFF6

# The consumption of Nonwoven, face mask, hand sanitizer and gloves
# Simple  | Multiple | Multiple_Fixed | SVM
mod_Cropeqs = 'Multiple_Fixed'

if (mod_Cropeqs == 'Simple'):
    SAR_Non_C = [-1.4766]
    SAR_Cot_C = [2.1069]
    SAR_Mai_C = [2.2381]
    SAR_Rub_C = [0.6981]        
    
if (mod_Cropeqs == 'Multiple_Fixed'):
    SAR_Non_C = [-1.4766]
    SAR_Cot_C = [2.0644]
    SAR_Mai_C = [1.2400]
    SAR_Rub_C = [-2.1123]
  
# Land cover transfor mode
# 'For1Mar2', 'Mar1For2'
ExpSeq = 'Mar1For2'         
# FA0:Global Forest Resources Assessment 2020
ForArea = 4060.0 # Mha
# Estimated from Potapov et al.,2021
MarArea = 50.0   # Mha
# COVID-19 influence Yes | No
covid = 1
covidinf = ['nocovid','covid'] 
# Unit emission of fertilizer production and application in agriculture (Xing et al.,2021)
# N, P2O5 and K2O fertilizers, respectively. t CO2-eq/t crop
mod_fertilizers = 'Xing'     # Tian | Xing 
if mod_fertilizers == 'Xing':
    ferdic = {'N':'0.118','P2O5':'0.022','K2O':'0.037'}
    # About 0.42 of the nitrogen added to croplands from Zhang2017
    ENPK = 0.18/0.42 # N, P2O5 and K2O emission(t CO2-eq) for each t crops
if mod_fertilizers == 'Tian':
    ENPK = 3.4+0.8*2*0.71 # Mt N/yr: Tian et al.,2020
# --------------------------------------------
#   2.4 Carbon reduction scenarios
# --------------------------------------------
## Carbon-Neutrality | Net-Zero-Emissions | No-Action | BECCS
mod_Carbonreduction = 'BECCS'
# Peak carbon emissions time
Actime = 2030

###################################################
##################################################
#   3. OSCAR
##################################################

if (scen_Other != ''):
    scen_LULCC = scen_Ehalo = scen_RFant = scen_RFnat = scen_Other
    if (scen_RFant[:3] == 'RCP'):
        scen_RFant = 'cst'
    if (scen_RFnat[:3] == 'RCP'):
        scen_RFnat = 'cst'

execfile('OSCAR-loadD.py')
execfile('OSCAR-loadP.py')
execfile('OSCAR-format.py')
execfile('OSCAR-covid-19-food.py')


