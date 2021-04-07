import sys, os
import numpy as np
from numpy import mean,median
from progressbar import *
from termcolor import colored
from astropy import constants as apct
from astropy import units as u
from itertools import product as itlpd

from astropy.io.fits import getdata as apgtdt
from astropy.wcs import WCS as apwcs
from astropy import cosmology as apcosmo

from spectral_cube import SpectralCube as scspc

from Fnc_Syn_Utl import *

import platform
py_ver = (platform.python_version_tuple())
py_ver_ma = py_ver[0]
py_ver_mc = py_ver[1]
py_ver_mx = py_ver[2]
print
print (colored('Python version: ' + str(py_ver_ma) + '.' + str(py_ver_mc) +'.' +  str(py_ver_mx),'yellow'))
print


home = os.path.expanduser("~") + '/Desktop/Example-VSAT-Syn/'
line = '13CO' 

def fwhm2sigma(fwhm):
	return fwhm/(2*np.sqrt(2*np.log(2)))
def sigma2fwhm(sigma):
	return sigma*(2*np.sqrt(2*np.log(2)))

#Catalogue
cat_parent               =   'CII_HATLAS'
CAT_PARENT               =   cat_parent.upper()

#Image Directories
cats_dir                 =   home     + '/Catalogues/'
cat_dir                  =   cats_dir + CAT_PARENT + '/'
dts_dir                  =   home     + line +'DataSets/'             	 #DataSet_directory
ext_dir                  =   home     + line +'Extracted-Im/'         	 #Extracted_directory
fts_dir                  =   ext_dir  + 'FITS/'                       	 #Fits_directory
img_dir                  =   ext_dir  + 'IMAGES/'                     	 #Images_directory

#Frequecies
subcube_width            =   2000                              		  	 #kms-1
channel_width            =   250                             		  	 #kms-1

#Input Table
sbsmn                    =   0
sbsms                    =   'RDS_B'						 			#STM,SFR,LCO,sSF,MH2,SFE,SDG,SDS,TDT,RDS
cat_ipt_tbl              =   cat_dir + 'CII_Sources_HATLAS-' + line + '-' + str(sbsms) + '-' +str(sbsmn) 
unq_tbl_ipt              =   'yes'
head_un_ipt              =   'ident'									#'spec1d'

#Simulation Parameters
cube_number              = 5#                                          #Number of cubes
iterations               = 10                                         #Number of iterations

#Image size
nx                       = 256                                         #pixel x physical dimension
ny                       = 256                                         #pixel y physical dimension
nz                       = 17                                          #pixel z spectral/velocity axis
nchan                    = (2*subcube_width / channel_width) + 1       #pixel z spectral/velocity axis
scale                    = 0.5                                         #0.5arcsec=1pixel

#Noise parameters
amp_noise                = 1                                           #Noise Amplitude

#Spectral Gaussian
fixed_amp_ns             = True                                        #Varying/Fixed among individual cubes SNR
fixed_width_str          = True                                        #Varying/Fixed among individual cubes width
random_choice_type       = 'uniform'                                   #uniform,gauss
A_min_1dg                = 1                                           #A min for gaussian star
A_max_1dg                = 1                                           #A max for gaussian star
ofs_min_1dg              = 0                                           #offset min for gaussian star
ofs_max_1dg              = 0                                           #offset max for gaussian star
sigma_min_1dg            = fwhm2sigma(0.20)                            #sigma x min for velocity width star #0.20:  50.0   21.23 | 0.48:  120.0   50.95 | 2.00:  500.0  212.33
sigma_max_1dg            = fwhm2sigma(2.00)                            #sigma x max for velocity width star #2.36: 590.0  250.54 | 3.10:  775.0  329.00 | 4.71: 1177.5  500.03 | 9.42: 2355.0 1000.07 
n_noise_min              = 1#92#1#3#0.5773502692                       #min n times rms noise 
n_noise_max              = 100#102#100#2.3094010768                    #max n times rms noise 
sigma_gas_ctr            = fwhm2sigma(1)                               #gauss center gaussian distribution star
sigma_gas_sgm            = fwhm2sigma(0.20)                            #gauss sigma  gaussian distribution star
n_noise_ctr              = 1.5396007178                                #gauss center gaussian distribution rms noise
n_noise_sgm              = 1.5396007178                                #gauss sigma  gaussian distribution rms noise
noise_idp_slices         = False                                       #Independent noise for slices

#Source parameters
n                        = 1                                           #number of stars
A_min                    = 1                                           #A min for gaussian star
A_max                    = 1                                           #A max for gaussian star
Theta                    = 'value'                                     #Theta (random)/value (deg)
Theta_val                = 0                                           #Theta (random)/value (deg)
ofs_min                  = 0                                           #offset min for gaussian star
ofs_max                  = 1                                           #offset max for gaussian star
sigmax_min               = 1                                           #sigma x min for gaussian star
sigmax_max               = fwhm2sigma(5/scale)#5                       #sigma x max for gaussian star
sigmay_min               = 1                                           #sigma y min for gaussian star
sigmay_max               = fwhm2sigma(5/scale)#5                       #sigma y max for gaussian star
starshape                = 'circular'                                  #circular/ellipse
tbl_strs_prp             = False                                       #create table
displ_image              = False                                       #display image

#PSF 2D
fwhm_2d                  = 5/scale                                     #fwhm  (pixels)
sigma_fwhm_2d            = fwhm2sigma(fwhm_2d)                         #sigma (pixels)

#Stacking
stack_light     		 = 	 True 							 			#True: SUM MED AVG False: Histograms and pct (1,2,3-sigma)stacks

#Sigma-Clip
sigma_clipping           =   True                            			# Sigma clipping
sigma_cut                =   0                               			# sigma cut
sigma_cen_fct            =   mean                            			# median, mean
sigma_msk_fill_val       =   np.nan                          			# np.nan, value

#Fit
tbl_out_fit              = 'no'             							#create table
displ_fit                = 'yes'             							#Display fit

#tables
tbl_format_ipt           =   'csv'                           			#ascii,csv,fits,ascii.fixed_width_two_line
tbl_format_opt           =   'csv'                           			#ascii,csv,fits,ascii.fixed_width_two_line

#Results
stk_hme_dir              = home + 'Stack_Results-SyntheticCubes-00/'
img_dir_res              = stk_hme_dir + 'IMAGES/' 
stp_dir_res              = stk_hme_dir + 'STAMPS/' 
tbl_dir_res              = stk_hme_dir + 'TABLES/' 
plt_dir_res              = stk_hme_dir + 'PLOTS/' 
stk_dir_res              = stk_hme_dir + 'STACKS/'

#Output  Tables
ind_tbl                  =   'yes'
grl_tbl                  =   'yes'

grl_tbl_nmB              =   'Prs_Bkg_'+ CAT_PARENT 
grl_tbl_nmF              =   'Prs_Frg_'+ CAT_PARENT 

unq_tbl_opt              =   'yes'
hed_un_opt_F             =   'id_F'                                #header of column for uniqueness
hed_un_opt_B             =   'id_B'                                #header of column for uniqueness
grl_tbl_nmB_U            =   'Prs_Bkg_'+ CAT_PARENT +'_U'
grl_tbl_nmF_U            =   'Prs_Frg_'+ CAT_PARENT +'_U'

if line == '13CO':
	restframe_frequency      =   110.20137E9           
elif line == '12CO':
	restframe_frequency      =   115.271208E9
elif line == '18CO':
	restframe_frequency      =   109.78217340E9

#Filenames for Output Images
ifn_raw             = img_dir_res + 'raw.fits'                          #raw image   
ifn_noise           = img_dir_res + 'noise.fits'                        #noise image     
ifn_noise_n         = img_dir_res + 'noise_n.fits'                      #normalized noise image       
ifn_stars           = img_dir_res + 'stars.fits'                        #pure stars image     
ifn_noise_psf       = img_dir_res + 'noise_psf.fits'                    #stars + norm noise + psf image               
ifn_noise_psf_stars = img_dir_res + 'noise_psf_stars.fits'              #stars + norm noise image           

#Filenames for Output Tables 
output_table_obj    = 'Objectlist_I_MC.dat'                             #output table filename
output_table_fit    = 'FittingRes_I_MC.dat'                             #output table filename

#Output table file name
output_table_fit_MC   = 'FittingRes_MC_'+str(n)+'X'+str(iterations)+'_fwhm_'+str(fwhm_2d)+'_noise_'+str(n_noise_min)+'-'+str(n_noise_max)+'_'+str(starshape)+'.dat'   #output table filename MonteCarlo
output_table_fit_MC_E = 'FittingRes_MC_'+str(n)+'X'+str(iterations)+'_fwhm_'+str(fwhm_2d)+'_noise_'+str(n_noise_min)+'-'+str(n_noise_max)+'_'+str(starshape)+'_ERR.dat' #output table filename MonteCarlo ERROR

#############################################################################################################################
DIR_CAT_IPT = [cats_dir]
DIR_SPC_IPT = [img_dir]
DIR_RES     = [stk_hme_dir,img_dir_res,stp_dir_res,tbl_dir_res,plt_dir_res,stk_dir_res]


if tbl_format_ipt == 'ascii' or tbl_format_ipt == 'ascii.fixed_width_two_line':
	tbl_ext_ipt = '.dat'
elif tbl_format_ipt == 'csv':	
	tbl_ext_ipt = '.csv'
elif tbl_format_ipt == 'fits':	
	tbl_ext_ipt = '.fits'

if tbl_format_opt == 'ascii' or tbl_format_opt == 'ascii.fixed_width_two_line':
	tbl_ext_opt = '.dat'
elif tbl_format_opt == 'csv':	
	tbl_ext_opt = '.csv'
elif tbl_format_opt == 'fits':	
	tbl_ext_opt = '.fits'

cat_tbl    = cat_ipt_tbl + tbl_ext_ipt
cat_tbl_U  = cat_ipt_tbl + tbl_ext_opt

#op_tbl_B   = bkg_dir_res + grl_tbl_nmB   + '_' + str(rad_sep[0]) + '-' + str(rad_sep[-1]) + tbl_ext_opt      #header of column for uniqueness
#op_tbl_F   = frg_dir_res + grl_tbl_nmF   + '_' + str(rad_sep[0]) + '-' + str(rad_sep[-1]) + tbl_ext_opt      #header of column for uniqueness
#op_tbl_B_U = bkg_dir_res + grl_tbl_nmB   + '_' + str(rad_sep[0]) + '-' + str(rad_sep[-1]) + '_U' + tbl_ext_opt      #header of column for uniqueness
#op_tbl_F_U = frg_dir_res + grl_tbl_nmF   + '_' + str(rad_sep[0]) + '-' + str(rad_sep[-1]) + '_U' + tbl_ext_opt      #header of column for uniqueness

sofl = apct.c
sofl = sofl.to(u.km/u.s)

cosmo_H0 = 70
cosmo_omegaM = 0.3
cosmo = apcosmo.FlatLambdaCDM(cosmo_H0,cosmo_omegaM)
#############################################################################################################################	
def Check_directories(cat_tbl_chk,cat_chk,*args, **kwargs):
	DIR_RES = kwargs.get('DIR_RES',[
				stk_hme_dir,img_dir_res,stp_dir_res,tbl_dir_res,plt_dir_res,stk_dir_res])
	print
	print ('Checking input directories of the catalogue : ',cat_chk)
	if os.path.exists(str(cat_tbl_chk)+str(tbl_ext_ipt))==True :#and os.path.exists(DIR_SPC_IPT[0])==True:
		print
		print ('Catalogue table exists             : ', str(cat_tbl_chk)+str(tbl_ext_ipt))
		print ('Spectra directories exists         : ', str(DIR_SPC_IPT[0]))
		#print ('Spectra directories exists (noise) : ', str(DIR_SPC_IPT[1]))
		print
		print ('Checking Result Directories.')
		print

		for tree in DIR_RES:
			if os.path.isdir(tree)==True:
				pass
				print ('Directory exists: ', tree)
			elif os.path.isdir(tree)==False:
				print ('Directory does not exist, creating it: ', tree)
				os.makedirs(tree)
	elif os.path.exists(str(cat_tbl_chk)+str(tbl_ext_ipt))==False :#or os.path.exists(DIR_SPC_IPT[0])==False:
		print
		print ('Some of the directories does not exist.')
		print ('Check input directories. ')
		print (str(cat_tbl_chk)+str(tbl_ext_ipt))
		print( DIR_SPC_IPT[0])
#############################################################################################################################
Check_directories(cat_ipt_tbl,cat_parent,DIR_RES=DIR_RES)
