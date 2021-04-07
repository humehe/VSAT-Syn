import sys, os
import numpy as np
from numpy import mean,median
from progressbar import *
from termcolor import colored

from Fnc_Syn_Dir import *
from Fnc_Syn_Utl import *
from Fnc_Syn_Syn import *
from Fnc_Syn_Mth import *
from Fnc_Syn_Tbl import *
from Fnc_Syn_Spc import *
from Fnc_Syn_Stk import *
from Fnc_Syn_Plt import *


os.system('clear')


###########################################################################################
###############################CREATE & STACK M SLICE N CUBES##############################
###########################################################################################
####################################CREATE & STACK#########################################
##########################################1################################################
##########################################0################################################
print
print colored('Info from table: ' + str(cat_ipt_tbl) + ' ' + str(tbl_format_ipt),'cyan')
print

Cat_Ipt_Tbl   = Table_Read(cat_tbl,tbl_format_ipt)
fits          = Cat_Ipt_Tbl[2]
delta_nu      = Cat_Ipt_Tbl[4]
z             = Cat_Ipt_Tbl[8]  
Lfir          = Cat_Ipt_Tbl[11] 
nu            = Cat_Ipt_Tbl[13] 
vel           = Cat_Ipt_Tbl[14] 
num_obj       = len(Cat_Ipt_Tbl[0])

z_sample_avg  = np.mean(z)
z_sample_med  = np.median(z)
z_sample_1sl  = np.nanpercentile(z, 15.9)
z_sample_1sh  = np.nanpercentile(z, 84.1)
z_sample_2sl  = np.nanpercentile(z, 2.30)
z_sample_2sh  = np.nanpercentile(z, 97.7)
z_sample_3sl  = np.nanpercentile(z, 0.20)
z_sample_3sh  = np.nanpercentile(z, 99.8)
z_sample_p25  = np.nanpercentile(z, 25.0)
z_sample_p75  = np.nanpercentile(z, 75.0)

#Results
img_dir_res   = stk_hme_dir + 'IMAGES/'  + str(channel_width) +'/'
stp_dir_res   = stk_hme_dir + 'STAMPS/'  + str(channel_width) +'/'
tbl_dir_res   = stk_hme_dir + 'TABLES/'  + str(channel_width) +'/'
plt_dir_res   = stk_hme_dir + 'PLOTS/'   + str(channel_width) +'/'
stk_dir_res   = stk_hme_dir + 'STACKS/'  + str(channel_width) +'/'
DIR_CAT_IPT   = [cats_dir]
DIR_SPC_IPT   = [img_dir]
DIR_RES       = [stk_hme_dir,img_dir_res,stp_dir_res,tbl_dir_res,plt_dir_res,stk_dir_res]
Check_directories(cat_ipt_tbl,cat_parent,DIR_RES=DIR_RES)
#######################################0################################################
#Values FIXED values for all cubes
print
if fixed_amp_ns == True:
	print colored('Noise Amplitude Values FIXED values for all cubes','yellow') 
elif fixed_amp_ns == False:
	print colored('Noise Amplitude Values VARY values for all cubes','yellow') 
if fixed_width_str == True:
	print colored('Star width Values FIXED values for all cubes','yellow') 
elif fixed_width_str == False:
	print colored('Star width Values VARY values for all cubes','yellow') 
print
if random_choice_type == 'uniform':
	nse_amp_rnd_fxd_itr = np.random.uniform(n_noise_min,n_noise_max,iterations)
	str_sgm_rnd_fxd_itr = np.random.uniform(sigma_min_1dg,sigma_max_1dg,iterations)
	print
	print colored('Uniform distribution (Fixed between cubes):','yellow')
	print colored('Parameters (noise)        : '+ str(n_noise_min)                       + ', ' +str(n_noise_max)                       + ', ' +str(iterations),'yellow')
	print colored('Parameters (noise-exp)    : '+ str(n_noise_min*np.sqrt(cube_number))  + ', ' +str(n_noise_max*np.sqrt(cube_number))  + ', ' +str(iterations),'yellow')
	print colored('Parameters (str-chn)      : '+ str(sigma_min_1dg)                     + ', ' +str(sigma_max_1dg)                     + ', ' +str(iterations),'yellow')
	print colored('Parameters (str-sgm-km/s) : '+ str(sigma_min_1dg*channel_width)       + ', ' +str(sigma_max_1dg*channel_width)       + ', ' +str(iterations),'yellow')
	print

elif random_choice_type == 'gauss':
	nse_amp_rnd_fxd_itr = np.random.normal(n_noise_ctr,n_noise_sgm,iterations)
	str_sgm_rnd_fxd_itr = np.random.normal(sigma_gas_ctr,sigma_gas_sgm,iterations)
	print
	print colored('Gaussian distribution (Fixed between cubes):','yellow')
	print colored('Parameters (noise)    : '+str(n_noise_ctr)                       + ', ' + str(n_noise_sgm)  + ', ' + str(iterations),'yellow')
	print colored('Parameters (noise-exp): '+str(n_noise_ctr*np.sqrt(cube_number))  + ', ' + str(n_noise_sgm*np.sqrt(cube_number))  + ', ' + str(iterations),'yellow')
	print colored('Parameters (star)     : '+str(sigma_gas_ctr)                     + ', ' + str(sigma_gas_sgm)+ ', ' + str(iterations),'yellow')
	print

########################################1 CREATE N CUBES######################################
for repetition in range(1,iterations+1):
	rep_nse_itr = repetition - 1
	cube_number = cube_number
	c1 =[]
	c2 =[]
	c3 =[]
	c4 =[]
	c5 =[]
	c6 =[]
	c7 =[]
	c8 =[]
	c9 =[]
	c10=[]
	c11=[]
	c12=[]
	c13=[]
	c14=[]
	c15=[]
	c16=[]
	c17=[]

	#Values VARY values for all cubes
	if random_choice_type == 'uniform':
		nse_amp_rnd_fxd_cbe  = np.random.uniform(n_noise_min,n_noise_max,cube_number)
		str_sgm_rnd_fxd_cbe  = np.random.uniform(sigma_min_1dg,sigma_max_1dg,cube_number)
		print
		print colored('Uniform distribution (Varying between cubes):','yellow')
		print colored('Parameters: '+ str(n_noise_min)  + ', ' + str(n_noise_max)  + ', ' + str(cube_number),'yellow')
		print colored('Parameters: '+ str(sigma_min_1dg)+ ', ' + str(sigma_max_1dg)+ ', ' + str(cube_number),'yellow')
		print

	elif random_choice_type == 'gauss':
		nse_amp_rnd_fxd_cbe = np.random.normal(n_noise_ctr,n_noise_sgm,cube_number)
		str_sgm_rnd_fxd_cbe = np.random.normal(sigma_gas_ctr,sigma_gas_sgm,cube_number)
		print
		print colored('Gaussian distribution (Varying between cubes):','yellow')
		print colored('Parameters: '+ str(n_noise_ctr)  + ', ' + str(n_noise_sgm)  + ', ' + str(cube_number),'yellow')
		print colored('Parameters: '+ str(sigma_gas_ctr)+ ', ' + str(sigma_gas_sgm)+ ', ' + str(cube_number),'yellow')
		print

	print colored('Creating Synthetic DataCube','white')
	pb1 = ProgressBar(cube_number)
	pb1.start()
	for individual_datacube in range(1,cube_number+1):
		pb1.update(individual_datacube)
		rep_nse     = individual_datacube - 1

		if fixed_amp_ns == True:
			nse_amp_rnd_cbe    = nse_amp_rnd_fxd_itr[rep_nse_itr]
		elif fixed_amp_ns == False and random_choice_type == 'gauss':
			nse_amp_rnd_cbe    = nse_amp_rnd_fxd_cbe[rep_nse]
			print
			print colored('Gaussian distribution (Varying between cubes):','yellow')
			print colored('Parameters (noise)    : '+ str(n_noise_ctr)  + ', ' + str(n_noise_sgm)  + ', '               + str(cube_number),'yellow')
			print colored('Parameters (noise-exp): '+ str(n_noise_ctr)  + ', ' + str(n_noise_sgm*np.sqrt(cube_number))  + ', ' + str(cube_number),'yellow')
			print colored('Parameters (star)     : '+ str(sigma_gas_ctr)+ ', ' + str(sigma_gas_sgm)+ ', '               + str(cube_number),'yellow')
			print
		elif fixed_amp_ns == False and random_choice_type == 'uniform':
			nse_amp_rnd_cbe    = nse_amp_rnd_fxd_cbe[rep_nse]
			print
			print colored('Uniform distribution (Varying between cubes):','yellow')
			print colored('Parameters (noise)    : '+ str(n_noise_min)  + ', ' + str(n_noise_max)  + ', '               + str(cube_number),'yellow')
			print colored('Parameters (noise-exp): '+ str(n_noise_min)  + ', ' + str(n_noise_max*np.sqrt(cube_number))  + ', ' + str(cube_number),'yellow')
			print colored('Parameters (star)     : '+ str(sigma_min_1dg)+ ', ' + str(sigma_max_1dg)+ ', '               + str(cube_number),'yellow')
			print
		if fixed_width_str == True:
			str_sgm_rnd_cbe    = str_sgm_rnd_fxd_itr[rep_nse_itr]                                   #same for all 27 
		elif fixed_width_str == False and random_choice_type == 'gauss':
			str_sgm_rnd_cbe    = str_sgm_rnd_fxd_itr[rep_nse_itr]                                   #same for all 27 
			print
			print colored('Gaussian distribution (Varying between cubes):','yellow')
			print colored('Parameters (noise)    : '+ str(n_noise_ctr)  + ', ' + str(n_noise_sgm)  + ', '               + str(cube_number),'yellow')
			print colored('Parameters (noise-exp): '+ str(n_noise_ctr)  + ', ' + str(n_noise_sgm*np.sqrt(cube_number))  + ', ' + str(cube_number),'yellow')
			print colored('Parameters (star)     : '+ str(sigma_gas_ctr)+ ', ' + str(sigma_gas_sgm)+ ', '               + str(cube_number),'yellow')
			print
		elif fixed_width_str == False and random_choice_type == 'uniform':
			str_sgm_rnd_cbe    = str_sgm_rnd_fxd_itr[rep_nse_itr]                                   #same for all 27 
			print
			print colored('Uniform distribution (Varying between cubes):','yellow')
			print colored('Parameters (noise)    : '+ str(n_noise_min)  + ', ' + str(n_noise_max)  + ', '               + str(cube_number),'yellow')
			print colored('Parameters (noise-exp): '+ str(n_noise_min)  + ', ' + str(n_noise_max*np.sqrt(cube_number))  + ', ' + str(cube_number),'yellow')
			print colored('Parameters (star)     : '+ str(sigma_min_1dg)+ ', ' + str(sigma_max_1dg)+ ', '               + str(cube_number),'yellow')
			print

		Synthetic_Cube_Output = Create_Synthetic_Cube(nx,ny,nz,sigma_fwhm_2d,
								A_min_csi         = A_min          , A_max_csi        = A_max                           ,
								ofs_min_csi       = ofs_min        , ofs_max_csi      = ofs_max                         ,
								sigmax_min_csi    = sigmax_min     , sigmax_max_csi   = sigmax_max                      ,
								sigmay_min_csi    = sigmay_min     , sigmay_max_csi   = sigmay_max                      ,
								sx_fxd_str        = sigmax_max     , sy_fxd_str       = sigmay_max                      ,
								chn_wth_sze       = channel_width  , theta_csi        = Theta                           ,
								theta_vl_1dg_str  = Theta_val      ,
								shape_csi         = starshape      , amp_star_gauss   = True                            ,
								fixed_width_str   = fixed_width_str, sgm_1d_str_fxd   = str_sgm_rnd_cbe                 ,
								A_min_1dg_str     = A_min_1dg      , A_max_1dg_str    = A_max_1dg                       ,
								sigma_min_1dg_str = sigma_min_1dg  , sigma_max_1dg_str = sigma_max_1dg                  ,
								fixed_amp_ns      = fixed_amp_ns   , amp_nse_type      = 'constant'                     , 
								amp_1d_nse_fxd    = nse_amp_rnd_cbe,
								A_min_1dg_nse     = n_noise_min    , A_max_1dg_nse     = n_noise_max                    ,
								cube_ofn_sfx      = str(repetition)+'-'+str(individual_datacube),
								dst_img_dir       = img_dir_res
								)
		c1.append(Synthetic_Cube_Output[0])
		c2.append(Synthetic_Cube_Output[1])
		c3.append(Synthetic_Cube_Output[2])
		c4.append(Synthetic_Cube_Output[1]*Synthetic_Cube_Output[2])
		c5.append(Synthetic_Cube_Output[3])
		c6.append(Synthetic_Cube_Output[4])
		c7.append(Synthetic_Cube_Output[5])
		c8.append(Synthetic_Cube_Output[6])
		c9.append(Synthetic_Cube_Output[7])
		c10.append(Synthetic_Cube_Output[8])
		c11.append(Synthetic_Cube_Output[9])
		c12.append(Synthetic_Cube_Output[10])
		c13.append(Synthetic_Cube_Output[11])
		c14.append(Synthetic_Cube_Output[12])
		c15.append(Synthetic_Cube_Output[13])
		c16.append(Synthetic_Cube_Output[14])
		c17.append(Synthetic_Cube_Output[15])

 	mct01 = aptbl.Table()
 	mct01['cube_fn']   = c1
 	mct01['src_amp']   = c2
 	mct01['nse_amp']   = c3
 	mct01['snr_amp']   = c4
 	mct01['sig_nmb']   = c5
 	mct01['sig_vel']   = c6
 	mct01['sig_fwh']   = c7
 	mct01['src_ofs']   = c8
 	mct01['x_ctr_2D']  = c9
 	mct01['y_ctr_2D']  = c10
 	mct01['amp_2D']    = c11
 	mct01['amp_max_2D']= c12
 	mct01['sgm_x_2D']  = c13
 	mct01['sgm_y_2D']  = c14
 	mct01['theta_2D']  = c15
 	mct01['offset_2D'] = c16
 	mct01['TN_2D']     = c17
 	TABLESTATNAME_1    = tbl_dir_res + 'Synthetic_Cubes-'+str(cube_number)+'-'+str(repetition)+'.dat'
 	TABLESTATNAME_2    = tbl_dir_res + 'Synthetic_Cubes-'+str(cube_number)+'-'+str(repetition)+'.csv'
 	mct01.write(TABLESTATNAME_1, format='ascii.fixed_width_two_line', overwrite = True)
 	mct01.write(TABLESTATNAME_2, format=tbl_format_opt, overwrite = True)
 	print
 	print colored('MC Statistics table: '+TABLESTATNAME_1,'green')
 	print colored('MC Statistics table: '+TABLESTATNAME_2,'green')
 	#######################################1 CREATE N CUBES######################################

 	############################################2################################################
 	##########################################STACKS#############################################
 	sbsms         = 'RDS' 
 	sbsmn         = 0
 	cat_tbl       = cat_dir + 'CII_Sources_HATLAS-' + line + '-' + str(sbsms) + '-' +str(sbsmn) + tbl_ext_ipt
 	cat_tbl_stk   = TABLESTATNAME_2
 	cw            = [250]
 	print colored('Info from table: ' + str(cat_tbl_stk) + ' ' + str(tbl_format_ipt),'magenta')
 	print


 	fits = Table_Read_Syn_Singl_It(cat_tbl_stk,format_tbl=tbl_format_ipt)[1]

	nchan          = (2*subcube_width /channel_width)+1
	slice_nmb      = int(np.ceil(((nchan-1)/2)))#-1 #19 
	fits_new       = fits
	print colored('Reading files as : '+str(fits[0]),'yellow')
	cubetoread     = [img_dir_res + str(file) for file in fits_new]
	[Cube_Mask_Nan(img)     for img in cubetoread]
	print (colored('Reading files as : '+str(cubetoread[0]),'yellow'))
	weights        = np.arange(0,len(fits),1)
	stk_ofn_prfx   = (cat_tbl_stk.split('.',1)[0].rsplit('/',1)[1]) + '-' + str(sbsms) + '-' +str(sbsmn) #+ '-' +str(index)
	Stack_Res      = Cube_Stack(cubetoread,stk_ofn_prfx,weights,
								sig_clp     = False,sufix=channel_width,freq_obs_f=restframe_frequency,
								stack_lite  = stack_light,
								cp_bs_hdrs  = False,
								stt_var     = True,
								spc_wdt_dir = channel_width,
								stt_mst_tbl = Cat_Ipt_Tbl      , stt_hdr='RDS_B',
								stt_syn     = True         , stt_syn_tbl = cat_tbl_stk)
	#############################Add Headers to Stack Results##############################
	name = cat_parent + '-' + str(sbsmn)
	name = cat_parent + '-' + str(sbsms) + '-' +str(sbsmn)
	name = stk_ofn_prfx
	bs_func     = ''
	sufix       = channel_width
	spc_dir_dst = stk_dir_res

	if repetition >1:
		print
 		print colored('Deleting input files for stacking!','yellow')
 		[os.system('rm ' + str(img)) for img in cubetoread]
 		print colored("\n".join([str(img) for img in cubetoread]),'yellow')
 		print
 	else:
 		pass
	###########################################2################################################
	#########################################STACKS#############################################
###############################CREATE & STACK M SLICE N CUBES##############################
###########################################################################################



