from astropy.io import fits as apfts
from random import uniform as rdunf
from random import choice as rdchc
from random import random as rdrnd
from astropy import convolution as apcvl
from astropy.io import fits as apfts

from Fnc_Syn_Dir import *
from Fnc_Syn_Mth import *
from Fnc_Syn_Tbl import *

####Fnc_Syn_Syn####
def create_raw_image(n_pix_x,n_pix_y):
	raw_image_zeros   = np.zeros((n_pix_y,n_pix_x))
	return raw_image_zeros

def add_stars2image(image_in,image_star,*args,**kwargs):
	return image_in + image_star

def star_gen_prp(nx_sg,ny_sg,A_min_sg,A_max_sg,ofs_min_sg,ofs_max_sg,sigmax_min_sg,sigmax_max_sg,sigmay_min_sg,sigmay_max_sg,shape_sg,theta_sg,*args,**kwargs):
	fixed_position_str = kwargs.get('fixed_position_str',False)
	x_fxd_str          = kwargs.get('x_fxd_str',(nx_sg / 2))
	y_fxd_str          = kwargs.get('y_fxd_str',(ny_sg / 2))

	fixed_size_str     = kwargs.get('fixed_size_str',False)
	sx_fxd_str         = kwargs.get('sx_fxd_str',10)
	sy_fxd_str         = kwargs.get('sy_fxd_str',10)

	fixed_ampl_str     = kwargs.get('fixed_ampl_str',False)
	am_fxd_str         = kwargs.get('am_fxd_str',1)

	fixed_offset_str   = kwargs.get('fixed_offset_str',False)
	offset_fxd_str     = kwargs.get('offset_fxd_str',0)

	theta_cte_vl       = kwargs.get('theta_cte_vl',0)

	#position and angle
	r       = rdrnd() * nx_sg
	theta_p = rdunf(0., (2. * np.pi))

	# Compute position
	if fixed_position_str == False:
		x        = (nx_sg / 2) + (r * np.cos(theta_p))
		y        = (ny_sg / 2) + (r * np.sin(theta_p))
	elif fixed_position_str == True:
		x        = x_fxd_str
		y        = y_fxd_str

	#Generate random parameters for the 2d gaussian
	#Amplitude
	if fixed_ampl_str == False:
		am = rdunf(A_min_sg,A_max_sg)
	elif fixed_ampl_str == True:
		am = am_fxd_str
	#Offset
	if fixed_offset_str == False:
		offset = rdunf(ofs_min_sg,ofs_max_sg)
	elif fixed_offset_str == True:
		offset = offset_fxd_str
	#SIZE
	if fixed_size_str == False:
		sigma_x  = rdunf(sigmax_min_sg,sigmax_max_sg)
		sigma_y  = rdunf(sigmay_min_sg,sigmay_max_sg)
	elif fixed_size_str == True:
		sigma_x  = sx_fxd_str
		sigma_y  = sy_fxd_str

	#SHAPE
	if shape_sg    == 'ellipse':
		sigma_x  = sigma_x
		sigma_y  = sigma_y
	elif shape_sg  == 'circular':
		sigma_x  = sigma_x
		sigma_y  = sigma_x

	#ORIENTATION
	if theta_sg =='random':
		theta    = rdunf(0., (2. * np.pi))
	elif theta_sg =='value':
		theta = np.deg2rad(theta_cte_vl)

	return x,y,theta,am,offset,sigma_x,sigma_y

def create_star_image(nstrs_csi,nx_csi,ny_csi,A_min_csi,A_max_csi,ofs_min_csi,ofs_max_csi,sigmax_min_csi,sigmax_max_csi,sigmay_min_csi,sigmay_max_csi,shape_csi,theta_csi,*args,**kwargs):
	fixed_position_str = kwargs.get('fixed_position_str',False)
	x_fxd_str          = kwargs.get('x_fxd_str',(nx_csi / 2))
	y_fxd_str          = kwargs.get('y_fxd_str',(ny_csi / 2))

	fixed_size_str     = kwargs.get('fixed_size_str',False)
	sx_fxd_str         = kwargs.get('sx_fxd_str'    ,10)
	sy_fxd_str         = kwargs.get('sy_fxd_str'    ,10)

	fixed_ampl_str     = kwargs.get('fixed_ampl_str',False)
	am_fxd_str         = kwargs.get('am_fxd_str'    ,1)

	fixed_offset_str   = kwargs.get('fixed_offset_str',False)
	offset_fxd_str     = kwargs.get('offset_fxd_str'  ,0)

	wrt_otp_tbl_str    = kwargs.get('wrt_otp_tbl_str',False)
	theta_csi_vl       = kwargs.get('theta_csi_vl',0)

	star_0_cumulative  = create_raw_image(nx,ny)

	X  = []
	Y  = []
	T  = []
	A  = []
	AM = []
	OS = []
	SX = []
	SY = []
	TN = []

	for j in range(nstrs_csi):

		star_0 = create_raw_image(nx,ny)
		star_outpout=star_gen_prp(
			nx_sg              = nx_csi            ,ny_sg          = ny_csi        ,
			A_min_sg           = A_min_csi         ,A_max_sg       = A_max_csi     ,
			ofs_min_sg         = ofs_min_csi       ,ofs_max_sg     = ofs_max_csi   ,
			sigmax_min_sg      = sigmax_min_csi    ,sigmax_max_sg  = sigmax_max_csi,
			sigmay_min_sg      = sigmay_min_csi    ,sigmay_max_sg  = sigmay_max_csi,
			shape_sg           = shape_csi         ,theta_sg       = theta_csi     , theta_cte_vl = theta_csi_vl,
			fixed_position_str = fixed_position_str,x_fxd_str      = x_fxd_str     , y_fxd_str    = y_fxd_str,
			fixed_size_str     = fixed_size_str    ,sx_fxd_str     = sx_fxd_str    , sy_fxd_str   = sy_fxd_str,
			fixed_ampl_str     = fixed_ampl_str    ,am_fxd_str     = am_fxd_str    )

		posx_g   = star_outpout[0]
		posy_g   = star_outpout[1]
		theta_g  = star_outpout[2]
		a_g      = star_outpout[3]
		ofs_g    = 0
		sigmax_g = star_outpout[5]
		sigmay_g = star_outpout[6]
		n_t_n_g  = 0

		count=0

		while posx_g < 0 or posx_g > nx_csi or posy_g < 0 or posy_g > ny_csi:
			star_outpout=star_gen_prp(
				nx_sg              = nx_csi            ,ny_sg          = ny_csi        ,
				A_min_sg           = A_min_csi         ,A_max_sg       = A_max_csi     ,
				ofs_min_sg         = ofs_min_csi       ,ofs_max_sg     = ofs_max_csi   ,
				sigmax_min_sg      = sigmax_min_csi    ,sigmax_max_sg  = sigmax_max_csi,
				sigmay_min_sg      = sigmay_min_csi    ,sigmay_max_sg  = sigmay_max_csi,
				shape_sg           = shape_csi         ,theta_sg       = theta_csi     , theta_cte_vl = theta_csi_vl,
				fixed_position_str = fixed_position_str,x_fxd_str      = x_fxd_str     , y_fxd_str    = y_fxd_str,
				fixed_size_str     = fixed_size_str    ,sx_fxd_str     = sx_fxd_str    , sy_fxd_str   = sy_fxd_str,
				fixed_ampl_str     = fixed_ampl_str    ,am_fxd_str     = am_fxd_str    )


			posx_g   = star_outpout[0]
			posy_g   = star_outpout[1]
			theta_g  = star_outpout[2]
			a_g      = star_outpout[3]
			ofs_g    = 0
			sigmax_g = star_outpout[5]
			sigmay_g = star_outpout[6]
			n_t_n_g  = 0

			count  = count+1
		else: 

			x = np.linspace(0, nx, nx)
			y = np.linspace(0, ny, ny)
			x, y = np.meshgrid(x, y)

			#create data
			star_array        = func_2D_Gaussian_star((x, y), posx_g, posy_g, a_g, sigmax_g, sigmay_g, theta_g, ofs_g)
			am_max_g          = max(star_array)
			star_image        = (star_array.reshape(nx,ny))
			star_0            = add_stars2image(star_0,star_image)
			star_0_cumulative = add_stars2image(star_0,star_0_cumulative)		
	
			X.append(posx_g)
			Y.append(posy_g)
			T.append(theta_g)
			A.append(a_g)
			AM.append(am_max_g)
			OS.append(ofs_g)
			SX.append(sigmax_g)
			SY.append(sigmay_g)
			TN.append(n_t_n_g)

	X  = np.asarray(X[0])
	Y  = np.asarray(Y[0])
	T  = np.asarray(T[0])
	A  = np.asarray(A[0])
	AM = np.asarray(AM[0])
	OS = np.asarray(OS[0])
	SX = np.asarray(SX[0])
	SY = np.asarray(SY[0])
	TN = np.asarray(TN[0])
	if wrt_otp_tbl_str == True:
		# Create Stars data
		r = astropy.table.Table()
		r['X '] = X 
		r['Y '] = Y 
		r['A '] = A 
		r['A_M']= AM 
		r['SX'] = SX
		r['SY'] = SY
		r['T '] = T 
		r['OS'] = OS
		r['xN'] = TN
		r.write('Objectlist_I_MC.csv', format='csv',overwrite=True)	
		r.write(output_table_obj, format='ascii.fixed_width_two_line',overwrite=True)	
		print 'Results containing stars data: ',output_table_obj
	elif wrt_otp_tbl_str == False:
		pass
	return star_0_cumulative,X,Y,A,AM,SX,SY,T,OS,TN

def displ_fits(image,scale,colormap,*args,**kwargs):
    image_data = apfts.getdata(image)
    if scale =='norm':
        plt.imshow(image_data, origin='lower',cmap=colormap)
    elif scale =='log':
        plt.imshow(image_data, origin='lower',cmap=colormap, norm=LogNorm()) #log scale
    plt.colorbar()
    plt.show()


def Create_Synthetic_Cube(x_cube_dim,y_cube_dim,z_cube_dim,krn_conv_noise,*args,**kwargs):
	A_min_csi      = kwargs.get('A_min_csi'     ,A_min)
	A_max_csi      = kwargs.get('A_max_csi'     ,A_max)
	ofs_min_csi    = kwargs.get('ofs_min_csi'   ,ofs_min)
	ofs_max_csi    = kwargs.get('ofs_max_csi'   ,ofs_max)
	sigmax_min_csi = kwargs.get('sigmax_min_csi',sigmax_min)
	sigmax_max_csi = kwargs.get('sigmax_max_csi',sigmax_max)
	sigmay_min_csi = kwargs.get('sigmay_min_csi',sigmay_min)
	sigmay_max_csi = kwargs.get('sigmay_max_csi',sigmay_max)
	theta_csi      = kwargs.get('theta_csi'     ,Theta)
	shape_csi      = kwargs.get('shape_csi'     ,starshape)
	table_csi      = kwargs.get('table_csi'     ,tbl_strs_prp)
 
	A_noise            = kwargs.get('A_noise'           ,1)
	fixed_width_str    = kwargs.get('fixed_width_str'   ,True)	
	fixed_position_str = kwargs.get('fixed_position_str',True)
	fixed_size_str     = kwargs.get('fixed_size_str'    ,True)
	fixed_ampl_str     = kwargs.get('fixed_ampl_str'    ,True)
	fixed_offset_str   = kwargs.get('fixed_offset_str'  ,True)
	fixed_ampl_nse     = kwargs.get('fixed_ampl_nse'    ,True)
	

	n_stars            = kwargs.get('n_stars'          ,1)
	x_fxd_str          = kwargs.get('x_fxd_str'        ,(x_cube_dim / 2))
	y_fxd_str          = kwargs.get('y_fxd_str'        ,(y_cube_dim / 2))
	sx_fxd_str         = kwargs.get('sx_fxd_str'       ,5)
	sy_fxd_str         = kwargs.get('sy_fxd_str'       ,5)
	am_fxd_str         = kwargs.get('am_fxd_str'       ,1)
	offset_fxd_str     = kwargs.get('offset_fxd_str'   ,0)
	noise_fxd_str      = kwargs.get('noise_fxd_str'    ,1)
	sgm_1d_str_fxd     = kwargs.get('sgm_1d_str_fxd'   ,1)

	amp_star_gauss     = kwargs.get('amp_star_gauss'    ,True) 
	A_min_1dg_str      = kwargs.get('A_min_1dg_str'     ,1)
	A_max_1dg_str      = kwargs.get('A_max_1dg_str'     ,1.1)
	sigma_min_1dg_str  = kwargs.get('sigma_min_1dg_str' ,fwhm2sigma(2))
	sigma_max_1dg_str  = kwargs.get('sigma_max_1dg_str' ,fwhm2sigma(3))
	ofs_min_1dg_str    = kwargs.get('ofs_min_1dg_str'   ,0)
	ofs_max_1dg_str    = kwargs.get('ofs_max_1dg_str'   ,0.1)
	theta_vl_1dg_str   = kwargs.get('theta_vl_1dg_str'  ,0)
	shape_str          = kwargs.get('shape_str'         ,'circle')
	
	amp_nse_type       = kwargs.get('amp_nse_type'      ,'constant_u') 
	A_min_1dg_nse      = kwargs.get('A_min_1dg_nse'     ,1)
	A_max_1dg_nse      = kwargs.get('A_max_1dg_nse'     ,1)
	sigma_min_1dg_nse  = kwargs.get('sigma_min_1dg_nse' ,1)
	sigma_max_1dg_nse  = kwargs.get('sigma_max_1dg_nse' ,1)
	ofs_min_1dg_nse    = kwargs.get('ofs_min_1dg_nse'   ,0)
	ofs_max_1dg_nse    = kwargs.get('ofs_max_1dg_nse'   ,0)
	chn_wth_sze        = kwargs.get('chn_wth_sze'       ,channel_width)
	amp_1d_nse_fxd     = kwargs.get('amp_1d_nse_fxd'    ,10)

	cube_ofn_sfx       = kwargs.get('cube_ofn_sfx'      ,'')
	write_step_fits    = kwargs.get('write_step_fits'   ,False)

	nse_idp_all_slcs   = kwargs.get('nse_idp_all_slcs'  ,False)

	dst_img_dir        = kwargs.get('dst_img_dir'  ,None)
 
	img_stat_hst_f     = []
	hdu  = []

	if fixed_ampl_nse == False and amp_nse_type == 'gauss':
		sp_axs         = np.arange(-math.floor(z_cube_dim/2.),math.floor(z_cube_dim/2.)+1,1)		
		amp_1d_nse     = rdunf(A_min_1dg_nse    ,A_max_1dg_nse)
		sgm_1d_nse     = rdunf(sigma_min_1dg_nse/2,sigma_max_1dg_nse/2)
		ofs_1d_nse     = rdunf(ofs_min_1dg_nse  ,ofs_max_1dg_nse)
		gas_amp_nse    = func_1D_Gaussian(sp_axs,ofs_1d_nse,amp_1d_nse,sgm_1d_nse)
		amp_nse_2b_svd = max(gas_amp_nse)
	elif fixed_ampl_nse == False and amp_nse_type == 'uniform':
		sp_axs         = np.arange(-math.floor(z_cube_dim/2.),math.floor(z_cube_dim/2.)+1,1)
		uas_amp_nse    = np.random.uniform(A_min_1dg_nse,A_max_1dg_nse,sp_axs.shape)
		amp_nse_2b_svd = max(uas_amp_nse)
	elif fixed_ampl_nse == False and amp_nse_type=='constant_u':
		amp_nse_2b_svd = rdunf(A_min_1dg_nse,A_max_1dg_nse)
	elif fixed_ampl_nse == True and (amp_nse_type =='constant' or amp_nse_type == 'constant_u'):
		amp_nse_2b_svd = amp_1d_nse_fxd
	elif fixed_ampl_nse == False:
		pass
	else:
		print
		print (colored('Conditions not well defined! (322-Syn-Syn.py)','yellow'))
		print (fixed_ampl_nse)
		print (amp_nse_type)
		print
		quit()

	if fixed_ampl_str == True and amp_star_gauss==True and fixed_width_str==False:
		sp_axs      = np.arange(-math.floor(z_cube_dim/2.),math.floor(z_cube_dim/2.)+1,1)
		amp_1d_str  = rdunf(A_min_1dg_str    ,A_max_1dg_str)
		sgm_1d_str  = rdunf(sigma_min_1dg_str,sigma_max_1dg_str)
		ofs_1d_str  = 0
		gas_amp_str = func_1D_Gaussian(sp_axs,ofs_1d_str,amp_1d_str,sgm_1d_str)
		amp_str_2b_svd = max(gas_amp_str)
	elif fixed_ampl_str == True and amp_star_gauss==True and fixed_width_str==True:
		sp_axs      = np.arange(-math.floor(z_cube_dim/2.),math.floor(z_cube_dim/2.)+1,1)
		amp_1d_str  = rdunf(A_min_1dg_str    ,A_max_1dg_str)
		sgm_1d_str  = sgm_1d_str_fxd
		ofs_1d_str  = 0
		gas_amp_str = func_1D_Gaussian(sp_axs,ofs_1d_str,amp_1d_str,sgm_1d_str_fxd)
		amp_str_2b_svd = max(gas_amp_str)
	elif fixed_ampl_str == True and amp_star_gauss==False:
		sp_axs      = np.arange(-math.floor(z_cube_dim/2.),math.floor(z_cube_dim/2.)+1,1)
		amp_1d_str  = A_max_1dg_str
		sgm_1d_str  = sigma_max_1dg_nse
		ofs_1d_str  = 0#
		amp_2b_svd  = amp_1d_str
		gas_amp_str = amp_1d_str
		amp_str_2b_svd = A_max_1dg_str
	elif fixed_ampl_str == False:
		pass

	gas_amp_str    = amp_nse_2b_svd * gas_amp_str
	sgm_1d_str_nmb = sgm_1d_str 
	sgm_1d_str_vlc = sgm_1d_str * chn_wth_sze
	sgm_1d_str_vfw = sigma2fwhm(sgm_1d_str_vlc)
	raw_image_0    = create_raw_image(x_cube_dim,y_cube_dim)

	if nse_idp_all_slcs == False:
		# Add noise
		noise      = A_noise*np.random.normal(0, 1, raw_image_0.shape)
	elif nse_idp_all_slcs == True:
		pass

	for i,freq in enumerate(range(z_cube_dim)):
		#Create raw fits of size x_cube_dim y_cube_dim
		raw_image=create_raw_image(x_cube_dim,y_cube_dim)

		if nse_idp_all_slcs == False:
			pass
		elif nse_idp_all_slcs == True:
			# Add noise
			noise      = A_noise*np.random.normal(0, 1, raw_image.shape)
		
		# Convolve with a gaussian
		g2d       = astpy_conv_gaus_2dkernel(krn_conv_noise)#krn_conv_noise
		noise_psf = apcvl.convolve(noise, g2d, boundary='extend')

		#Noise Stats
		noise_avg  = np.mean(noise_psf)
		noise_med  = np.median(noise_psf)
		noise_std  = np.std(noise_psf)
		noise_rms  = noise_std#noise_med**2 + noise_std**2

		#Normalize noise 
		imagenoise   = raw_image+noise_psf
		imagenoise_n = imagenoise/noise_rms 

		#Normalized Noise Stats
		noise_avg  = np.mean(imagenoise_n)
		noise_med  = np.median(imagenoise_n)
		noise_std  = np.std(imagenoise_n)
		noise_rms  = noise_std#noise_med**2 + noise_std**2

		if fixed_ampl_str == True and amp_star_gauss==True:
			am_fxd_str  = gas_amp_str[i] 
		elif fixed_ampl_str == True and amp_star_gauss==False:
			am_fxd_str     = A_max_1dg_str

		#Create stars
		star_array = create_star_image(nstrs_csi= n_stars,
			nx_csi             = x_cube_dim        ,ny_csi         = y_cube_dim,
			A_min_csi          = A_min_csi         ,A_max_csi      = A_max_csi ,
			ofs_min_csi        = ofs_min_csi       ,ofs_max_csi    = ofs_max_csi,
			sigmax_min_csi     = sigmax_min_csi    ,sigmax_max_csi = sigmax_max_csi,
			sigmay_min_csi     = sigmay_min_csi    ,sigmay_max_csi = sigmay_max_csi,
			theta_csi          = theta_csi         ,theta_vl_1dg_str=theta_vl_1dg_str,
			shape_csi          = shape_csi ,			
			table_csi          = table_csi         ,
			fixed_position_str = fixed_position_str, x_fxd_str      = x_fxd_str      ,y_fxd_str  = y_fxd_str ,
			fixed_size_str     = fixed_size_str    , sx_fxd_str     = sx_fxd_str     ,sy_fxd_str = sy_fxd_str,
			fixed_ampl_str     = fixed_ampl_str    , am_fxd_str     = am_fxd_str     ,
			fixed_offset_str   = fixed_offset_str  , offset_fxd_str = offset_fxd_str )

		#Stars + Norm noise
		img_noise_psf_stars = star_array[0] + imagenoise_n

		# Display Image
		displ_image = False
		if displ_image == True:
			displ_fits(img_noise_psf_stars,'norm','viridis')
		elif displ_image == False:
			pass
		img_stat_hst_f.append(img_noise_psf_stars)

		if write_step_fits == True:
			# Write out to FITS image
			apfts.writeto(ifn_raw            , raw_image          , overwrite=True)#raw
			apfts.writeto(ifn_noise          , imagenoise         , overwrite=True)#noise
			apfts.writeto(ifn_noise_psf      , noise_psf          , overwrite=True)#convolved noise
			apfts.writeto(ifn_noise_n        , imagenoise_n       , overwrite=True)#normalized
			apfts.writeto(ifn_stars          , star_array[0]      , overwrite=True)#stars
			apfts.writeto(ifn_noise_psf_stars, img_noise_psf_stars, overwrite=True)#Stars + Norm noise

			print colored(ifn_raw,'yellow')
			print colored(ifn_noise,'yellow')
			print colored(ifn_noise_psf,'yellow')
			print colored(ifn_noise_n,'yellow')
			print colored(ifn_stars,'yellow')
			print colored(ifn_noise_psf_stars,'yellow')

		elif write_step_fits == False:
			pass
	
	hdu  = apfts.PrimaryHDU(img_stat_hst_f)
	hdul = apfts.HDUList([hdu])
	if dst_img_dir != None:
		SyntheticCube=dst_img_dir + 'SyntheticCube'+cube_ofn_sfx+'.fits'
		if os.path.exists(dst_img_dir) == False:
			print
			print (colored(dst_img_dir,'yellow'))
			print (colored('Directory does not exist!','yellow'))
			print (colored('Creating it.','yellow'))
			print
			os.system('mkdir ' + dst_img_dir)
		else:
			pass
	else:
		SyntheticCube=img_dir_res + 'SyntheticCube'+cube_ofn_sfx+'.fits'

	hdul.writeto(SyntheticCube,overwrite=True)
	Header_Get_Add(SyntheticCube,'SIMPLE', 'T',header_comment='conforms to FITS standard ')
	Header_Get_Add(SyntheticCube,'BITPIX', -32,header_comment='array data type           ')
	Header_Get_Add(SyntheticCube,'NAXIS' ,   3,header_comment='number of array dimensions')
	Header_Get_Add(SyntheticCube,'NAXIS1', 256)
	Header_Get_Add(SyntheticCube,'NAXIS2', 256)
	Header_Get_Add(SyntheticCube,'NAXIS3',  17)

	Header_Get_Add(SyntheticCube,'BMAJ'    ,1.250000000000E-03)
	Header_Get_Add(SyntheticCube,'BMIN'    ,1.250000000000E-03)
	Header_Get_Add(SyntheticCube,'BPA'     ,0.0000000)
	Header_Get_Add(SyntheticCube,'BTYPE'   ,'Intensity')
	Header_Get_Add(SyntheticCube,'OBJECT'  ,'HOT2_EI_CII_G09.v2.58')
	Header_Get_Add(SyntheticCube,'BUNIT'   ,'Jy'                ,header_comment=' Brightness (pixel) unit')
	Header_Get_Add(SyntheticCube,'ALTRVAL' ,-1.999999999950E+06 ,header_comment='Alternate frequency reference value')
	Header_Get_Add(SyntheticCube,'ALTRPIX' ,1.000000000000E+00  ,header_comment='Alternate frequency reference pixel')
	Header_Get_Add(SyntheticCube,'VELREF ' ,258 ,header_comment='1 LSR, 2 HEL, 3 OBS, +256 Radio COMMENT casacore non-standard usage: 4 LSD, 5 GEO, 6 SOU, 7 GAL')  
	Header_Get_Add(SyntheticCube,'TELESCOP','ALMA')
	Header_Get_Add(SyntheticCube,'OBSERVER','hmendez')
	Header_Get_Add(SyntheticCube,'TIMESYS' ,'UTC')
	Header_Get_Add(SyntheticCube,'OBSRA'   ,1.363858337500E+02)
	Header_Get_Add(SyntheticCube,'OBSDEC'  ,2.039406944444E+00)
	Header_Get_Add(SyntheticCube,'DATE'    ,'2017-07-06T21:54:46.399391' ,header_comment='Date FITS file was written')
	Header_Get_Add(SyntheticCube,'ORIGIN'  ,'CASA 4.7.2-REL (r39762)'                                             )
	Header_Get_Add(SyntheticCube,'WCSAXES' ,3)
	Header_Get_Add(SyntheticCube,'CRPIX1'  ,129.0)
	Header_Get_Add(SyntheticCube,'CRPIX2'  ,129.0)
	Header_Get_Add(SyntheticCube,'CRPIX3'  ,1.0)
	Header_Get_Add(SyntheticCube,'CDELT1'  , -0.0001388888888889)
	Header_Get_Add(SyntheticCube,'CDELT2'  ,0.0001388888888889)
	Header_Get_Add(SyntheticCube,'CDELT3'  ,249.99999999999)
	Header_Get_Add(SyntheticCube,'CUNIT1'  ,'deg')
	Header_Get_Add(SyntheticCube,'CUNIT2'  ,'deg')
	Header_Get_Add(SyntheticCube,'CUNIT3'  ,'km s-1')
	Header_Get_Add(SyntheticCube,'CTYPE1'  ,'RA---SIN')
	Header_Get_Add(SyntheticCube,'CTYPE2'  ,'DEC--SIN')
	Header_Get_Add(SyntheticCube,'CTYPE3'  ,'VRAD')
	Header_Get_Add(SyntheticCube,'CRVAL1'  ,136.38583375)
	Header_Get_Add(SyntheticCube,'CRVAL2'  ,2.039406944444)
	Header_Get_Add(SyntheticCube,'CRVAL3'  ,-1999.9999999531)
	Header_Get_Add(SyntheticCube,'PV2_1'   ,0.0)
	Header_Get_Add(SyntheticCube,'PV2_2'   ,0.0)
	Header_Get_Add(SyntheticCube,'LONPOLE' ,180.0)
	Header_Get_Add(SyntheticCube,'LATPOLE' ,2.039406944444)
	Header_Get_Add(SyntheticCube,'RESTFRQ' ,restframe_frequency)
	Header_Get_Add(SyntheticCube,'RADESYS' ,'FK5')
	Header_Get_Add(SyntheticCube,'EQUINOX' ,2000.0)
	Header_Get_Add(SyntheticCube,'SPECSYS' ,'BARYCENT')
	Header_Get_Add(SyntheticCube,'OBSGEO-X',2225142.180269)
	Header_Get_Add(SyntheticCube,'OBSGEO-Y',-5440307.370349)
	Header_Get_Add(SyntheticCube,'OBSGEO-Z',-2481029.851874)

	Cat_Ipt_Tbl   = Table_Read(cat_tbl,tbl_format_ipt)
	fits          = Cat_Ipt_Tbl[2]
	delta_nu      = Cat_Ipt_Tbl[4]
	z             = Cat_Ipt_Tbl[8] 
	Lfir          = Cat_Ipt_Tbl[11]
	nu            = Cat_Ipt_Tbl[13]
	vel           = Cat_Ipt_Tbl[14]
	num_obj       = len(Cat_Ipt_Tbl[0])

	z_sample_avg     = np.mean(z)
	z_sample_med     = np.median(z)
	z_sample_1sl     = np.nanpercentile(z, 15.9)
	z_sample_1sh     = np.nanpercentile(z, 84.1)
	z_sample_2sl     = np.nanpercentile(z, 2.30)
	z_sample_2sh     = np.nanpercentile(z, 97.7)
	z_sample_3sl     = np.nanpercentile(z, 0.20)
	z_sample_3sh     = np.nanpercentile(z, 99.8)
	z_sample_p25     = np.nanpercentile(z, 25.0)
	z_sample_p75     = np.nanpercentile(z, 75.0)


	Header_Get_Add(SyntheticCube,'STK_NUM',cube_number    ,header_comment='Redshift Average')              
	Header_Get_Add(SyntheticCube,'STZ_AVG',z_sample_avg   ,header_comment='Redshift Average')              
	Header_Get_Add(SyntheticCube,'STZ_MED',z_sample_med   ,header_comment='Redshift Median')               
	Header_Get_Add(SyntheticCube,'STZ_1SL',z_sample_1sl   ,header_comment='Redshift 1 sgm lw lmt 15.9 pct')
	Header_Get_Add(SyntheticCube,'STZ_1SH',z_sample_1sh   ,header_comment='Redshift 1 sgm hg lmt 84.1 pct')
	Header_Get_Add(SyntheticCube,'STZ_2SL',z_sample_2sl   ,header_comment='Redshift 2 sgm lw lmt 2.30 pct')
	Header_Get_Add(SyntheticCube,'STZ_2SH',z_sample_2sh   ,header_comment='Redshift 2 sgm hg lmt 97.7 pct')
	Header_Get_Add(SyntheticCube,'STZ_3SL',z_sample_3sl   ,header_comment='Redshift 3 sgm lw lmt 0.20 pct')
	Header_Get_Add(SyntheticCube,'STZ_3SH',z_sample_3sh   ,header_comment='Redshift 3 sgm hg lmt 99.8 pct')
	Header_Get_Add(SyntheticCube,'STZ_P25',z_sample_p25   ,header_comment='Redshift 25 pct')               
	Header_Get_Add(SyntheticCube,'STZ_P75',z_sample_p75   ,header_comment='Redshift 75 pct')               

	X_2D  = (star_array[1])
	Y_2D  = (star_array[2])
	A_2D  = (star_array[3])
	AM_2D = (star_array[4])
	SX_2D = (star_array[5])
	SY_2D = (star_array[6])
	T_2D  = (star_array[7])
	OS_2D = (star_array[8])
	TN_2D = (star_array[9])

	hdul.close()
	SyntheticCube_clean = SyntheticCube.rsplit('/',1)[1]
	return SyntheticCube_clean,amp_str_2b_svd,amp_nse_2b_svd,sgm_1d_str_nmb,sgm_1d_str_vlc,sgm_1d_str_vfw,ofs_1d_str,X_2D,Y_2D,A_2D,AM_2D,SX_2D,SY_2D,T_2D,OS_2D,TN_2D

####Fnc_Syn_Syn####
