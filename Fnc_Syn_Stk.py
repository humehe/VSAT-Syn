import bottleneck as bn
from astropy import stats as apsts

import scipy.integrate as integrate

from Fnc_Syn_Dir import *
from Fnc_Syn_Spc import *
from Fnc_Syn_Tbl import *

####Fnc_Syn_Stk####
def Cube_Stack(Cubes2bStacked,name,wght_img_2bstack,sig_clp,*args, **kwargs):
	wrt_fits         = kwargs.get('wrt_fits'       ,True)
	pst_msk          = kwargs.get('pst_msk'        ,False)
	pst_smt          = kwargs.get('pst_smt'        ,False)
	pst_cnt          = kwargs.get('pst_cnt'        ,False)
	stack_ext        = kwargs.get('stack_ext'      ,None)
	new_CRVAL1_head  = kwargs.get('new_CRVAL1_head',None)
	new_CDELT1_head  = kwargs.get('new_CDELT1_head',None)
	smt_spc_pst      = kwargs.get('smt_spc_pst'    ,False)
	smooth_shape     = kwargs.get('smooth_shape'   ,'gaussian')
	wght_type        = kwargs.get('wght_type'      ,None)
	
	sufix            = kwargs.get('sufix'          ,'')
	freq_obs_f       = kwargs.get('freq_obs_f'     ,99999)

	stack_lite       = kwargs.get('stack_lite'     ,True)

	spc_wdt_dir      = kwargs.get('spc_wdt_dir'    ,500)

	cp_bs_hdrs       = kwargs.get('cp_bs_hdrs'     ,False)

	stt_var			 = kwargs.get('stt_var',False)
	stt_mst_tbl		 = kwargs.get('stt_mst_tbl',None)
	stt_syn			 = kwargs.get('stt_syn',True)
	stt_syn_tbl		 = kwargs.get('stt_syn_tbl',None)
	stt_hdr			 = kwargs.get('stt_hdr',None)

	[Cube_Mask_Nan(img)     for img in Cubes2bStacked]

	img_2bstack    = [apgtdt(img,memmap=False) for img in Cubes2bStacked]
	wcs            = kwargs.get('wcs'            ,apwcs(Cubes2bStacked[0]))
	try:
		wcs       = wcs.dropaxis(3)
	except IndexError:
		pass

	print
	print ('Number of galaxies to be stacked (histogram): ',len(img_2bstack))

	if sig_clp == True:
		img_flt       = apsts.sigma_clip(img_2bstack,sigma=sigma_cut,axis=0,iters=None,cenfunc=sigma_cen_fct, copy=True)

		print
		print (colored('Sigma-clipping for stacking!','yellow'))
		print (colored('Sigma Cut                    : ' + str(sigma_cut),'yellow'))
		print (colored('Central function             : ' + str(sigma_cen_fct), 'yellow'))
		print (colored('Central Value for clipping   : ' + str(sigma_cen_fct),'yellow'))

		img_flt.set_fill_value(sigma_msk_fill_val)
		img_flt_filled = img_flt.filled()
		img_stat       = img_flt_filled

	elif sig_clp == False:
		img_stat   = img_2bstack

	wght_img_copy = wght_img_2bstack
	wght_img_stat = wght_img_2bstack
	wght_img_stat = np.asarray(wght_img_stat)
	img_staw      = []

	img_stat_smw_f = []

	[img_staw.append(np.asarray(img_stat)[j]*np.asarray(wght_img_stat)[j]) for j in range(len(wght_img_stat))]
	img_staw      = np.asarray(img_staw)
	[img_stat_smw_f.append(np.divide(np.asarray(img_staw)[j],np.asarray(img_stat)[j])) for j in range(len(img_stat))]

	print
	print (colored('Original shape                                               : '+str(np.asarray(img_stat).shape),'cyan'))
	img_stat = np.squeeze(img_stat) 
	N,F,l,m = np.asarray(img_stat).shape 
	print (colored('Squeezed useless extra dimensions                            : '+str(np.asarray(img_stat).shape),'cyan'))
	print (colored('Dimension Numer of Cubes, Number of channels, X size, Y size : '+str(N)+', '+str(F)+', '+str(l)+', '+str(m),'cyan'))
	print

	img_res_sum = scspc(data=bn.nansum(np.array(img_stat)             , axis=0)  ,wcs=wcs) 
	img_res_avg = scspc(data=bn.nanmean(np.array(img_stat)            , axis=0)  ,wcs=wcs) 
	img_res_med = scspc(data=bn.nanmedian(np.array(img_stat)          , axis=0)  ,wcs=wcs) 

	print
	print (colored('Sum, Mean, Median : Stacked data cubes OK','yellow'))
	print 

	if stack_lite == False:

		img_stat_hst_f = []
		img_stat_hsw_f = []

		#BEGINS HISTO
		widgets = ['Computing histogram for Stacks: ', Percentage(), ' ', Bar(marker='*',left='[',right=']'),
				   ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options
		pbar = ProgressBar(widgets=widgets, maxval=F)
		pbar.start()
		for freq in range(F):
			pbar.update(freq)
			FREQ = np.asarray(img_stat)[:,freq,:,:]

			img_stat_hst_y = []
			img_stat_hsw_y = []

			for y_dim in range(l):
				Y_ROW = np.asarray(img_stat)[:,freq,y_dim,:]
				Transpose  = np.asarray(Y_ROW).T
				Transposw  = np.asarray(Y_ROW).T
				img_stat_hst_x = []
				img_stat_hsw_x = []

				for x_dim in range(len(Transpose)):
					if np.isnan(sigma_msk_fill_val) == True:
						non_msk_num = int(np.count_nonzero(~np.isnan(Transpose[x_dim])))
						msk_num     = int(np.count_nonzero(np.isnan(Transpose[x_dim])))
						img_stat_hst_x.append(float(non_msk_num))

						non_msk_num_wghts = int(np.count_nonzero(~np.isnan(Transposw[x_dim])))
						msk_num_wghts     = int(np.count_nonzero(np.isnan(Transposw[x_dim])))
						img_stat_hsw_x.append(float(non_msk_num_wghts))

					elif np.isnan(sigma_msk_fill_val) == False:
						pass
						non_msk_num = int(np.count_nonzero(Transpose[x_dim]!=sigma_msk_fill_val))
						img_stat_hst_x.append(float(non_msk_num))

						non_msk_num_wghts = int(np.count_nonzero(Transposw[x_dim]!=sigma_msk_fill_val))
						img_stat_hsw_x.append(float(non_msk_num_wghts))
					else:
						pass
				
				img_stat_hst_x = np.reshape(img_stat_hst_x,(m))
				img_stat_hsw_x = np.reshape(img_stat_hsw_x,(m))

				img_stat_hst_y.append(img_stat_hst_x)
				img_stat_hsw_y.append(img_stat_hsw_x)

			img_stat_hst_f.append(img_stat_hst_y)
			img_stat_hsw_f.append(img_stat_hsw_y)
		pbar.finish()
		#ENDS HISTO

		img_sts_hst = scspc(data=np.asarray(img_stat_hst_f)                          ,wcs=wcs) 
		img_res_std = scspc(data=bn.nanstd(np.array(img_stat)             , axis=0)  ,wcs=wcs)


		print
		print (colored('Histogram, Std: Stacked data cubes OK','yellow'))
		print 

		img_res_suw_pre = np.asarray(bn.nansum(np.array(img_staw)                , axis=0))
		img_sts_wsu_pre = np.asarray(bn.nansum(np.array(img_stat_smw_f)          , axis=0))

		img_sts_wsu_pre = np.squeeze(img_sts_wsu_pre)
		img_res_suw_pre = np.squeeze(img_res_suw_pre)

		print
		print (colored('Weights Sum Weighted Sum pre computations: OK','yellow'))
		print 

		img_sts_hsw = scspc(data=np.asarray(img_stat_hsw_f)                          ,wcs=wcs)
		img_sts_wsu = scspc(data=img_sts_wsu_pre                                     ,wcs=wcs)
		img_res_suw = scspc(data=img_res_suw_pre                                     ,wcs=wcs)
		img_res_avw = scspc(data=img_res_suw_pre.astype(float)/img_sts_wsu_pre.astype(float) ,wcs=wcs)
		
		print
		print (colored('SW Histogram, Sum of weights, Weighted Sum: Stacked data cubes OK','yellow'))
		print

		img_res_1sl = scspc(data=np.nanpercentile(np.array(img_stat), 15.9, axis=0)  ,wcs=wcs)	
		img_res_1sh = scspc(data=np.nanpercentile(np.array(img_stat), 84.1, axis=0)  ,wcs=wcs)
		img_res_2sl = scspc(data=np.nanpercentile(np.array(img_stat), 2.30, axis=0)  ,wcs=wcs)
		img_res_2sh = scspc(data=np.nanpercentile(np.array(img_stat), 97.7, axis=0)  ,wcs=wcs)
		img_res_3sl = scspc(data=np.nanpercentile(np.array(img_stat), 0.20, axis=0)  ,wcs=wcs)
		img_res_3sh = scspc(data=np.nanpercentile(np.array(img_stat), 99.8, axis=0)  ,wcs=wcs)
		img_res_p25 = scspc(data=np.nanpercentile(np.array(img_stat), 25.0, axis=0)  ,wcs=wcs)
		img_res_p75 = scspc(data=np.nanpercentile(np.array(img_stat), 75.0, axis=0)  ,wcs=wcs)

		print ('Stacked images through : sum, mean, median, and percentiles: ')
		print ('17., 83.0, (1 sigma)')
		print ('2.5, 97.5, (2 sigma)')
		print ('0.5, 99.5, (3 sigma)')
		print ('25., 75.0, (interquantile)')
		print		
		print (colored('Percentiles: Stacked data cubes OK','yellow'))
		print 
	elif stack_lite == True:
		pass

	bs_func = kwargs.get('bs_func','')

	if wrt_fits==True:
		if  '-BS-' in name:
			print (colored(name,'yellow'))
			spc_dir_dst = str_bst_stk + str(spc_wdt_dir) +'/'
			if os.path.exists(spc_dir_dst)==False:
				print
				print (colored('Stacked width directory does not exist!','yellow'))
				print (colored('Creating it!','yellow'))
				print
				os.makedirs(spc_dir_dst)
			else:
				pass
		elif  '-BS_MST' in name:
			print (colored(name,'yellow'))
			spc_dir_dst = stt_bst_stk + str(spc_wdt_dir) +'/'
			if os.path.exists(spc_dir_dst)==False:
				print
				print (colored('Stacked width directory does not exist!','yellow'))
				print (colored('Creating it!','yellow'))
				print
				os.makedirs(spc_dir_dst)
			else:
				pass
		else:
			spc_dir_dst = stk_dir_res + str(spc_wdt_dir) +'/'
			if os.path.exists(spc_dir_dst)==False:
				print
				print (colored('Stacked width directory does not exist!','yellow'))
				print (colored('Creating it!','yellow'))
				print
				os.makedirs(spc_dir_dst)
			else:
				pass

		spec_file_sum_ofn = spc_dir_dst + str(name) + bs_func + '-stk-sum-' + str(sufix) + 'kms.fits'
		spec_file_avg_ofn = spc_dir_dst + str(name) + bs_func + '-stk-avg-' + str(sufix) + 'kms.fits'
		spec_file_med_ofn = spc_dir_dst + str(name) + bs_func + '-stk-med-' + str(sufix) + 'kms.fits'
		spec_file_hst_ofn = spc_dir_dst + str(name) + bs_func + '-stk-hst-' + str(sufix) + 'kms.fits'
		spec_file_std_ofn = spc_dir_dst + str(name) + bs_func + '-stk-std-' + str(sufix) + 'kms.fits'
		spec_file_p25_ofn = spc_dir_dst + str(name) + bs_func + '-stk-p25-' + str(sufix) + 'kms.fits'
		spec_file_p75_ofn = spc_dir_dst + str(name) + bs_func + '-stk-p75-' + str(sufix) + 'kms.fits'
		spec_file_1sl_ofn = spc_dir_dst + str(name) + bs_func + '-stk-1sl-' + str(sufix) + 'kms.fits'
		spec_file_1sh_ofn = spc_dir_dst + str(name) + bs_func + '-stk-1sh-' + str(sufix) + 'kms.fits'
		spec_file_2sl_ofn = spc_dir_dst + str(name) + bs_func + '-stk-2sl-' + str(sufix) + 'kms.fits'
		spec_file_2sh_ofn = spc_dir_dst + str(name) + bs_func + '-stk-2sh-' + str(sufix) + 'kms.fits'
		spec_file_3sl_ofn = spc_dir_dst + str(name) + bs_func + '-stk-3sl-' + str(sufix) + 'kms.fits'
		spec_file_3sh_ofn = spc_dir_dst + str(name) + bs_func + '-stk-3sh-' + str(sufix) + 'kms.fits'	

		spec_file_hsw_ofn = spc_dir_dst + str(name) + bs_func + '-stk-hsw-' + str(sufix) + 'kms.fits'
		spec_file_wsu_ofn = spc_dir_dst + str(name) + bs_func + '-stk-wsu-' + str(sufix) + 'kms.fits'
		spec_file_suw_ofn = spc_dir_dst + str(name) + bs_func + '-stk-suw-' + str(sufix) + 'kms.fits'
		spec_file_avw_ofn = spc_dir_dst + str(name) + bs_func + '-stk-avw-' + str(sufix) + 'kms.fits'

		spec_file_sum     = img_res_sum.write(spec_file_sum_ofn,overwrite=True)
		spec_file_avg     = img_res_avg.write(spec_file_avg_ofn,overwrite=True)
		spec_file_med     = img_res_med.write(spec_file_med_ofn,overwrite=True)

		Cube_Freq2VelAxis(spec_file_sum_ofn)
		Cube_Freq2VelAxis(spec_file_avg_ofn)
		Cube_Freq2VelAxis(spec_file_med_ofn)

		if stack_lite == False:
			spec_file_hst     = img_sts_hst.write(spec_file_hst_ofn,overwrite=True)
			spec_file_std     = img_res_std.write(spec_file_std_ofn,overwrite=True)

			spec_file_p25     = img_res_p25.write(spec_file_p25_ofn,overwrite=True)
			spec_file_p75     = img_res_p75.write(spec_file_p75_ofn,overwrite=True)
			spec_file_1sl     = img_res_1sl.write(spec_file_1sl_ofn,overwrite=True)
			spec_file_1sh     = img_res_1sh.write(spec_file_1sh_ofn,overwrite=True)
			spec_file_2sl     = img_res_2sl.write(spec_file_2sl_ofn,overwrite=True)
			spec_file_2sh     = img_res_2sh.write(spec_file_2sh_ofn,overwrite=True)
			spec_file_3sl     = img_res_3sl.write(spec_file_3sl_ofn,overwrite=True)
			spec_file_3sh     = img_res_3sh.write(spec_file_3sh_ofn,overwrite=True)

			spec_file_hsw     = img_sts_hsw.write(spec_file_hsw_ofn,overwrite=True)
			spec_file_wsu     = img_sts_wsu.write(spec_file_wsu_ofn,overwrite=True)
			spec_file_suw     = img_res_suw.write(spec_file_suw_ofn,overwrite=True)

			spec_file_avw     = img_res_avw.write(spec_file_avw_ofn,overwrite=True)

			Cube_Freq2VelAxis(spec_file_hst_ofn)
			Cube_Freq2VelAxis(spec_file_std_ofn)

			Cube_Freq2VelAxis(spec_file_p25_ofn)
			Cube_Freq2VelAxis(spec_file_p75_ofn)
			Cube_Freq2VelAxis(spec_file_1sl_ofn)
			Cube_Freq2VelAxis(spec_file_1sh_ofn)
			Cube_Freq2VelAxis(spec_file_2sl_ofn)
			Cube_Freq2VelAxis(spec_file_2sh_ofn)
			Cube_Freq2VelAxis(spec_file_3sl_ofn)
			Cube_Freq2VelAxis(spec_file_3sh_ofn)
			Cube_Freq2VelAxis(spec_file_hsw_ofn)
			Cube_Freq2VelAxis(spec_file_wsu_ofn)
			Cube_Freq2VelAxis(spec_file_suw_ofn)
			Cube_Freq2VelAxis(spec_file_avw_ofn)

			OPT_STCK_FLS = [spec_file_sum_ofn,spec_file_avg_ofn,spec_file_med_ofn,spec_file_hst_ofn,
			spec_file_std_ofn,
			spec_file_p25_ofn,spec_file_p75_ofn,
			spec_file_1sl_ofn,spec_file_1sh_ofn,
			spec_file_2sl_ofn,spec_file_2sh_ofn,
			spec_file_3sl_ofn,spec_file_3sh_ofn,
			spec_file_hsw_ofn,spec_file_wsu_ofn,spec_file_suw_ofn,spec_file_avw_ofn]
		elif stack_lite == True:
			OPT_STCK_FLS = [spec_file_sum_ofn,spec_file_avg_ofn,spec_file_med_ofn]
		[Header_Updt(spec_sts_res,'STK_NUM' ,len(img_2bstack), header_comment = 'Number of galaxies used for Stack') for spec_sts_res in OPT_STCK_FLS]
	else:
		pass


	print ('Imaged Stacked files names: ')
	print
	print (colored(spec_file_sum_ofn,'cyan'))
	print (colored(spec_file_avg_ofn,'cyan'))
	print (colored(spec_file_med_ofn,'cyan'))
	if stack_lite == False:
		print (colored(spec_file_hst_ofn,'cyan'))
		print (colored(spec_file_std_ofn,'cyan'))
		print (colored(spec_file_p25_ofn,'cyan'))
		print (colored(spec_file_p75_ofn,'cyan'))
		print (colored(spec_file_1sl_ofn,'cyan'))
		print (colored(spec_file_1sh_ofn,'cyan'))
		print (colored(spec_file_2sl_ofn,'cyan'))
		print (colored(spec_file_2sh_ofn,'cyan'))
		print (colored(spec_file_3sl_ofn,'cyan'))
		print (colored(spec_file_3sh_ofn,'cyan'))

		print (colored(spec_file_hsw_ofn,'yellow'))
		print (colored(spec_file_avw_ofn,'yellow'))
		print (colored(spec_file_suw_ofn,'yellow'))
	elif stack_lite == True:
		pass

	if stack_lite == True:
		FNL_SPEC_RES = [spec_file_med,spec_file_avg,spec_file_sum]
	elif stack_lite == False:
		FNL_SPEC_RES = [
					spec_file_med,spec_file_avg,spec_file_sum,spec_file_std,
					spec_file_hst,
					spec_file_1sl,spec_file_1sh,
					spec_file_2sl,spec_file_2sh,
					spec_file_3sl,spec_file_3sh,
					spec_file_p25,spec_file_p75,
					spec_file_hsw,spec_file_wsu,spec_file_suw,spec_file_avw]
	if cp_bs_hdrs == True:
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'BSCALE')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'BZERO')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'BMAJ')     for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'BMIN')     for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'BPA')      for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'BTYPE')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'EQUINOX')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'RADESYS')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'BUNIT')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'RADESYS')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'LONPOLE')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'LATPOLE')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC1_1')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC2_1')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC3_1')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC1_2')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC2_2')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC3_2')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC1_3')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC2_3')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PC3_3')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CTYPE1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CRVAL1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CDELT1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CRPIX1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CUNIT1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CTYPE2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CRVAL2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CDELT2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CRPIX2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CUNIT2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CTYPE3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CRVAL3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CDELT3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CRPIX3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'CUNIT3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PV2_1')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'PV2_2')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'RESTFRQ')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'SPECSYS')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'ALTRVAL')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'ALTRPIX')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'VELREF')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'TELESCOP') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'OBSERVER') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'DATE-OBS') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'TIMESYS')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'OBSRA')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'OBSDEC')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'OBSGEO-X') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'OBSGEO-Y') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'OBSGEO-Z') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'DATE')     for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,Cubes2bStacked[0],'ORIGIN')   for stk_res_flr in OPT_STCK_FLS]
	else:
		pass
	if stt_var == True:
		print
		print (colored('Adding stat to fits headers!','yellow'))
		print
		tbl_sts = Table_Ipt_Cat_Stats(stt_mst_tbl,stt_hdr)
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][0] ,tbl_sts[1][0] ,header_comment='Redshift Average')                         for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][1] ,tbl_sts[1][1] ,header_comment='Redshift Median')                          for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][2] ,tbl_sts[1][2] ,header_comment='Redshift 1 sgm lw lmt 15.9 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][3] ,tbl_sts[1][3] ,header_comment='Redshift 1 sgm hg lmt 84.1 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][4] ,tbl_sts[1][4] ,header_comment='Redshift 2 sgm lw lmt 2.30 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][5] ,tbl_sts[1][5] ,header_comment='Redshift 2 sgm hg lmt 97.7 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][6] ,tbl_sts[1][6] ,header_comment='Redshift 3 sgm lw lmt 0.20 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][7] ,tbl_sts[1][7] ,header_comment='Redshift 3 sgm hg lmt 99.8 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][8] ,tbl_sts[1][8] ,header_comment='Redshift 25 pct')                          for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][9] ,tbl_sts[1][9] ,header_comment='Redshift 75 pct')                          for Stacked_Cube in OPT_STCK_FLS]

	else:
		pass

	if stt_syn == True:
		print
		print (colored('Adding synthetic info!','yellow'))
		tbl_sts = Table_Read_Syn_Singl_It(stt_syn_tbl,tbl_format_ipt)
	 	src_amp_avg = np.mean(tbl_sts[2])
	 	nse_amp_avg = np.mean(tbl_sts[3])
	 	snr_amp_avg = np.mean(tbl_sts[4])
	 	sig_nmb_avg = np.mean(tbl_sts[5])
	 	sig_vel_avg = np.mean(tbl_sts[6])
	 	sig_fwh_avg = np.mean(tbl_sts[7])
	 	src_ofs_avg = np.mean(tbl_sts[8])
	 	xctr_2D_avg = np.mean(tbl_sts[9])
	 	yctr_2D_avg = np.mean(tbl_sts[10])
	 	ampl_2D_avg = np.mean(tbl_sts[11])
	 	amax_2D_avg = np.mean(tbl_sts[12])
	 	sgmx_2D_avg = np.mean(tbl_sts[13])
	 	sgmy_2D_avg = np.mean(tbl_sts[14])
	 	thet_2D_avg = np.mean(tbl_sts[15])
	 	ofst_2D_avg = np.mean(tbl_sts[16])
	 	tmXn_2D_avg = np.mean(tbl_sts[17])

	 	src_amp_med = np.median(tbl_sts[2])
	 	nse_amp_med = np.median(tbl_sts[3])
	 	snr_amp_med = np.median(tbl_sts[4])
	 	sig_nmb_med = np.median(tbl_sts[5])
	 	sig_vel_med = np.median(tbl_sts[6])
	 	sig_fwh_med = np.median(tbl_sts[7])
	 	src_ofs_med = np.median(tbl_sts[8])
	 	xctr_2D_med = np.median(tbl_sts[9])
	 	yctr_2D_med = np.median(tbl_sts[10])
	 	ampl_2D_med = np.median(tbl_sts[11])
	 	amax_2D_med = np.median(tbl_sts[12])
	 	sgmx_2D_med = np.median(tbl_sts[13])
	 	sgmy_2D_med = np.median(tbl_sts[14])
	 	thet_2D_med = np.median(tbl_sts[15])
	 	ofst_2D_med = np.median(tbl_sts[16])
	 	tmXn_2D_med = np.median(tbl_sts[17])

	 	src_amp_std = np.std(tbl_sts[2])
	 	nse_amp_std = np.std(tbl_sts[3])
	 	snr_amp_std = np.std(tbl_sts[4])
	 	sig_nmb_std = np.std(tbl_sts[5])
	 	sig_vel_std = np.std(tbl_sts[6])
	 	sig_fwh_std = np.std(tbl_sts[7])
	 	src_ofs_std = np.std(tbl_sts[8])
	 	xctr_2D_std = np.std(tbl_sts[9])
	 	yctr_2D_std = np.std(tbl_sts[10])
	 	ampl_2D_std = np.std(tbl_sts[11])
	 	amax_2D_std = np.std(tbl_sts[12])
	 	sgmx_2D_std = np.std(tbl_sts[13])
	 	sgmy_2D_std = np.std(tbl_sts[14])
	 	thet_2D_std = np.std(tbl_sts[15])
	 	ofst_2D_std = np.std(tbl_sts[16])
	 	tmXn_2D_std = np.std(tbl_sts[17])

	 	src_amp_min = min(tbl_sts[2])
	 	nse_amp_min = min(tbl_sts[3])
	 	snr_amp_min = min(tbl_sts[4])
	 	sig_nmb_min = min(tbl_sts[5])
	 	sig_vel_min = min(tbl_sts[6])
	 	sig_fwh_min = min(tbl_sts[7])
	 	src_ofs_min = min(tbl_sts[8])
	 	xctr_2D_min = min(tbl_sts[9])
	 	yctr_2D_min = min(tbl_sts[10])
	 	ampl_2D_min = min(tbl_sts[11])
	 	amax_2D_min = min(tbl_sts[12])
	 	sgmx_2D_min = min(tbl_sts[13])
	 	sgmy_2D_min = min(tbl_sts[14])
	 	thet_2D_min = min(tbl_sts[15])
	 	ofst_2D_min = min(tbl_sts[16])
	 	tmXn_2D_min = min(tbl_sts[17])

	 	src_amp_max = max(tbl_sts[2])
	 	nse_amp_max = max(tbl_sts[3])
	 	snr_amp_max = max(tbl_sts[4])
	 	sig_nmb_max = max(tbl_sts[5])
	 	sig_vel_max = max(tbl_sts[6])
	 	sig_fwh_max = max(tbl_sts[7])
	 	src_ofs_max = max(tbl_sts[8])
	 	xctr_2D_max = max(tbl_sts[9])
	 	yctr_2D_max = max(tbl_sts[10])
	 	ampl_2D_max = max(tbl_sts[11])
	 	amax_2D_max = max(tbl_sts[12])
	 	sgmx_2D_max = max(tbl_sts[13])
	 	sgmy_2D_max = max(tbl_sts[14])
	 	thet_2D_max = max(tbl_sts[15])
	 	ofst_2D_max = max(tbl_sts[16])
	 	tmXn_2D_max = max(tbl_sts[17])

		[Header_Get_Add(Stacked_Cube,'AMS_AVG' ,src_amp_avg   ,header_comment='Synthetic Cubes Source Amplitude AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMN_AVG' ,nse_amp_avg   ,header_comment='Synthetic Cubes Noise Amplitude  AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SNR_AVG' ,snr_amp_avg   ,header_comment='Synthetic Cubes SNR Amplitude    AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGN_AVG' ,sig_nmb_avg   ,header_comment='Synthetic Cubes Sigma chan num   AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGV_AVG' ,sig_vel_avg   ,header_comment='Synthetic Cubes Sigma Vel kms-1  AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'FWH_AVG' ,sig_fwh_avg   ,header_comment='Synthetic Cubes FWHM Vel kms-1   AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OFS_AVG' ,src_ofs_avg   ,header_comment='Synthetic Cubes Offset           AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'XC2_AVG' ,xctr_2D_avg   ,header_comment='Synthetic Cubes Ctr x            AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'YC2_AVG' ,yctr_2D_avg   ,header_comment='Synthetic Cubes Ctr y            AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AP2_AVG' ,ampl_2D_avg   ,header_comment='Synthetic Cubes Amp              AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AM2_AVG' ,amax_2D_avg   ,header_comment='Synthetic Cubes Amp max          AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SX2_AVG' ,sgmx_2D_avg   ,header_comment='Synthetic Cubes Sigma x          AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SY2_AVG' ,sgmy_2D_avg   ,header_comment='Synthetic Cubes Sigma y          AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TH2_AVG' ,thet_2D_avg   ,header_comment='Synthetic Cubes Theta            AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OF2_AVG' ,ofst_2D_avg   ,header_comment='Synthetic Cubes Offset           AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TN2_AVG' ,tmXn_2D_avg   ,header_comment='Synthetic Cubes XNoise           AVG')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMS_MED' ,src_amp_med   ,header_comment='Synthetic Cubes Source Amplitude MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMN_MED' ,nse_amp_med   ,header_comment='Synthetic Cubes Noise Amplitude  MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SNR_MED' ,snr_amp_med   ,header_comment='Synthetic Cubes SNR Amplitude    MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGN_MED' ,sig_nmb_med   ,header_comment='Synthetic Cubes Sigma chan num   MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGV_MED' ,sig_vel_med   ,header_comment='Synthetic Cubes Sigma Vel kms-1  MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'FWH_MED' ,sig_fwh_med   ,header_comment='Synthetic Cubes FWHM Vel kms-1   MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OFS_MED' ,src_ofs_med   ,header_comment='Synthetic Cubes Offset           MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'XC2_MED' ,xctr_2D_med   ,header_comment='Synthetic Cubes Ctr x            MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'YC2_MED' ,yctr_2D_med   ,header_comment='Synthetic Cubes Ctr y            MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AP2_MED' ,ampl_2D_med   ,header_comment='Synthetic Cubes Amp              MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AM2_MED' ,amax_2D_med   ,header_comment='Synthetic Cubes Amp max          MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SX2_MED' ,sgmx_2D_med   ,header_comment='Synthetic Cubes Sigma x          MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SY2_MED' ,sgmy_2D_med   ,header_comment='Synthetic Cubes Sigma y          MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TH2_MED' ,thet_2D_med   ,header_comment='Synthetic Cubes Theta            MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OF2_MED' ,ofst_2D_med   ,header_comment='Synthetic Cubes Offset           MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TN2_MED' ,tmXn_2D_med   ,header_comment='Synthetic Cubes XNoise           MED')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMS_STD' ,src_amp_std   ,header_comment='Synthetic Cubes Source Amplitude STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMN_STD' ,nse_amp_std   ,header_comment='Synthetic Cubes Noise Amplitude  STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SNR_STD' ,snr_amp_std   ,header_comment='Synthetic Cubes SNR Amplitude    STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGN_STD' ,sig_nmb_std   ,header_comment='Synthetic Cubes Sigma chan num   STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGV_STD' ,sig_vel_std   ,header_comment='Synthetic Cubes Sigma Vel kms-1  STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'FWH_STD' ,sig_fwh_std   ,header_comment='Synthetic Cubes FWHM Vel kms-1   STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OFS_STD' ,src_ofs_std   ,header_comment='Synthetic Cubes Offset           STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'XC2_STD' ,xctr_2D_std   ,header_comment='Synthetic Cubes Ctr x            STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'YC2_STD' ,yctr_2D_std   ,header_comment='Synthetic Cubes Ctr y            STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AP2_STD' ,ampl_2D_std   ,header_comment='Synthetic Cubes Amp              STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AM2_STD' ,amax_2D_std   ,header_comment='Synthetic Cubes Amp max          STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SX2_STD' ,sgmx_2D_std   ,header_comment='Synthetic Cubes Sigma x          STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SY2_STD' ,sgmy_2D_std   ,header_comment='Synthetic Cubes Sigma y          STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TH2_STD' ,thet_2D_std   ,header_comment='Synthetic Cubes Theta            STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OF2_STD' ,ofst_2D_std   ,header_comment='Synthetic Cubes Offset           STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TN2_STD' ,tmXn_2D_std   ,header_comment='Synthetic Cubes XNoise           STD')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMS_MIN' ,src_amp_min   ,header_comment='Synthetic Cubes Source Amplitude MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMN_MIN' ,nse_amp_min   ,header_comment='Synthetic Cubes Noise Amplitude  MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SNR_MIN' ,snr_amp_min   ,header_comment='Synthetic Cubes SNR Amplitude    MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGN_MIN' ,sig_nmb_min   ,header_comment='Synthetic Cubes Sigma chan num   MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGV_MIN' ,sig_vel_min   ,header_comment='Synthetic Cubes Sigma Vel kms-1  MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'FWH_MIN' ,sig_fwh_min   ,header_comment='Synthetic Cubes FWHM Vel kms-1   MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OFS_MIN' ,src_ofs_min   ,header_comment='Synthetic Cubes Offset           MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'XC2_MIN' ,xctr_2D_min   ,header_comment='Synthetic Cubes Ctr x            MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'YC2_MIN' ,yctr_2D_min   ,header_comment='Synthetic Cubes Ctr y            MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AP2_MIN' ,ampl_2D_min   ,header_comment='Synthetic Cubes Amp              MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AM2_MIN' ,amax_2D_min   ,header_comment='Synthetic Cubes Amp max          MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SX2_MIN' ,sgmx_2D_min   ,header_comment='Synthetic Cubes Sigma x          MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SY2_MIN' ,sgmy_2D_min   ,header_comment='Synthetic Cubes Sigma y          MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TH2_MIN' ,thet_2D_min   ,header_comment='Synthetic Cubes Theta            MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OF2_MIN' ,ofst_2D_min   ,header_comment='Synthetic Cubes Offset           MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TN2_MIN' ,tmXn_2D_min   ,header_comment='Synthetic Cubes XNoise           MIN')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMS_MAX' ,src_amp_max   ,header_comment='Synthetic Cubes Source Amplitude MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AMN_MAX' ,nse_amp_max   ,header_comment='Synthetic Cubes Noise Amplitude  MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SNR_MAX' ,snr_amp_max   ,header_comment='Synthetic Cubes SNR Amplitude    MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGN_MAX' ,sig_nmb_max   ,header_comment='Synthetic Cubes Sigma chan num   MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SGV_MAX' ,sig_vel_max   ,header_comment='Synthetic Cubes Sigma Vel kms-1  MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'FWH_MAX' ,sig_fwh_max   ,header_comment='Synthetic Cubes FWHM Vel kms-1   MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OFS_MAX' ,src_ofs_max   ,header_comment='Synthetic Cubes Offset           MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'XC2_MAX' ,xctr_2D_max   ,header_comment='Synthetic Cubes Ctr x            MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'YC2_MAX' ,yctr_2D_max   ,header_comment='Synthetic Cubes Ctr y            MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AP2_MAX' ,ampl_2D_max   ,header_comment='Synthetic Cubes Amp              MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'AM2_MAX' ,amax_2D_max   ,header_comment='Synthetic Cubes Amp max          MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SX2_MAX' ,sgmx_2D_max   ,header_comment='Synthetic Cubes Sigma x          MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'SY2_MAX' ,sgmy_2D_max   ,header_comment='Synthetic Cubes Sigma y          MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TH2_MAX' ,thet_2D_max   ,header_comment='Synthetic Cubes Theta            MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'OF2_MAX' ,ofst_2D_max   ,header_comment='Synthetic Cubes Offset           MAX')       for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,'TN2_MAX' ,tmXn_2D_max   ,header_comment='Synthetic Cubes XNoise           MAX')       for Stacked_Cube in OPT_STCK_FLS]
	else:
		pass

	return OPT_STCK_FLS

def Cube_Stack_2D(CCubes2bStacked,name,wght_img_2bstack,sig_clp,*args, **kwargs):
	wrt_fits         = kwargs.get('wrt_fits'       ,True)
	pst_msk          = kwargs.get('pst_msk'        ,False)
	pst_smt          = kwargs.get('pst_smt'        ,False)
	pst_cnt          = kwargs.get('pst_cnt'        ,False)
	stack_ext        = kwargs.get('stack_ext'      ,None)
	new_CRVAL1_head  = kwargs.get('new_CRVAL1_head',None)
	new_CDELT1_head  = kwargs.get('new_CDELT1_head',None)
	smt_spc_pst      = kwargs.get('smt_spc_pst'    ,False)
	smooth_shape     = kwargs.get('smooth_shape'   ,'gaussian')
	wght_type        = kwargs.get('wght_type'      ,None)
	wcs              = kwargs.get('wcs'            ,None)
	sufix            = kwargs.get('sufix'          ,'')
	freq_obs_f       = kwargs.get('freq_obs_f'     ,99999)

	stack_lite       = kwargs.get('stack_lite'     ,True)

	spc_wdt_dir      = kwargs.get('spc_wdt_dir'    ,500)

	cp_bs_hdrs       = kwargs.get('cp_bs_hdrs'     ,False)

	stt_var			 = kwargs.get('stt_var',False)
	stt_mst_tbl		 = kwargs.get('stt_mst_tbl',None)
	stt_hdr			 = kwargs.get('stt_hdr',None)

	lnw_wdt			 = kwargs.get('lnw_wdt',None)


	img_2bstack    = [apgtdt(img,memmap=False) for img in CCubes2bStacked]
	wcs            = kwargs.get('wcs'            ,apwcs(CCubes2bStacked[0]))
	try:
		wcs       = wcs.dropaxis(3) 
	except IndexError:
		pass

	print
	print 'Number of galaxies to be stacked (histogram): ',len(img_2bstack)

	if sig_clp == True:
		img_flt       = astropy.stats.sigma_clip(img_2bstack,sigma=sigma_cut,axis=0,iters=None,cenfunc=sigma_cen_fct, copy=True)

		print
		print colored('Sigma-clipping for stacking!','yellow')
		print colored('Sigma Cut                    : ' + str(sigma_cut),'yellow')
		print colored('Central function             : ' + str(sigma_cen_fct), 'yellow')
		print colored('Central Value for clipping   : ' + str(sigma_cen_fct),'yellow')

		img_flt.set_fill_value(sigma_msk_fill_val)
		img_flt_filled = img_flt.filled()
		img_stat       = img_flt_filled
	elif sig_clp == False:
		img_stat   = img_2bstack

	print 
	print np.asarray(img_stat).shape
	img_stat = np.squeeze(img_stat) 
	print
	print np.asarray(img_stat).shape

	wght_img_copy = wght_img_2bstack
	wght_img_stat = wght_img_2bstack 
	wght_img_stat = np.asarray(wght_img_stat)
	img_staw      = []

	img_stat_smw_f = []
	[img_staw.append(np.asarray(img_stat)[j]*np.asarray(wght_img_stat)[j]) for j in range(len(wght_img_stat))]
	img_staw      = np.asarray(img_staw)
	[img_stat_smw_f.append(np.divide(np.asarray(img_staw)[j],np.asarray(img_stat)[j])) for j in range(len(img_stat))]

	print
	print colored('Original shape                                               : '+str(np.asarray(img_stat).shape),'cyan')
	img_stat = np.squeeze(img_stat) 
	N,l,m  = np.asarray(img_stat).shape 
	print colored('Squeezed useless extra dimensions                            : '+str(np.asarray(img_stat).shape),'cyan')
	print colored('Dimension Numer of Cubes, X size, Y size : '+str(N)+', '+str(l)+', '+str(m),'cyan')
	print

	img_res_sum = bn.nansum(np.array(img_stat)             , axis=0)
	img_res_avg = bn.nanmean(np.array(img_stat)            , axis=0)
	img_res_med = bn.nanmedian(np.array(img_stat)          , axis=0)

	print
	print colored('Sum, Mean, Median : Stacked data cubes OK','yellow')
	print 

	if stack_lite == False:
		#BEGINS HISTO
		img_stat_hst_y = []
		img_stat_hsw_y = []
		pb = ProgressBar(l)
		for y_dim in range(l):
			pb.update()
			Y_ROW = np.asarray(img_stat)[:,y_dim,:]
			Transpose  = np.asarray(Y_ROW).T
			Transposw  = np.asarray(Y_ROW).T
			img_stat_hst_x = []
			img_stat_hsw_x = []
			for x_dim in range(len(Transpose)):
				if np.isnan(sigma_msk_fill_val) == True:
					non_msk_num = int(np.count_nonzero(~np.isnan(Transpose[x_dim])))
					msk_num     = int(np.count_nonzero(np.isnan(Transpose[x_dim])))
					img_stat_hst_x.append(float(non_msk_num))

					non_msk_num_wghts = int(np.count_nonzero(~np.isnan(Transposw[x_dim])))
					msk_num_wghts     = int(np.count_nonzero(np.isnan(Transposw[x_dim])))
					img_stat_hsw_x.append(float(non_msk_num_wghts))

				elif np.isnan(sigma_msk_fill_val) == False:
					pass
					non_msk_num = int(np.count_nonzero(Transpose[x_dim]!=sigma_msk_fill_val))
					img_stat_hst_x.append(float(non_msk_num))

					non_msk_num_wghts = int(np.count_nonzero(Transposw[x_dim]!=sigma_msk_fill_val))
					img_stat_hsw_x.append(float(non_msk_num_wghts))
				else:
					pass
			
			img_stat_hst_x = np.reshape(img_stat_hst_x,(m))
			img_stat_hsw_x = np.reshape(img_stat_hsw_x,(m))

			img_stat_hst_y.append(img_stat_hst_x)
			img_stat_hsw_y.append(img_stat_hsw_x)
			#ENDS HISTO

		img_sts_hst = np.asarray(img_stat_hst_y)
		img_res_std = bn.nanstd(np.array(img_stat), axis=0)
		print
		print colored('Histogram, Std: Stacked data cubes OK','yellow')
		print 

		img_res_suw_pre = np.asarray(bn.nansum(np.array(img_staw)                , axis=0))
		img_sts_wsu_pre = np.asarray(bn.nansum(np.array(img_stat_smw_f)          , axis=0))

		img_sts_wsu_pre = np.squeeze(img_sts_wsu_pre)
		img_res_suw_pre = np.squeeze(img_res_suw_pre)

		print
		print colored('Weights Sum Weighted Sum pre computations: OK','yellow')
		print
		print img_res_suw_pre.shape

		img_sts_hsw = data=np.asarray(img_stat_hsw_y)                           
		img_sts_wsu = data=img_sts_wsu_pre                                      
		img_res_suw = data=img_res_suw_pre                                      
		img_res_avw = data=img_res_suw_pre.astype(float)/img_sts_wsu_pre.astype(float) 
		
		print
		print colored('SW Histogram, Sum of weights, Weighted Sum: Stacked data cubes OK','yellow')
		print

		img_res_1sl = np.nanpercentile(np.array(img_stat), 15.9, axis=0)
		img_res_1sh = np.nanpercentile(np.array(img_stat), 84.1, axis=0)
		img_res_2sl = np.nanpercentile(np.array(img_stat), 2.30, axis=0)
		img_res_2sh = np.nanpercentile(np.array(img_stat), 97.7, axis=0)
		img_res_3sl = np.nanpercentile(np.array(img_stat), 0.20, axis=0)
		img_res_3sh = np.nanpercentile(np.array(img_stat), 99.8, axis=0)
		img_res_p25 = np.nanpercentile(np.array(img_stat), 25.0, axis=0)
		img_res_p75 = np.nanpercentile(np.array(img_stat), 75.0, axis=0)

		print 'Stacked images through : sum, mean, median, and percentiles: '
		print '17., 83.0, (1 sigma)'
		print '2.5, 97.5, (2 sigma)'
		print '0.5, 99.5, (3 sigma)'
		print '25., 75.0, (interquantile)'
		print		
		print colored('Percentiles: Stacked data cubes OK','yellow')
		print 
	elif stack_lite == True:
		pass

	bs_func = kwargs.get('bs_func','')

	if wrt_fits==True:
		if  '-BS-' in name:
			print (colored(name,'yellow'))
			spc_dir_dst = str_bst_stk + str(spc_wdt_dir) +'/'
			if os.path.exists(spc_dir_dst)==False:
				print
				print (colored('Stacked width directory does not exist!','yellow'))
				print (colored('Creating it!','yellow'))
				print
				os.makedirs(spc_dir_dst)
			else:
				pass
		elif  '-BS_MST' in name:
			print (colored(name,'yellow'))
			spc_dir_dst = stt_bst_stk + str(spc_wdt_dir) +'/'
			if os.path.exists(spc_dir_dst)==False:
				print
				print (colored('Stacked width directory does not exist!','yellow'))
				print (colored('Creating it!','yellow'))
				print
				os.makedirs(spc_dir_dst)
			else:
				pass
		else:
			spc_dir_dst = stk_dir_res + str(spc_wdt_dir) +'/'
			if os.path.exists(spc_dir_dst)==False:
				print
				print (colored('Stacked width directory does not exist!','yellow'))
				print (colored('Creating it!','yellow'))
				print
				os.makedirs(spc_dir_dst)
			else:
				pass

		spec_file_sum_ofn = spc_dir_dst + str(name) + bs_func + '-stk-sum-' + str(sufix) + '.fits'
		spec_file_avg_ofn = spc_dir_dst + str(name) + bs_func + '-stk-avg-' + str(sufix) + '.fits'
		spec_file_med_ofn = spc_dir_dst + str(name) + bs_func + '-stk-med-' + str(sufix) + '.fits'
		spec_file_hst_ofn = spc_dir_dst + str(name) + bs_func + '-stk-hst-' + str(sufix) + '.fits'
		spec_file_std_ofn = spc_dir_dst + str(name) + bs_func + '-stk-std-' + str(sufix) + '.fits'
		spec_file_p25_ofn = spc_dir_dst + str(name) + bs_func + '-stk-p25-' + str(sufix) + '.fits'
		spec_file_p75_ofn = spc_dir_dst + str(name) + bs_func + '-stk-p75-' + str(sufix) + '.fits'
		spec_file_1sl_ofn = spc_dir_dst + str(name) + bs_func + '-stk-1sl-' + str(sufix) + '.fits'
		spec_file_1sh_ofn = spc_dir_dst + str(name) + bs_func + '-stk-1sh-' + str(sufix) + '.fits'
		spec_file_2sl_ofn = spc_dir_dst + str(name) + bs_func + '-stk-2sl-' + str(sufix) + '.fits'
		spec_file_2sh_ofn = spc_dir_dst + str(name) + bs_func + '-stk-2sh-' + str(sufix) + '.fits'
		spec_file_3sl_ofn = spc_dir_dst + str(name) + bs_func + '-stk-3sl-' + str(sufix) + '.fits'
		spec_file_3sh_ofn = spc_dir_dst + str(name) + bs_func + '-stk-3sh-' + str(sufix) + '.fits'
		
		spec_file_hsw_ofn = spc_dir_dst + str(name) + bs_func + '-stk-hsw-' + str(sufix) + '.fits'
		spec_file_wsu_ofn = spc_dir_dst + str(name) + bs_func + '-stk-wsu-' + str(sufix) + '.fits'
		spec_file_suw_ofn = spc_dir_dst + str(name) + bs_func + '-stk-suw-' + str(sufix) + '.fits'
		spec_file_avw_ofn = spc_dir_dst + str(name) + bs_func + '-stk-avw-' + str(sufix) + '.fits'

		spec_file_sum     = Wrt_FITS_File(img_res_sum,spec_file_sum_ofn)
		spec_file_avg     = Wrt_FITS_File(img_res_avg,spec_file_avg_ofn)
		spec_file_med     = Wrt_FITS_File(img_res_med,spec_file_med_ofn)

		if stack_lite == False:
			spec_file_hst     = Wrt_FITS_File(img_sts_hst,spec_file_hst_ofn)
			spec_file_std     = Wrt_FITS_File(img_res_std,spec_file_std_ofn)

			spec_file_p25     = Wrt_FITS_File(img_res_p25,spec_file_p25_ofn)
			spec_file_p75     = Wrt_FITS_File(img_res_p75,spec_file_p75_ofn)
			spec_file_1sl     = Wrt_FITS_File(img_res_1sl,spec_file_1sl_ofn)
			spec_file_1sh     = Wrt_FITS_File(img_res_1sh,spec_file_1sh_ofn)
			spec_file_2sl     = Wrt_FITS_File(img_res_2sl,spec_file_2sl_ofn)
			spec_file_2sh     = Wrt_FITS_File(img_res_2sh,spec_file_2sh_ofn)
			spec_file_3sl     = Wrt_FITS_File(img_res_3sl,spec_file_3sl_ofn)
			spec_file_3sh     = Wrt_FITS_File(img_res_3sh,spec_file_3sh_ofn)

			spec_file_hsw     = Wrt_FITS_File(img_sts_hsw,spec_file_hsw_ofn)
			spec_file_wsu     = Wrt_FITS_File(img_sts_wsu,spec_file_wsu_ofn)
			spec_file_suw     = Wrt_FITS_File(img_res_suw,spec_file_suw_ofn)
			spec_file_avw     = Wrt_FITS_File(img_res_avw,spec_file_avw_ofn)

			OPT_STCK_FLS = [spec_file_sum_ofn,spec_file_avg_ofn,spec_file_med_ofn,spec_file_hst_ofn,
			spec_file_std_ofn,
			spec_file_p25_ofn,spec_file_p75_ofn,
			spec_file_1sl_ofn,spec_file_1sh_ofn,
			spec_file_2sl_ofn,spec_file_2sh_ofn,
			spec_file_3sl_ofn,spec_file_3sh_ofn,
			spec_file_hsw_ofn,spec_file_wsu_ofn,spec_file_suw_ofn,spec_file_avw_ofn]
		elif stack_lite == True:
			OPT_STCK_FLS = [spec_file_sum_ofn,spec_file_avg_ofn,spec_file_med_ofn]
		[Header_Updt(spec_sts_res,'STK_NUM' ,len(img_2bstack), header_comment = 'Number of galaxies used for Stack') for spec_sts_res in OPT_STCK_FLS]
	else:
		pass


	print 'Images Stacked files names: '
	print
	print colored(spec_file_sum_ofn,'cyan')
	print colored(spec_file_avg_ofn,'cyan')
	print colored(spec_file_med_ofn,'cyan')
	if stack_lite == False:
		print colored(spec_file_hst_ofn,'cyan')
		print colored(spec_file_std_ofn,'cyan')
		print colored(spec_file_p25_ofn,'cyan')
		print colored(spec_file_p75_ofn,'cyan')
		print colored(spec_file_1sl_ofn,'cyan')
		print colored(spec_file_1sh_ofn,'cyan')
		print colored(spec_file_2sl_ofn,'cyan')
		print colored(spec_file_2sh_ofn,'cyan')
		print colored(spec_file_3sl_ofn,'cyan')
		print colored(spec_file_3sh_ofn,'cyan')
		
		print colored(spec_file_hsw_ofn,'yellow')
		print colored(spec_file_avw_ofn,'yellow')
		print colored(spec_file_suw_ofn,'yellow')
	elif stack_lite == True:
		pass

	if stack_lite == True:
		FNL_SPEC_RES = [spec_file_med,spec_file_avg,spec_file_sum]
	elif stack_lite == False:
		FNL_SPEC_RES = [
					spec_file_med,spec_file_avg,spec_file_sum,spec_file_std,
					spec_file_hst,
					spec_file_1sl,spec_file_1sh,
					spec_file_2sl,spec_file_2sh,
					spec_file_3sl,spec_file_3sh,
					spec_file_p25,spec_file_p75,
					spec_file_hsw,spec_file_wsu,spec_file_suw,spec_file_avw]
	if cp_bs_hdrs == True:
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'BSCALE')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'BZERO')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'BMAJ')     for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'BMIN')     for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'BPA')      for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'BTYPE')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'EQUINOX')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'RADESYS')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'BUNIT')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'RADESYS')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'LONPOLE')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'LATPOLE')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC1_1')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC2_1')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC3_1')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC1_2')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC2_2')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC3_2')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC1_3')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC2_3')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PC3_3')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CTYPE1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CRVAL1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CDELT1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CRPIX1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CUNIT1')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CTYPE2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CRVAL2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CDELT2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CRPIX2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CUNIT2')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CTYPE3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CRVAL3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CDELT3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CRPIX3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'CUNIT3')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PV2_1')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'PV2_2')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'RESTFRQ')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'SPECSYS')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'ALTRVAL')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'ALTRPIX')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'VELREF')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'TELESCOP') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'OBSERVER') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'DATE-OBS') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'TIMESYS')  for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'OBSRA')    for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'OBSDEC')   for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'OBSGEO-X') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'OBSGEO-Y') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'OBSGEO-Z') for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'DATE')     for stk_res_flr in OPT_STCK_FLS]
		[Header_Copy(stk_res_flr,CCubes2bStacked[0],'ORIGIN')   for stk_res_flr in OPT_STCK_FLS]
	else:
		pass
	if stt_var == True:
		print
		print (colored('Adding stat to fits headers!','yellow'))
		print
		tbl_sts = Table_Ipt_Cat_Stats(stt_mst_tbl,stt_hdr)
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][0] ,tbl_sts[1][0] ,header_comment='Redshift Average')                         for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][1] ,tbl_sts[1][1] ,header_comment='Redshift Median')                          for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][2] ,tbl_sts[1][2] ,header_comment='Redshift 1 sgm lw lmt 15.9 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][3] ,tbl_sts[1][3] ,header_comment='Redshift 1 sgm hg lmt 84.1 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][4] ,tbl_sts[1][4] ,header_comment='Redshift 2 sgm lw lmt 2.30 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][5] ,tbl_sts[1][5] ,header_comment='Redshift 2 sgm hg lmt 97.7 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][6] ,tbl_sts[1][6] ,header_comment='Redshift 3 sgm lw lmt 0.20 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][7] ,tbl_sts[1][7] ,header_comment='Redshift 3 sgm hg lmt 99.8 pct')           for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][8] ,tbl_sts[1][8] ,header_comment='Redshift 25 pct')                          for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][9] ,tbl_sts[1][9] ,header_comment='Redshift 75 pct')                          for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][10],tbl_sts[1][10],header_comment=str(tbl_sts[2]) + ' Average')               for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][11],tbl_sts[1][11],header_comment=str(tbl_sts[2]) + ' Median')                for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][12],tbl_sts[1][12],header_comment=str(tbl_sts[2]) + ' 1 sgm lw lmt 15.9 pct') for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][13],tbl_sts[1][13],header_comment=str(tbl_sts[2]) + ' 1 sgm hg lmt 84.1 pct') for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][14],tbl_sts[1][14],header_comment=str(tbl_sts[2]) + ' 2 sgm lw lmt 2.30 pct') for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][15],tbl_sts[1][15],header_comment=str(tbl_sts[2]) + ' 2 sgm hg lmt 97.7 pct') for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][16],tbl_sts[1][16],header_comment=str(tbl_sts[2]) + ' 3 sgm lw lmt 0.20 pct') for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][17],tbl_sts[1][17],header_comment=str(tbl_sts[2]) + ' 3 sgm hg lmt 99.8 pct') for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][18],tbl_sts[1][18],header_comment=str(tbl_sts[2]) + ' 25 pct')                for Stacked_Cube in OPT_STCK_FLS]
		[Header_Get_Add(Stacked_Cube,tbl_sts[0][19],tbl_sts[1][19],header_comment=str(tbl_sts[2]) + ' 75 pct')                for Stacked_Cube in OPT_STCK_FLS]		
	else:
		pass
	return OPT_STCK_FLS
####Fnc_Syn_Stk####