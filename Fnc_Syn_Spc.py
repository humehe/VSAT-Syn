import sys, os
import bottleneck as bn

from regions.core import PixCoord
from regions.shapes.circle import CirclePixelRegion

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse

from matplotlib import rc
font = {'family' : 'serif'}
rc('font', **font)

from astropy import modeling as apmd

from scipy import integrate as scpint
from scipy import optimize as scpopt

from Fnc_Syn_Dir import *
from Fnc_Syn_Fts import *
from Fnc_Syn_Mth import *

from Fnc_Syn_Plt import align_yaxis, ScaledFormatter

####Fnc_Syn_Spc####
def fit_2D_Gaussian(Cube2bFit,*args,**kwargs):
	slc_nmb  = kwargs.get('slc_nmb' ,0)
	dest_dir = kwargs.get('dest_dir',None)
	verbose  = kwargs.get('verbose' ,None)
	clp_fnc  = kwargs.get('clp_fnc' ,'sum')
	circular = kwargs.get('circular',True)
	x_ref    = kwargs.get('x_ref',0)
	y_ref    = kwargs.get('y_ref',0)

	dest_dir_plt = kwargs.get('dest_dir',None)
	dest_dir_clp = kwargs.get('dest_dir',None)

	z_avg     = kwargs.get('z_avg',Header_Get(Cube2bFit,'STZ_AVG'))
	z_med     = kwargs.get('z_med',Header_Get(Cube2bFit,'STZ_MED'))
	frq_r     = kwargs.get('frq_r',Header_Get(Cube2bFit,'RESTFRQ'))
	z_f2l     = z_med

	prefix    = kwargs.get('prefix','')

	if dest_dir_clp != None:
		Cube2bclp_2D_opt = dest_dir_clp + prefix + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+clp_fnc+'.fits'
	elif dest_dir_clp == None:
		Cube2bclp_2D_opt = stp_dir_res  + prefix + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+clp_fnc+'.fits'

	Cube_Info = Cube_Header_Get(Cube2bFit,frq_r* u.Hz)
	FRQ_AXS   = Cube_Info[16].value
	VEL_AXS   = Cube_Info[17].value

	scale_deg         = Header_Get(Cube2bFit,'CDELT2')
	scale_arcsec      = scale_deg*3600

	if dest_dir != None:
		PLOTFILENAME = str(dest_dir)  + '/' + prefix + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'.pdf'
	elif dest_dir == None:
		PLOTFILENAME = plt_dir_res    + '/' + prefix + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'.pdf'

	cube_data    = np.asarray(apgtdt(Cube2bFit,memmap=False) )

	slice_fwhm = (Header_Get(Cube2bFit,'FTS_FWH'))
	slice_cwdt = (Header_Get(Cube2bFit,'STT_VEL'))
	slice_nmbr = (Header_Get(Cube2bFit,'MAX_SNS')) 

	if slc_nmb == None:
		if clp_fnc == 'sum':
			slice_nmbr = (Header_Get(Cube2bFit,'MAX_SNS')) 
		elif clp_fnc == 'med':
			slice_nmbr = (Header_Get(Cube2bFit,'MAX_SNM')) 
		elif clp_fnc == 'avg':
			slice_nmbr = (Header_Get(Cube2bFit,'MAX_SNA')) 
	else:
		pass

	try:
		slice_wdnb = int(np.ceil(slice_fwhm / slice_cwdt))
		slice_nblw = int(slice_nmbr-int(np.ceil(slice_fwhm / slice_cwdt))) 
		slice_nbhg = int(slice_nmbr+int(np.ceil(slice_fwhm / slice_cwdt))) 
		slice_vllw = VEL_AXS[slice_nblw]
		slice_vlhg = VEL_AXS[slice_nbhg]
	except:
		slice_nblw = 0
		slice_nbhg = -1
		slice_vllw = 0
		slice_vlhg = -1

	if slc_nmb != None:
		data_2b_plot = cube_data[slc_nmb]
		Message1 = 'Fitting gaussian with slice number : ' + str(slc_nmb+1)
		Message2 = 'For datacube : ' + Cube2bFit
		plt_tlt = 'Slice: ' + str(slc_nmb+1) + '-' + str(round(VEL_AXS[slc_nmb],0)) + ' km/s'
	elif slc_nmb == None:
		Message1 = 'Fitting gaussian through cube collapse ('+str(clp_fnc)+')'
		Message2 = 'For datacube : ' + Cube2bFit
		plt_tlt = '2D Collapse ('+str(clp_fnc).upper()+') VW:' + str(int(slice_fwhm)) + 'km/s ['+ str(int(slice_nmbr+1))+ '$\pm$' + str(int(slice_wdnb)) + ']'
		data_2b_plt = np.asarray(apgtdt(Cube2bFit,memmap=False))
		data_2b_plt_clp = data_2b_plt[slice_nblw:slice_nbhg]
		if clp_fnc == 'sum':
			data_2b_plot = np.asarray(np.nansum(np.array(data_2b_plt_clp)   , axis=0))
			clp_hdr = 'S'  
			clp_hdc = 'SUM'  
			slice_nmbr = (Header_Get(Cube2bFit,'MAX_SNS')) 
		elif clp_fnc == 'med':
			data_2b_plot = np.asarray(np.nanmedian(np.array(data_2b_plt_clp), axis=0))
			clp_hdr = 'M'  
			clp_hdc = 'MED'  
			slice_nmbr = (Header_Get(Cube2bFit,'MAX_SNM')) 
		elif clp_fnc == 'avg':
			data_2b_plot = np.asarray(np.nanmean(np.array(data_2b_plt_clp)  , axis=0))
			clp_hdr = 'A'
			clp_hdc = 'AVG'
			slice_nmbr = (Header_Get(Cube2bFit,'MAX_SNA')) 
		Wrt_FITS_File(data_2b_plot,Cube2bclp_2D_opt)

		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'XXT_MIN',header_comment = 'Image Extent X MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'XXT_MAX',header_comment = 'Image Extent X MAX')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'YXT_MIN',header_comment = 'Image Extent Y MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'YXT_MAX',header_comment = 'Image Extent Y MAX')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'XCT_FIT',header_comment = 'Image Center X ')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'YCT_FIT',header_comment = 'Image Center Y ')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'RAD_EXT',header_comment = 'Image Extent Radii')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STK_NUM',header_comment = 'Number of galaxies used for Stack')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_AVG',header_comment = 'Redshift Average ')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_MED',header_comment = 'Redshift Median ')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_1SL',header_comment = 'Redshift 1 sgm lw lmt 15.9 pct') 
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_1SH',header_comment = 'Redshift 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_2SL',header_comment = 'Redshift 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_2SH',header_comment = 'Redshift 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_3SL',header_comment = 'Redshift 3 sgm lw lmt 0.20 pct')  
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_3SH',header_comment = 'Redshift 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_P25',header_comment = 'Redshift 25 pct')                   
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_P75',header_comment = 'Redshift 75 pct')                    

		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_AVG',header_comment = str(Splt_Hdr_Cmt) + ' Average')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_MED',header_comment = str(Splt_Hdr_Cmt) + ' Median')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_1SL',header_comment = str(Splt_Hdr_Cmt) + ' 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_1SH',header_comment = str(Splt_Hdr_Cmt) + ' 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_2SL',header_comment = str(Splt_Hdr_Cmt) + ' 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_2SH',header_comment = str(Splt_Hdr_Cmt) + ' 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_3SL',header_comment = str(Splt_Hdr_Cmt) + ' 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_3SH',header_comment = str(Splt_Hdr_Cmt) + ' 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_P25',header_comment = str(Splt_Hdr_Cmt) + ' 25 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_P75',header_comment = str(Splt_Hdr_Cmt) + ' 75 pct')

		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_VEL',header_comment = 'CbeWth [km/s]')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_FLS',header_comment = 'TFlx SUM All Chns')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TFL',header_comment = 'TFlx SUM All Chns * CbeWth')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LMS',header_comment = 'TFlx SUM All CH 2 Lum')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LLM',header_comment = 'TFlx SUM All CH 2 Lum [log]')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LMT',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LLT',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log]')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_FLE',header_comment = 'TFlx SUM All Chns Err')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TFE',header_comment = 'TFlx SUM All Chns * CbeWth Err')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SL1',header_comment = 'TFlx SUM All CH 2 Lum Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SH1',header_comment = 'TFlx SUM All CH 2 Lum Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LL1',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LH1',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_ML1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_MH1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TL1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TH1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SL2',header_comment = 'TFlx SUM All CH 2 Lum Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SH2',header_comment = 'TFlx SUM All CH 2 Lum Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LL2',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LH2',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_ML2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_MH2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TL2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TH2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SL3',header_comment = 'TFlx SUM All CH 2 Lum Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SH3',header_comment = 'TFlx SUM All CH 2 Lum Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LL3',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LH3',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_ML3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_MH3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TL3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TH3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SNS',header_comment = 'Vel Prf Max Chn Location SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SVS',header_comment = 'Vel Prf Max Frequency Location [Hz] SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_VLS',header_comment = 'Vel Prf Max Chn Value SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SNA',header_comment = 'Vel Prf Max Chn Location AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SVA',header_comment = 'Vel Prf Max Frequency Location [Hz] AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_VLA',header_comment = 'Vel Prf Max Chn Value AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SNM',header_comment = 'Vel Prf Max Chn Location MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SVM',header_comment = 'Vel Prf Max Frequency Location [Hz] MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_VLM',header_comment = 'Vel Prf Max Chn Value MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_AMP',header_comment = '1DGF Amplitude AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_CTR',header_comment = '1DGF Center AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_SIG',header_comment = '1DGF Sigma AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_FWH',header_comment = '1DGF FWHM  AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_A2A',header_comment = '1DGF Area A AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_A2M',header_comment = '1DGF Area M AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LUM',header_comment = '1DGF Ar2Lum AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LLM',header_comment = '1DGF Ar2Lum [log] AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_APE',header_comment = '1DGF Amplitude Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_CTE',header_comment = '1DGF Center Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_SGE',header_comment = '1DGF Sigma Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_FWE',header_comment = '1DGF FWHM  Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_AAE',header_comment = '1DGF Area A Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_AME',header_comment = '1DGF Area M Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_ML1',header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_MH1',header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LL1',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LH1',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_ML2',header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_MH2',header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LL2',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LH2',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_ML3',header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_MH3',header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LL3',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LH3',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_ML1',header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_MH1',header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LL1',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LH1',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_ML2',header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_MH2',header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LL2',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LH2',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_ML3',header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_MH3',header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LL3',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LH3',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_AMP',header_comment = '1DGF Amplitude MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_CTR',header_comment = '1DGF Center MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_SIG',header_comment = '1DGF Sigma MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_FWH',header_comment = '1DGF FWHM  MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_A2A',header_comment = '1DGF Area A MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_A2M',header_comment = '1DGF Area M MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LUM',header_comment = '1DGF Ar2Lum MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LLM',header_comment = '1DGF Ar2Lum [log] MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_APE',header_comment = '1DGF Amplitude Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_CTE',header_comment = '1DGF Center Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_SGE',header_comment = '1DGF Sigma Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_FWE',header_comment = '1DGF FWHM  Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_AAE',header_comment = '1DGF Area A Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_AME',header_comment = '1DGF Area M Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_ML1',header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_MH1',header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LL1',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LH1',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_ML2',header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_MH2',header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LL2',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LH2',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_ML3',header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_MH3',header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LL3',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LH3',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_ML1',header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_MH1',header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LL1',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LH1',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_ML2',header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_MH2',header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LL2',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LH2',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_ML3',header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_MH3',header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LL3',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LH3',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_AMP',header_comment = '1DGF Amplitude SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_CTR',header_comment = '1DGF Center SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_SIG',header_comment = '1DGF Sigma SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_FWH',header_comment = '1DGF FWHM  SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_A2A',header_comment = '1DGF Area A SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_A2M',header_comment = '1DGF Area M SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LUM',header_comment = '1DGF Ar2Lum SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LLM',header_comment = '1DGF Ar2Lum [log] SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_APE',header_comment = '1DGF Amplitude Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_CTE',header_comment = '1DGF Center Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_SGE',header_comment = '1DGF Sigma Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_FWE',header_comment = '1DGF FWHM Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_AAE',header_comment = '1DGF Area A Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_AME',header_comment = '1DGF Area M Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_ML1',header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_MH1',header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LL1',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LH1',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_ML2',header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_MH2',header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LL2',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LH2',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_ML3',header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_MH3',header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LL3',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LH3',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_ML1',header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_MH1',header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LL1',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LH1',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_ML2',header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_MH2',header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LL2',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LH2',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_ML3',header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_MH3',header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LL3',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LH3',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm hg lmt 99.8 pct')

		print
		print (colored('Input Cube for collapse : ' + Cube2bFit,'magenta'))
		print (colored('Resulting Collapsed Cube: ' + Cube2bclp_2D_opt,'yellow'))

	nx_f2DG, ny_f2DG = data_2b_plot.shape
	nx,ny            = nx_f2DG,ny_f2DG

	X0_f2DG     = kwargs.get('X0_f2DG',int(np.ceil(nx_f2DG/2)))
	Y0_f2DG     = kwargs.get('Y0_f2DG',int(np.ceil(ny_f2DG/2)))
	A_f2DG      = kwargs.get('A_f2DG',1)
	SIGMAX_f2DG = kwargs.get('SIGMAX_f2DG',1)
	SIGMAY_f2DG = kwargs.get('SIGMAY_f2DG',1)
	THETA_f2DG  = kwargs.get('THETA_f2DG',0)
	OFS_f2DG    = kwargs.get('OFS_f2DG',0)
	displ_s_f   = kwargs.get('displ_s_f',False)
	verbose     = kwargs.get('verbose',False)

	# Create x and y indices
	x    = np.linspace(0, nx_f2DG, nx_f2DG)-0.5
	y    = np.linspace(0, ny_f2DG, ny_f2DG)-0.5
	x, y = np.meshgrid(x, y)

	data = data_2b_plot

	initial_guess = (X0_f2DG,Y0_f2DG,A_f2DG,SIGMAX_f2DG,SIGMAY_f2DG,THETA_f2DG,OFS_f2DG)

	xdata       = np.vstack((x.ravel(),y.ravel()))
	ydata       = data.ravel()
	try:
		popt, pcov  = scpopt.curve_fit(func_2D_Gaussian, xdata, ydata, 
					p0=initial_guess,
					bounds=([X0_f2DG-0.001,Y0_f2DG-0.001,-np.inf,-(SIGMAX_f2DG/2.)*2,-(SIGMAX_f2DG/2.)*2,-np.inf,-np.inf],
							[X0_f2DG+0.001,Y0_f2DG+0.001, np.inf, (SIGMAY_f2DG/2.)*2, (SIGMAY_f2DG/2.)*2, np.inf, np.inf]))

		perr        = np.sqrt(np.diag(pcov))
		data_fitted = func_2D_Gaussian((x, y), *popt, circular=circular)
		fit_res     = 'OK'
		X0_F        = np.round(popt[0],0)
		Y0_F        = np.round(popt[1],0)
		X_DIF       = np.round(X0_F,0) - X0_f2DG
		Y_DIF       = np.round(Y0_F,0) - Y0_f2DG


	except RuntimeError:
		popt, pcov  = [0,0,0,0,0,0,0],[0,0,0,0,0,0,0]
		perr        = [0,0,0,0,0,0,0]
		X0_F        = 0
		Y0_F        = 0
		X_DIF       = 0
		Y_DIF       = 0
		data_fitted = func_2D_Gaussian((x, y), *popt, circular=circular)
		fit_res     = 'ERR'
		print("Error - curve_fit failed")

	X0_F         = np.round(popt[0],0)
	Y0_F         = np.round(popt[1],0)
	A_F          = np.round(popt[2],9)
	SIGMAX_F     = np.round(popt[3],9)
	SIGMAY_F     = np.round(popt[4],9)
	THETA_F      = np.round(popt[5],9)
	OFFSET_F     = np.round(popt[6],9)

	X0_E         = np.round(perr[0],9)
	Y0_E         = np.round(perr[1],9)
	A_E          = np.round(perr[2],9)
	SIGMAX_E     = np.round(perr[3],9)
	SIGMAY_E     = np.round(perr[4],9)
	THETA_E      = np.round(perr[5],9)
	OFFSET_E     = np.round(perr[6],9)

	if circular ==True:
		SIGMAY_F = SIGMAX_F
		SIGMAY_E = SIGMAX_E
	elif circular == False:
		pass

	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CNI',slice_nblw ,header_comment = 'Initial Chn used for Collapse Num ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CNL',slice_nbhg ,header_comment = 'Last Chn used for Collapse Num ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CVI',slice_vllw ,header_comment = 'Initial Chn used for Collapse Vel ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CVL',slice_vlhg ,header_comment = 'Last Chn used for Collapse Vel ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_XCT',X0_F       ,header_comment = '2DGF X ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_YCT',Y0_F       ,header_comment = '2DGF Y ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_AMP',A_F        ,header_comment = '2DGF Amplitude ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_SGX',SIGMAX_F   ,header_comment = '2DGF Sigma X ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_SGY',SIGMAY_F   ,header_comment = '2DGF Sigma Y ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_THT',THETA_F    ,header_comment = '2DGF Theta ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_OFS',OFFSET_F   ,header_comment = '2DGF Offset ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_XCE',X0_E       ,header_comment = '2DGF X Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_YCE',Y0_E       ,header_comment = '2DGF Y Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_AME',A_E        ,header_comment = '2DGF Amplitude Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_SXE',SIGMAX_E   ,header_comment = '2DGF Sigma X Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_SYE',SIGMAY_E   ,header_comment = '2DGF Sigma Y Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_THE',THETA_E    ,header_comment = '2DGF Theta Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_OFE',OFFSET_E   ,header_comment = '2DGF Offset Err ' + clp_hdc)

	DGF_vlm = abs(SIGMAX_F) * abs(SIGMAX_F) * A_F * 2 * np.pi
	DGF_vx1 = abs(SIGMAX_F) * abs(SIGMAX_F) * A_F * 2 * np.pi  * slice_cwdt
	DGF_vx2 = abs(SIGMAX_F) * abs(SIGMAX_F) * A_F * 2 * np.pi  * fwhm2sigma(slice_fwhm)

	DGF_vle = (2*np.pi * (SIGMAX_F**2)) * (np.sqrt(((SIGMAX_F**2) * (A_E**2)) + (4 * (A_F**2) * (SIGMAX_E**2))))
	DGF_v1e = (2*np.pi * (SIGMAX_F**2)) * (np.sqrt(((SIGMAX_F**2) * (A_E**2)) + (4 * (A_F**2) * (SIGMAX_E**2)))) * slice_cwdt
	DGF_v2e = (2*np.pi * (SIGMAX_F**2)) * (np.sqrt(((SIGMAX_F**2) * (A_E**2)) + (4 * (A_F**2) * (SIGMAX_E**2)))) * fwhm2sigma(slice_fwhm)

	DGF_vol = FluxToLum(DGF_vlm,z_f2l,frq_r)
	DGF_v1l = FluxToLum(DGF_vx1,z_f2l,frq_r)
	DGF_v2l = FluxToLum(DGF_vx2,z_f2l,frq_r)

	redshift_inf_1 = Header_Get(Cube2bFit,'STZ_1SL')
	redshift_sup_1 = Header_Get(Cube2bFit,'STZ_1SH')
	redshift_inf_2 = Header_Get(Cube2bFit,'STZ_2SL')
	redshift_sup_2 = Header_Get(Cube2bFit,'STZ_2SH')
	redshift_inf_3 = Header_Get(Cube2bFit,'STZ_3SL')
	redshift_sup_3 = Header_Get(Cube2bFit,'STZ_3SH')
	
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_FLS',DGF_vlm   ,header_comment = '2DGF Vol ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_FT1',DGF_vx1   ,header_comment = '2DGF Vol X CbeWth ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_FT2',DGF_vx2   ,header_comment = '2DGF Vol X CbeWth 1DGF FWHM ' + clp_hdc)	

	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_FSE',DGF_vle   ,header_comment = '2DGF Vol ' + clp_hdc + ' Err')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_F1E',DGF_v1e   ,header_comment = '2DGF Vol X CbeWth ' + clp_hdc + ' Err')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_F2E',DGF_v2e   ,header_comment = '2DGF Vol X CbeWth 1DGF FWHM ' + clp_hdc + ' Err')

	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_LMS',DGF_vol[0],header_comment = '2DGF Vol Lum ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_LLS',DGF_vol[1],header_comment = '2DGF Vol Lum [log] ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_LMT',DGF_v1l[0],header_comment = '2DGF Vol X CbeWth Lum ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_LLT',DGF_v1l[1],header_comment = '2DGF Vol X CbeWth Lum [log] ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_LMF',DGF_v2l[0],header_comment = '2DGF Vol X CbeWth 1DGF FWHM Lum ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_LLF',DGF_v2l[1],header_comment = '2DGF Vol X CbeWth 1DGF FWHM Lum [log] ' + clp_hdc)
	
	lum_area_err_1_s = Luminosity_Error(DGF_vol[0],redshift_inf_1,redshift_sup_1,DGF_vle,frq_r=frq_r)
	lum_area_err_2_s = Luminosity_Error(DGF_vol[0],redshift_inf_2,redshift_sup_2,DGF_vle,frq_r=frq_r)
	lum_area_err_3_s = Luminosity_Error(DGF_vol[0],redshift_inf_3,redshift_sup_3,DGF_vle,frq_r=frq_r)

	lum_area_err_1_t = Luminosity_Error(DGF_v1l[0],redshift_inf_1,redshift_sup_1,DGF_v1e,frq_r=frq_r)
	lum_area_err_2_t = Luminosity_Error(DGF_v1l[0],redshift_inf_2,redshift_sup_2,DGF_v1e,frq_r=frq_r)
	lum_area_err_3_t = Luminosity_Error(DGF_v1l[0],redshift_inf_3,redshift_sup_3,DGF_v1e,frq_r=frq_r)

	lum_area_err_1_f = Luminosity_Error(DGF_v2l[0],redshift_inf_1,redshift_sup_1,DGF_v2e,frq_r=frq_r)
	lum_area_err_2_f = Luminosity_Error(DGF_v2l[0],redshift_inf_2,redshift_sup_2,DGF_v2e,frq_r=frq_r)
	lum_area_err_3_f = Luminosity_Error(DGF_v2l[0],redshift_inf_3,redshift_sup_3,DGF_v2e,frq_r=frq_r)

	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_S1L',lum_area_err_1_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_S1H',lum_area_err_1_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_S1L',lum_area_err_1_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_S1H',lum_area_err_1_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_S2L',lum_area_err_2_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_S2H',lum_area_err_2_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_S2L',lum_area_err_2_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_S2H',lum_area_err_2_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_S3L',lum_area_err_3_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_S3H',lum_area_err_3_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_S3L',lum_area_err_3_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_S3H',lum_area_err_3_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')

	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_T1L',lum_area_err_1_t[0], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_T1H',lum_area_err_1_t[1], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_T1L',lum_area_err_1_t[2], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_T1H',lum_area_err_1_t[3], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_T2L',lum_area_err_2_t[0], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_T2H',lum_area_err_2_t[1], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_T2L',lum_area_err_2_t[2], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_T2H',lum_area_err_2_t[3], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_T3L',lum_area_err_3_t[0], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_T3H',lum_area_err_3_t[1], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_T3L',lum_area_err_3_t[2], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_T3H',lum_area_err_3_t[3], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')

	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_F1L',lum_area_err_1_f[0], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_F1H',lum_area_err_1_f[1], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_F1L',lum_area_err_1_f[2], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_F1H',lum_area_err_1_f[3], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_F2L',lum_area_err_2_f[0], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_F2H',lum_area_err_2_f[1], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_F2L',lum_area_err_2_f[2], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_F2H',lum_area_err_2_f[3], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_F3L',lum_area_err_3_f[0], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2L_F3H',lum_area_err_3_f[1], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_F3L',lum_area_err_3_f[2], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2M_F3H',lum_area_err_3_f[3], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')

	Header_Add(Cube2bFit, clp_hdr + 'CL_CNI',slice_nblw ,header_comment = 'Initial Chn used for Collapse Num ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + 'CL_CNL',slice_nbhg ,header_comment = 'Last Chn used for Collapse Num ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + 'CL_CVI',slice_vllw ,header_comment = 'Initial Chn used for Collapse Vel ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + 'CL_CVL',slice_vlhg ,header_comment = 'Last Chn used for Collapse Vel ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_XCT',X0_F       ,header_comment = '2DGF X ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_YCT',Y0_F       ,header_comment = '2DGF Y ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_AMP',A_F        ,header_comment = '2DGF Amplitude ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_SGX',SIGMAX_F   ,header_comment = '2DGF Sigma X ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_SGY',SIGMAY_F   ,header_comment = '2DGF Sigma Y ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_THT',THETA_F    ,header_comment = '2DGF Theta ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_OFS',OFFSET_F   ,header_comment = '2DGF Offset ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_XCE',X0_E       ,header_comment = '2DGF X Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_YCE',Y0_E       ,header_comment = '2DGF Y Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_AME',A_E        ,header_comment = '2DGF Amplitude Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_SXE',SIGMAX_E   ,header_comment = '2DGF Sigma X Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_SYE',SIGMAY_E   ,header_comment = '2DGF Sigma Y Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_THE',THETA_E    ,header_comment = '2DGF Theta Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_OFE',OFFSET_E   ,header_comment = '2DGF Offset Err ' + clp_hdc)

	Header_Add(Cube2bFit,clp_hdr + '2G_FLS',DGF_vlm   ,header_comment = '2DGF Vol ' + clp_hdc)
	Header_Add(Cube2bFit,clp_hdr + '2G_FT1',DGF_vx1   ,header_comment = '2DGF Vol X CbeWth ' + clp_hdc)
	Header_Add(Cube2bFit,clp_hdr + '2G_FT2',DGF_vx2   ,header_comment = '2DGF Vol X CbeWth 1DGF FWHM ' + clp_hdc)	

	Header_Add(Cube2bFit,clp_hdr + '2G_FSE',DGF_vle   ,header_comment = '2DGF Vol ' + clp_hdc + ' Err')
	Header_Add(Cube2bFit,clp_hdr + '2G_F1E',DGF_v1e   ,header_comment = '2DGF Vol X CbeWth ' + clp_hdc + ' Err')
	Header_Add(Cube2bFit,clp_hdr + '2G_F2E',DGF_v2e   ,header_comment = '2DGF Vol X CbeWth 1DGF FWHM ' + clp_hdc + ' Err')

	Header_Add(Cube2bFit,clp_hdr + '2G_LMS',DGF_vol[0],header_comment = '2DGF Vol Lum ' + clp_hdc)
	Header_Add(Cube2bFit,clp_hdr + '2G_LLS',DGF_vol[1],header_comment = '2DGF Vol Lum [log] ' + clp_hdc)
	Header_Add(Cube2bFit,clp_hdr + '2G_LMT',DGF_v1l[0],header_comment = '2DGF Vol X CbeWth Lum ' + clp_hdc)
	Header_Add(Cube2bFit,clp_hdr + '2G_LLT',DGF_v1l[1],header_comment = '2DGF Vol X CbeWth Lum [log] ' + clp_hdc)
	Header_Add(Cube2bFit,clp_hdr + '2G_LMF',DGF_v2l[0],header_comment = '2DGF Vol X CbeWth 1DGF FWHM Lum ' + clp_hdc)
	Header_Add(Cube2bFit,clp_hdr + '2G_LLF',DGF_v2l[1],header_comment = '2DGF Vol X CbeWth 1DGF FWHM Lum [log] ' + clp_hdc)
	

	Header_Add(Cube2bFit,clp_hdr + '2L_S1L',lum_area_err_1_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_S1H',lum_area_err_1_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_S1L',lum_area_err_1_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_S1H',lum_area_err_1_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_S2L',lum_area_err_2_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_S2H',lum_area_err_2_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_S2L',lum_area_err_2_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_S2H',lum_area_err_2_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_S3L',lum_area_err_3_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_S3H',lum_area_err_3_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_S3L',lum_area_err_3_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_S3H',lum_area_err_3_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')

	Header_Add(Cube2bFit,clp_hdr + '2L_T1L',lum_area_err_1_t[0], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_T1H',lum_area_err_1_t[1], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_T1L',lum_area_err_1_t[2], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_T1H',lum_area_err_1_t[3], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_T2L',lum_area_err_2_t[0], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_T2H',lum_area_err_2_t[1], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_T2L',lum_area_err_2_t[2], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_T2H',lum_area_err_2_t[3], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_T3L',lum_area_err_3_t[0], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_T3H',lum_area_err_3_t[1], header_comment = '2DGF Vol Lum T X CbeWth ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_T3L',lum_area_err_3_t[2], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_T3H',lum_area_err_3_t[3], header_comment = '2DGF Vol Lum T X CbeWth [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')

	Header_Add(Cube2bFit,clp_hdr + '2L_F1L',lum_area_err_1_f[0], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_F1H',lum_area_err_1_f[1], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_F1L',lum_area_err_1_f[2], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_F1H',lum_area_err_1_f[3], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_F2L',lum_area_err_2_f[0], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_F2H',lum_area_err_2_f[1], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_F2L',lum_area_err_2_f[2], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_F2H',lum_area_err_2_f[3], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_F3L',lum_area_err_3_f[0], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bFit,clp_hdr + '2L_F3H',lum_area_err_3_f[1], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_F3L',lum_area_err_3_f[2], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct')
	Header_Add(Cube2bFit,clp_hdr + '2M_F3H',lum_area_err_3_f[3], header_comment = '2DGF Vol Lum T X CbeWth 1DGF FWHM [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct')

	if verbose == True:
		print
		print ('Initial Guess:')
		print ('X0_G         : ',X0_f2DG)
		print ('Y0_G         : ',Y0_f2DG)
		print ('A_G          : ',A_f2DG)
		print ('SIGMAX_G     : ',SIGMAX_f2DG)
		print ('SIGMAY_G     : ',SIGMAY_f2DG)
		print ('THETA_G      : ',THETA_f2DG)
		print ('OFFSET_G     : ',OFS_f2DG)
		print 
		print (colored(Message1,'yellow'))
		print (colored(Message2,'yellow'))

		print
		print ('Fit Values   :')
		print ('X0_F         : ',X0_F    ,' +- ',X0_E    )
		print ('Y0_F         : ',Y0_F    ,' +- ',Y0_E    )
		print ('A_F          : ',A_F     ,' +- ',A_E     )
		print ('SIGMAX_F     : ',SIGMAX_F,' +- ',SIGMAX_E)
		print ('SIGMAY_F     : ',SIGMAY_F,' +- ',SIGMAY_E)
		print ('THETA_F      : ',THETA_F ,' +- ',THETA_E )
		print ('OFFSET_F     : ',OFFSET_F,' +- ',OFFSET_E)
		print ('Area_F       : ',DGF_vlm ,' +- ',DGF_vle)
		print ('Volume_F     : ',DGF_vx1 ,' +- ',DGF_v1e)
		print
		print ('Shift from the X coordinate center:',X_DIF)
		print ('Shift from the Y coordinate center:',Y_DIF)
		print
		print (colored('Generated Plot: ' + str(PLOTFILENAME) ,'cyan'))
		print
	elif verbose == False:
		pass
		
	if displ_s_f == True:
		fxsize=9
		fysize=8
		f = plt.figure(num=None, figsize=(fxsize, fysize), dpi=180, facecolor='w',
			edgecolor='k')
		plt.subplots_adjust(
			left 	= (16/25.4)/fxsize, 
			bottom 	= (12/25.4)/fysize, 
			right 	= 1 - (6/25.4)/fxsize, 
			top 	= 1 - (15/25.4)/fysize)
		plt.subplots_adjust(hspace=0)

		gs0 = gridspec.GridSpec(1, 1)
		gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
		ax110 = plt.Subplot(f, gs11[0,0])
		f.add_subplot(ax110)

		ax110.set_rasterization_zorder(1)
		plt.autoscale(enable=True, axis='y', tight=False)

		plt.title(plt_tlt + ' (' +  str(x_ref) + ','+str(y_ref)+')',family='serif')
		plt.xlabel('X',fontsize=20,family = 'serif')
		plt.ylabel('Y',fontsize=20,family = 'serif')
		plt.tick_params(which='both', width=1.0)
		plt.tick_params(which='major', length=10)
		plt.tick_params(which='minor', length=5)
		ax110.minorticks_on()

		if ('_ms.' in Cube2bFit) or ('dta_in.' in Cube2bFit) or ('dta_ot.' in Cube2bFit):
			tick_color = 'white'
		elif ('msk_in.' in Cube2bFit) or ('crc.' in Cube2bFit) or ('msk_ot.' in Cube2bFit):
			tick_color = 'black'
		else:
			tick_color = 'white'

		ax110.xaxis.set_tick_params(which='both',labelsize=16,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)
		ax110.yaxis.set_tick_params(which='both',labelsize=16,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)

		plt.imshow(ydata.reshape(nx, ny), cmap=plt.cm.viridis, origin='lower',
		    extent=(x.min(), x.max(), y.min(), y.max()))
		divider = make_axes_locatable(ax110)
		cax  = divider.append_axes("right", size="5%", pad=0.05)	
		cbar = plt.colorbar(cax=cax)
		cbar.set_label('S [Jy]', rotation=270,family = 'serif')

		min_y, max_y = ax110.get_ylim()
		min_x, max_x = ax110.get_xlim()	

		plt.text(0,max_y-(max_y/10),
				'1 pix = '  + str(scale_arcsec) + ' arcsec',  
				ha='left' , va='baseline',color='white',fontsize=16,
				family = 'serif')

		X0,Y0 = X0_F,Y0_F

		sigx  = SIGMAX_F
		sigy  = SIGMAY_F
		theta = THETA_F

		try:
			colors=['white','white','white','white','white']
			for j in xrange(1, 4):
			    ell = Ellipse(xy=(X0, Y0),
			        width=sigx*2*j, height=sigy*2*j,
			        angle=theta,
			        edgecolor=colors[j])
			    ell.set_facecolor('none')
			    ax110.add_artist(ell)

			plt.text(0,0.5,
				'Fit'+
				': X$_{0}$ '        + str(x_ref+X_DIF)  +
				', Y$_{0}$ '        + str(y_ref+Y_DIF)  +
				', A : '            + str(np.round(popt[2],3))  + ' $\pm$ ' + str(np.round(A_E,5))      +
				', $\sigma_{x}$ : ' + str(np.round(SIGMAX_F,3)) + ' $\pm$ ' + str(np.round(SIGMAX_E,3)) +
				', $\sigma_{y}$ : ' + str(np.round(SIGMAY_F,3)) + ' $\pm$ ' + str(np.round(SIGMAY_E,3)) +
				', S(A) : '         + str(np.round(DGF_vlm,3))  + ' $\pm$ ' + str(np.round(DGF_vle,3))  +
				', S(V) : '         + str(np.round(DGF_vx1,3))  + ' $\pm$ ' + str(np.round(DGF_v1e,3)),
				ha='left' , va='bottom',color='white',fontsize=16,family = 'serif')

			plt.text(0,0,
				'L : '        + str(np.round(DGF_v1l[0],3))  + '-' + str(np.round(lum_area_err_1_t[0],3)) + '+' + str(str(np.round(lum_area_err_1_t[1],3))) +
				', log(L) : ' + str(np.round(DGF_v1l[1],3))  + '-' + str(np.round(lum_area_err_1_t[2],3)) + '+' + str(str(np.round(lum_area_err_1_t[3],3))),
				ha='left' , va='bottom',color='white',fontsize=16,family = 'serif')

		except ValueError:
			pass

		plt.scatter(X0_F    + 0.0, Y0_F    + 0.0, s=25, c='white', marker='x')
		plt.scatter(X0_f2DG + 0.0, Y0_f2DG + 0.0, s=25, c='black', marker='+')

		ax110.xaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-X0_f2DG+x_ref ))
		ax110.yaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-Y0_f2DG+y_ref ))

		plt.savefig(PLOTFILENAME)
	elif displ_s_f == False:
		pass

	return popt,pcov,perr,fit_res,X_DIF,Y_DIF

def fit_1D_Gaussian(Cube2bPlot_1D,*args, **kwargs):
	autoaxis      = kwargs.get('autoaxis',True)
	verbose       = kwargs.get('verbose' , False)
	epssave       = kwargs.get('epssave' , False)
	showplot      = kwargs.get('showplot', False) 
	amplitude     = kwargs.get('amplitude',1)
	mean          = kwargs.get('mean',0)
	stddev        = kwargs.get('stddev',1)
	cubewdthv     = kwargs.get('cubewdthv',1)
	dest_dir_plt  = kwargs.get('dest_dir_plt',None)
	z_avg         = kwargs.get('z_avg',Header_Get(Cube2bPlot_1D,'STZ_AVG'))
	z_med         = kwargs.get('z_med',Header_Get(Cube2bPlot_1D,'STZ_MED'))
	frq_r         = kwargs.get('frq_r',Header_Get(Cube2bPlot_1D,'RESTFRQ'))
	z_f2l         = z_med

	prefix        = kwargs.get('prefix','')

	redshift_inf_1 = Header_Get(Cube2bPlot_1D,'STZ_1SL')
	redshift_sup_1 = Header_Get(Cube2bPlot_1D,'STZ_1SH')
	redshift_inf_2 = Header_Get(Cube2bPlot_1D,'STZ_2SL')
	redshift_sup_2 = Header_Get(Cube2bPlot_1D,'STZ_2SH')
	redshift_inf_3 = Header_Get(Cube2bPlot_1D,'STZ_3SL')
	redshift_sup_3 = Header_Get(Cube2bPlot_1D,'STZ_3SH')

	FLUX2bPlot = Cube_Stat_Slice(Cube2bPlot_1D,cubewdthv=cubewdthv,frq_r=frq_r)
	if dest_dir_plt != None:
		PLOTFILENAME = str(dest_dir_plt)  + '/' + prefix + (str(Cube2bPlot_1D).split('.fits')[0]).split('/')[-1] + '-1DGF.pdf'
	elif dest_dir_plt == None:
		PLOTFILENAME = stm_dir_plt    + '/' + prefix + (str(Cube2bPlot_1D).split('.fits')[0]).split('/')[-1] + '-1DGF.pdf'
	
	Cube_Info = Cube_Header_Get(Cube2bPlot_1D,frq_r* u.Hz)
	FRQ_AXS   = Cube_Info[16].value
	VEL_AXS   = Cube_Info[17].value
	FLX_SUM   = FLUX2bPlot[0]
	FLX_AVG   = FLUX2bPlot[1]
	FLX_MED   = FLUX2bPlot[2]
	FLX_STD   = FLUX2bPlot[3]
	FLX_TOT   = FLUX2bPlot[4]

	XAXIS     = VEL_AXS
	print
	print (colored('Central frequency: ' + str(frq_r),'yellow'))
	print (colored('X-axis (FRQ): ' + str(FRQ_AXS),'yellow'))
	print (colored('X-axis (VEL): ' + str(VEL_AXS),'yellow'))
	print

	label_SUM = 'SUM'
	label_AVG = 'AVG'
	label_MED = 'MED'
	label_STD = 'STD'

	fxsize=11
	fysize=8
	f = plt.figure(num=None, figsize=(11, 8), dpi=180, facecolor='w',
		edgecolor='k')
	plt.subplots_adjust(
		left 	= (38/25.4)/fxsize,   
		bottom 	= (14/25.4)/fysize, 
		right 	= 1 - (22/25.4)/fxsize,
		top 	= 1 - (15/25.4)/fysize)
	plt.subplots_adjust(hspace=0)

	gs0 = gridspec.GridSpec(1, 1)
	##########################################SPEC-1###################################

	gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
		
	ax110 = plt.Subplot(f, gs11[0,0])
	f.add_subplot(ax110)

	ax110.set_rasterization_zorder(1)
	plt.autoscale(enable=True, axis='both', tight=False)
	ax110.xaxis.set_tick_params(which='both',labelsize=16,direction='in',color='black',bottom=True,top=True,left=True,right=True)
	ax110.yaxis.set_tick_params(which='both',labelsize=16,direction='in',color='black',bottom=True,top=True,left=True,right=True)
	ax110.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	xticklabels = ax110.get_xticklabels()
	plt.setp(xticklabels, visible=True,family='serif')
	yticklabels = ax110.get_yticklabels()
	plt.setp(yticklabels, visible=True,family='serif')

	minorLocator_x   = plt.MultipleLocator((XAXIS[1]-XAXIS[0])/2)
	majorLocator_x   = plt.MultipleLocator(XAXIS[1]-XAXIS[0])
	plt.tick_params(which='both', width=1.0)
	plt.tick_params(which='major', length=10)
	plt.tick_params(which='minor', length=5)
	ax110.minorticks_on()


	plt.xlabel('$v$ kms$^{-1}$'  ,fontsize=16,family = 'serif')
	plt.ylabel('S [Jy]',fontsize=16,family = 'serif')

	plt.scatter(XAXIS, FLX_AVG, color = 'r'   , marker = '*', alpha = 0.4)
	plt.scatter(XAXIS, FLX_MED, color = 'b'   , marker = 'o', alpha = 0.4)

	slc_nmb       = kwargs.get('slc_nmb',(np.ceil(len(FLX_AVG)/2)))
	max_rng_val   = kwargs.get('max_rng_val',(np.ceil(len(FLX_AVG)/10.0)))
	max_rng       = kwargs.get('max_rng',False)

	if max_rng == True and slc_nmb > 2:
		indx_in = int(slc_nmb - max_rng_val)-1
		indx_fn = int(slc_nmb + max_rng_val)+1
		indx_FLX_SUM = np.where(FLX_SUM == max(FLX_SUM[indx_in:indx_fn]))[0][0]
		indx_FLX_AVG = np.where(FLX_AVG == max(FLX_AVG[indx_in:indx_fn]))[0][0]
		indx_FLX_MED = np.where(FLX_MED == max(FLX_MED[indx_in:indx_fn]))[0][0]
		plt.text(VEL_AXS[indx_FLX_AVG], max(FLX_AVG[indx_in:indx_fn]),str(indx_FLX_AVG+1) + ' ' + str(round(max(FLX_AVG),6)), ha='left' , va='bottom',color='red' ,family = 'serif')
		plt.text(VEL_AXS[indx_FLX_MED], max(FLX_MED[indx_in:indx_fn]),str(indx_FLX_MED+1) + ' ' + str(round(max(FLX_MED),6)), ha='right', va='top'   ,color='blue',family = 'serif')
	elif max_rng == False or (max_rng == True and slc_nmb <= 2):
		indx_FLX_SUM = np.where(FLX_SUM == max(FLX_SUM))[0][0]
		indx_FLX_AVG = np.where(FLX_AVG == max(FLX_AVG))[0][0]
		indx_FLX_MED = np.where(FLX_MED == max(FLX_MED))[0][0]
		plt.text(VEL_AXS[indx_FLX_AVG], max(FLX_AVG),str(indx_FLX_AVG+1) + ' ' + str(round(max(FLX_AVG),6)), ha='left' , va='bottom',color='red' ,family = 'serif')
		plt.text(VEL_AXS[indx_FLX_MED], max(FLX_MED),str(indx_FLX_MED+1) + ' ' + str(round(max(FLX_MED),6)), ha='right', va='top'   ,color='blue',family = 'serif')

	mean_avg      = VEL_AXS[int(indx_FLX_AVG)]
	amplitude_avg = FLX_AVG[indx_FLX_AVG]

	mean_med      = VEL_AXS[int(indx_FLX_MED)]
	amplitude_med = FLX_MED[indx_FLX_MED]

	mean_sum      = VEL_AXS[int(indx_FLX_SUM)]
	amplitude_sum = FLX_SUM[indx_FLX_SUM]

	Header_Add(Cube2bPlot_1D,'MAX_SNS',int(indx_FLX_SUM)          ,header_comment='Vel Prf Max Chn Location SUM')
	Header_Add(Cube2bPlot_1D,'MAX_SVS',VEL_AXS[int(indx_FLX_SUM)] ,header_comment='Vel Prf Max Vel Location [km/s] SUM')
	Header_Add(Cube2bPlot_1D,'MAX_SVS',FRQ_AXS[int(indx_FLX_SUM)] ,header_comment='Vel Prf Max Frequency Location [Hz] SUM')
	Header_Add(Cube2bPlot_1D,'MAX_VLS',max(FLX_SUM)               ,header_comment='Vel Prf Max Chn Value SUM')
	Header_Add(Cube2bPlot_1D,'MAX_SNA',int(indx_FLX_AVG)          ,header_comment='Vel Prf Max Chn Location AVG')
	Header_Add(Cube2bPlot_1D,'MAX_SVA',VEL_AXS[int(indx_FLX_AVG)] ,header_comment='Vel Prf Max Vel Location [km/s] AVG')
	Header_Add(Cube2bPlot_1D,'MAX_SVA',FRQ_AXS[int(indx_FLX_AVG)] ,header_comment='Vel Prf Max Frequency Location [Hz] AVG')
	Header_Add(Cube2bPlot_1D,'MAX_VLA',max(FLX_AVG)               ,header_comment='Vel Prf Max Chn Value AVG')
	Header_Add(Cube2bPlot_1D,'MAX_SNM',int(indx_FLX_MED)          ,header_comment='Vel Prf Max Chn Location MED')
	Header_Add(Cube2bPlot_1D,'MAX_SVM',VEL_AXS[int(indx_FLX_MED)] ,header_comment='Vel Prf Max Vel Location [km/s] MED')
	Header_Add(Cube2bPlot_1D,'MAX_SVM',FRQ_AXS[int(indx_FLX_MED)] ,header_comment='Vel Prf Max Frequency Location [Hz] MED')
	Header_Add(Cube2bPlot_1D,'MAX_VLM',max(FLX_MED)               ,header_comment='Vel Prf Max Chn Value MED')

	g_init_avg = apmd.models.Gaussian1D(amplitude=amplitude_avg, mean=mean_avg, stddev=stddev)
	g_init_avg.amplitude.fixed = True
	fit_g_avg  = apmd.fitting.LevMarLSQFitter()
	try:
		g_avg            = fit_g_avg(g_init_avg, XAXIS, FLX_AVG)
		g_avg_cov        = fit_g_avg.fit_info['param_cov']
		if (g_avg_cov is None):
			g_avg_var    = np.zeros((1,2))
			g_avg_var[:] = np.nan
			g_avg_var    = np.squeeze((g_avg_var))
		elif np.linalg.det(g_avg_cov) < 0:
			g_avg_var    = np.zeros_like(np.diag(g_avg_cov))
			g_avg_var[:] = np.nan
			g_avg_var    = np.squeeze((g_avg_var))
		else:
			g_avg_var = np.sqrt(np.diag(g_avg_cov))
			g_avg_var = np.squeeze((g_avg_var))

		g_avg_var_mea    = g_avg_var[0]
		g_avg_var_std    = g_avg_var[-1]
		Area_avg         = scpint.quad(lambda x: g_avg.amplitude*np.exp(-((x-g_avg.mean)**2)/(2*g_avg.stddev**2)), -np.inf, np.inf)
		Area_avg_man     = g_avg.stddev[0] * amplitude_avg * np.sqrt(2*np.pi)
		Area_avg_man_err = g_avg_var_std   * amplitude_avg * np.sqrt(2*np.pi)

		Lum_Area_avg     = FluxToLum(Area_avg[0],z_f2l,frq_r)
		lum_area_err_1_a = Luminosity_Error(Area_avg[0],redshift_inf_1,redshift_sup_1,Area_avg[1],frq_r=frq_r)
		lum_area_err_2_a = Luminosity_Error(Area_avg[0],redshift_inf_2,redshift_sup_2,Area_avg[1],frq_r=frq_r)
		lum_area_err_3_a = Luminosity_Error(Area_avg[0],redshift_inf_3,redshift_sup_3,Area_avg[1],frq_r=frq_r)

		lum_area_err_1_m = Luminosity_Error(Area_avg[0],redshift_inf_1,redshift_sup_1,Area_avg_man_err,frq_r=frq_r)
		lum_area_err_2_m = Luminosity_Error(Area_avg[0],redshift_inf_2,redshift_sup_2,Area_avg_man_err,frq_r=frq_r)
		lum_area_err_3_m = Luminosity_Error(Area_avg[0],redshift_inf_3,redshift_sup_3,Area_avg_man_err,frq_r=frq_r)

		XAXIS_FIT        = np.arange(XAXIS[0],XAXIS[-1],0.01)
		plt.plot(XAXIS_FIT, g_avg(XAXIS_FIT), color='red',ls=':',alpha=0.4,
			label='Gaussian fit ('+label_AVG+
					') A : '        + str(np.round(g_avg.amplitude[0],3))              +
					', $\mu$ : '    + str(np.round(g_avg.mean[0],3))                   + ' $\pm$ ' + str(np.round(g_avg_var_mea,3))       +
					', $\sigma$ : ' + str(np.round(g_avg.stddev[0],3))                 + ' $\pm$ ' + str(np.round(g_avg_var_std,3))
					)

		Header_Add(Cube2bPlot_1D,'FTA_AMP',np.round(g_avg.amplitude[0],5)             , header_comment = '1DGF Amplitude AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CTR',np.round(g_avg.mean[0],6)                  , header_comment = '1DGF Center AVG')
		Header_Add(Cube2bPlot_1D,'FTA_SIG',np.round(g_avg.stddev[0],2)                , header_comment = '1DGF Sigma AVG')
		Header_Add(Cube2bPlot_1D,'FTA_FWH',linewidth_fwhm(np.round(g_avg.stddev[0],2)), header_comment = '1DGF FWHM  AVG')
		Header_Add(Cube2bPlot_1D,'FTA_A2A',Area_avg[0]                                , header_comment = '1DGF Area A AVG')
		Header_Add(Cube2bPlot_1D,'FTA_A2M',Area_avg_man                               , header_comment = '1DGF Area M AVG')
		Header_Add(Cube2bPlot_1D,'FTA_LUM',Lum_Area_avg[0]                            , header_comment = '1DGF Ar2Lum AVG')
		Header_Add(Cube2bPlot_1D,'FTA_LLM',Lum_Area_avg[1]                            , header_comment = '1DGF Ar2Lum [log] AVG')

		Header_Add(Cube2bPlot_1D,'FTA_APE',0                                          , header_comment = '1DGF Amplitude Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CTE',g_avg_var_mea                              , header_comment = '1DGF Center Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_SGE',g_avg_var_std                              , header_comment = '1DGF Sigma Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_FWE',g_avg_var_std*linewidth_fwhm(1)            , header_comment = '1DGF FWHM  Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_AAE',Area_avg[1]                                , header_comment = '1DGF Area A Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_AME',Area_avg_man_err                           , header_comment = '1DGF Area M Err AVG')

		Header_Add(Cube2bPlot_1D,'FTA_ML1',lum_area_err_1_a[0]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH1',lum_area_err_1_a[1]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL1',lum_area_err_1_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH1',lum_area_err_1_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTA_ML2',lum_area_err_2_a[0]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH2',lum_area_err_2_a[1]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL2',lum_area_err_2_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH2',lum_area_err_2_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FTA_ML3',lum_area_err_3_a[0]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH3',lum_area_err_3_a[1]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL3',lum_area_err_3_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH3',lum_area_err_3_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML1',lum_area_err_1_m[0]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH1',lum_area_err_1_m[1]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL1',lum_area_err_1_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH1',lum_area_err_1_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML2',lum_area_err_2_m[0]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH2',lum_area_err_2_m[1]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL2',lum_area_err_2_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH2',lum_area_err_2_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML3',lum_area_err_3_m[0]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH3',lum_area_err_3_m[1]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL3',lum_area_err_3_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH3',lum_area_err_3_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm hg lmt 99.8 pct')

	except (TypeError,RuntimeError):
		Header_Add(Cube2bPlot_1D,'FTA_AMP',np.nan, header_comment = '1DGF Amplitude AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CTR',np.nan, header_comment = '1DGF Center AVG')
		Header_Add(Cube2bPlot_1D,'FTA_SIG',np.nan, header_comment = '1DGF Sigma AVG')
		Header_Add(Cube2bPlot_1D,'FTA_FWH',np.nan, header_comment = '1DGF FWHM AVG')
		Header_Add(Cube2bPlot_1D,'FTA_A2A',np.nan, header_comment = '1DGF Area A AVG')
		Header_Add(Cube2bPlot_1D,'FTA_A2M',np.nan, header_comment = '1DGF Area M AVG')
		Header_Add(Cube2bPlot_1D,'FTA_LUM',np.nan, header_comment = '1DGF Ar2Lum AVG')
		Header_Add(Cube2bPlot_1D,'FTA_LLM',np.nan, header_comment = '1DGF Ar2Lum [log] AVG')

		Header_Add(Cube2bPlot_1D,'FTA_APE',np.nan, header_comment = '1DGF Amplitude Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CTE',np.nan, header_comment = '1DGF Center Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_SGE',np.nan, header_comment = '1DGF Sigma Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_FWE',np.nan, header_comment = '1DGF FWHM Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_AAE',np.nan, header_comment = '1DGF Area A Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_AME',np.nan, header_comment = '1DGF Area M Err AVG')

		Header_Add(Cube2bPlot_1D,'FTA_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTA_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm hg lmt 99.8 pct')

		print (colored('No 1D gaussian (avg) fit performed! : ' + str(Cube2bPlot_1D),'yellow'))

	g_init_med = apmd.models.Gaussian1D(amplitude=amplitude_med, mean=mean_med, stddev=stddev)
	g_init_med.amplitude.fixed = True
	fit_g_med = apmd.fitting.LevMarLSQFitter()
	try:
		g_med            = fit_g_med(g_init_med, XAXIS, FLX_MED)
		g_med_cov        = fit_g_med.fit_info['param_cov']
		if (g_med_cov is None):
			g_med_var    = np.zeros((1,2))
			g_med_var[:] = np.nan
			g_med_var    = np.squeeze((g_med_var))
		elif (np.linalg.det(g_med_cov) < 0):
			g_med_var    = np.zeros_like(np.diag(g_med_cov))
			g_med_var[:] = np.nan
			g_med_var    = np.squeeze((g_med_var))
		else:
			g_med_var    = np.sqrt(np.diag(g_med_cov))
			g_med_var    = np.squeeze((g_med_var))

		g_med_var_mea    = g_med_var[0]
		g_med_var_std    = g_med_var[-1]
		Area_med         = scpint.quad(lambda x: g_med.amplitude*np.exp(-((x-g_med.mean)**2)/(2*g_med.stddev**2)), -np.inf, np.inf)

		Area_med_man     = g_med.stddev[0] * amplitude_med * np.sqrt(2*np.pi)
		Area_med_man_err = g_med_var_std   * amplitude_med * np.sqrt(2*np.pi)

		Lum_Area_med     = FluxToLum(Area_med[0],z_f2l,frq_r)
		lum_area_err_1_a = Luminosity_Error(Area_med[0],redshift_inf_1,redshift_sup_1,Area_med[1],frq_r=frq_r)
		lum_area_err_2_a = Luminosity_Error(Area_med[0],redshift_inf_2,redshift_sup_2,Area_med[1],frq_r=frq_r)
		lum_area_err_3_a = Luminosity_Error(Area_med[0],redshift_inf_3,redshift_sup_3,Area_med[1],frq_r=frq_r)

		lum_area_err_1_m = Luminosity_Error(Area_med[0],redshift_inf_1,redshift_sup_1,Area_med_man_err,frq_r=frq_r)
		lum_area_err_2_m = Luminosity_Error(Area_med[0],redshift_inf_2,redshift_sup_2,Area_med_man_err,frq_r=frq_r)
		lum_area_err_3_m = Luminosity_Error(Area_med[0],redshift_inf_3,redshift_sup_3,Area_med_man_err,frq_r=frq_r)

		XAXIS_FIT        = np.arange(XAXIS[0],XAXIS[-1],0.01)
		plt.plot(XAXIS_FIT, g_med(XAXIS_FIT), color='blue',ls=':',alpha=0.4,
			label='Gaussian fit ('+label_MED+
					') A : '       + str(np.round(g_med.amplitude[0],3))              +
					', $\mu$ : '   + str(np.round(g_med.mean[0],3))                   + ' $\pm$ ' + str(np.round(g_med_var_mea,3))       +
					', $\sigma$ : '+ str(np.round(g_med.stddev[0],3))                 + ' $\pm$ ' + str(np.round(g_med_var_std,3))
					)

		Header_Add(Cube2bPlot_1D,'FTM_AMP',np.round(g_med.amplitude[0],5)             , header_comment = '1DGF Amplitude MED')
		Header_Add(Cube2bPlot_1D,'FTM_CTR',np.round(g_med.mean[0],6)                  , header_comment = '1DGF Center MED')
		Header_Add(Cube2bPlot_1D,'FTM_SIG',np.round(g_med.stddev[0],2)                , header_comment = '1DGF Sigma MED')
		Header_Add(Cube2bPlot_1D,'FTM_FWH',linewidth_fwhm(np.round(g_med.stddev[0],2)), header_comment = '1DGF FWHM MED')
		Header_Add(Cube2bPlot_1D,'FTM_A2A',Area_med[0]                                , header_comment = '1DGF Area MED')
		Header_Add(Cube2bPlot_1D,'FTM_A2M',Area_med_man                               , header_comment = '1DGF Area M MED')
		Header_Add(Cube2bPlot_1D,'FTM_LUM',Lum_Area_med[0]                            , header_comment = '1DGF Ar2Lum MED')
		Header_Add(Cube2bPlot_1D,'FTM_LLM',Lum_Area_med[1]                            , header_comment = '1DGF Ar2Lum [log] MED')

		Header_Add(Cube2bPlot_1D,'FTM_APE',0                                          , header_comment = '1DGF Amplitude Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_CTE',g_med_var_mea                              , header_comment = '1DGF Center Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_SGE',g_med_var_std                              , header_comment = '1DGF Sigma Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_FWE',g_med_var_std*linewidth_fwhm(1)            , header_comment = '1DGF FWHM Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_AAE',Area_med[1]                                , header_comment = '1DGF Area A Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_AME',Area_med_man_err                           , header_comment = '1DGF Area M Err MED')
		
		Header_Add(Cube2bPlot_1D,'FTM_ML1',lum_area_err_1_a[0]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH1',lum_area_err_1_a[1]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL1',lum_area_err_1_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH1',lum_area_err_1_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTM_ML2',lum_area_err_2_a[0]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH2',lum_area_err_2_a[1]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL2',lum_area_err_2_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH2',lum_area_err_2_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FTM_ML3',lum_area_err_3_a[0]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH3',lum_area_err_3_a[1]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL3',lum_area_err_3_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH3',lum_area_err_3_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML1',lum_area_err_1_m[0]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH1',lum_area_err_1_m[1]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL1',lum_area_err_1_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH1',lum_area_err_1_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML2',lum_area_err_2_m[0]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH2',lum_area_err_2_m[1]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL2',lum_area_err_2_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH2',lum_area_err_2_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML3',lum_area_err_3_m[0]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH3',lum_area_err_3_m[1]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL3',lum_area_err_3_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH3',lum_area_err_3_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm hg lmt 99.8 pct')

	except (TypeError,RuntimeError):
		Header_Add(Cube2bPlot_1D,'FTM_AMP',np.nan, header_comment = '1DGF Amplitude MED')
		Header_Add(Cube2bPlot_1D,'FTM_CTR',np.nan, header_comment = '1DGF Center MED')
		Header_Add(Cube2bPlot_1D,'FTM_SIG',np.nan, header_comment = '1DGF Sigma MED')
		Header_Add(Cube2bPlot_1D,'FTM_FWH',np.nan, header_comment = '1DGF FWHM MED')
		Header_Add(Cube2bPlot_1D,'FTM_A2A',np.nan, header_comment = '1DGF Area A MED')
		Header_Add(Cube2bPlot_1D,'FTM_A2M',np.nan, header_comment = '1DGF Area M MED')
		Header_Add(Cube2bPlot_1D,'FTM_LUM',np.nan, header_comment = '1DGF Ar2Lum MED')
		Header_Add(Cube2bPlot_1D,'FTM_LLM',np.nan, header_comment = '1DGF Ar2Lum [log] MED')

		Header_Add(Cube2bPlot_1D,'FTM_APE',np.nan, header_comment = '1DGF Amplitude Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_CTE',np.nan, header_comment = '1DGF Center Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_SGE',np.nan, header_comment = '1DGF Sigma Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_FWE',np.nan, header_comment = '1DGF FWHM Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_AAE',np.nan, header_comment = '1DGF Area A Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_AME',np.nan, header_comment = '1DGF Area M Err MED')

		Header_Add(Cube2bPlot_1D,'FTM_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTM_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FTM_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm hg lmt 99.8 pct')

		print (colored('No 1D gaussian (med) fit performed! : ' + str(Cube2bPlot_1D),'yellow'))

	lg=plt.legend(loc=3,prop={'size':14})
	lg.draw_frame(False)

	ax110_twin = ax110.twiny()
	ax110_p_major = ax110.get_xticks()
	ax110_twin_major = []
	ax110_twin_major = (np.linspace(1,len(VEL_AXS),len(VEL_AXS))).astype(int)
	ax110_twin.set_xticks(ax110_twin_major)
	ax110_twin.set_xticklabels(ax110_twin_major,fontsize=16,family = 'serif')
	ax110_twin.xaxis.set_tick_params(which='both',labelsize=16,direction='in',color='black',bottom=False,top=True,left=True,right=True)

	ax210_twin = ax110.twinx()
	ax210_twin.scatter(XAXIS, FLX_SUM, color = 'green'   , marker = '^', alpha = 0.4) #label = label_SUM,
	ax210_twin.yaxis.set_tick_params(which='both',labelsize=16,direction='in',color='black',bottom=False,top=True,left=False,right=True)

	min_y=min(FLX_SUM)-0.25
	max_y=max(FLX_SUM)+0.25
	plt.ylim([min_y,max_y])
	ymin, ymax = plt.ylim()
	plt.ylim((ymin,ymax))
	align_yaxis(ax110, 0, ax210_twin, 0)

	g_init_sum = apmd.models.Gaussian1D(amplitude=amplitude_sum, mean=mean_sum, stddev=stddev)
	g_init_sum.amplitude.fixed = True
	fit_g_sum  = apmd.fitting.LevMarLSQFitter()
	
	try:
		#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq
		#http://docs.astropy.org/en/stable/api/astropy.modeling.fitting.LevMarLSQFitter.html#astropy.modeling.fitting.LevMarLSQFitter
		g_sum            = fit_g_sum(g_init_sum, XAXIS, FLX_SUM)
		g_sum_cov        = fit_g_sum.fit_info['param_cov']
		if (g_sum_cov is None):
			g_sum_var    = np.zeros((1,2))
			g_sum_var[:] = np.nan
			g_sum_var    = np.squeeze((g_sum_var))
		elif np.linalg.det(g_sum_cov) < 0:
			g_sum_var    = np.zeros_like(np.diag(g_sum_cov))
			g_sum_var[:] = np.nan
			g_sum_var    = np.squeeze((g_sum_var))
		else:
			g_sum_var    = np.sqrt(np.diag(g_sum_cov))
			g_sum_var    = np.squeeze((g_sum_var))

		g_sum_var_mea    = (g_sum_var[0])
		g_sum_var_std    = (g_sum_var[-1])
		Area_sum         = scpint.quad(lambda x: g_sum.amplitude*np.exp(-((x-g_sum.mean)**2)/(2*g_sum.stddev**2)), -np.inf, np.inf)

		Area_sum_man     = g_sum.stddev[0] * amplitude_sum * np.sqrt(2*np.pi)
		Area_sum_man_err = g_sum_var_std   * amplitude_sum * np.sqrt(2*np.pi)

		Lum_Area_sum     = FluxToLum(Area_sum[0],z_f2l,frq_r)
		lum_area_err_1_a = Luminosity_Error(Lum_Area_sum[0],redshift_inf_1,redshift_sup_1,Area_sum[1],frq_r=frq_r)
		lum_area_err_2_a = Luminosity_Error(Lum_Area_sum[0],redshift_inf_2,redshift_sup_2,Area_sum[1],frq_r=frq_r)
		lum_area_err_3_a = Luminosity_Error(Lum_Area_sum[0],redshift_inf_3,redshift_sup_3,Area_sum[1],frq_r=frq_r)

		lum_area_err_1_m = Luminosity_Error(Lum_Area_sum[0],redshift_inf_1,redshift_sup_1,Area_sum_man_err,frq_r=frq_r)
		lum_area_err_2_m = Luminosity_Error(Lum_Area_sum[0],redshift_inf_2,redshift_sup_2,Area_sum_man_err,frq_r=frq_r)
		lum_area_err_3_m = Luminosity_Error(Lum_Area_sum[0],redshift_inf_3,redshift_sup_3,Area_sum_man_err,frq_r=frq_r)

		XAXIS_FIT     = np.arange(XAXIS[0],XAXIS[-1],0.01)
		ax210_twin.plot(XAXIS_FIT, g_sum(XAXIS_FIT), color='green',ls='--',alpha=0.4,
			label='Gaussian fit ('+ label_SUM +
					') A : '       + str(np.round(g_sum.amplitude[0],3))              +
					', $\mu$ : '   + str(np.round(g_sum.mean[0],3))                   + ' $\pm$ ' + str(np.round(g_sum_var_mea,3)))

		Header_Add(Cube2bPlot_1D,'FTS_AMP',np.round(g_sum.amplitude[0],5)             , header_comment = '1DGF Amplitude SUM')
		Header_Add(Cube2bPlot_1D,'FTS_CTR',np.round(g_sum.mean[0],6)                  , header_comment = '1DGF Center SUM')
		Header_Add(Cube2bPlot_1D,'FTS_SIG',np.round(g_sum.stddev[0],2)                , header_comment = '1DGF Sigma SUM')
		Header_Add(Cube2bPlot_1D,'FTS_FWH',linewidth_fwhm(np.round(g_sum.stddev[0],2)), header_comment = '1DGF FWHM SUM')
		Header_Add(Cube2bPlot_1D,'FTS_A2A',Area_sum[0]                                , header_comment = '1DGF Area SUM')
		Header_Add(Cube2bPlot_1D,'FTS_A2M',Area_sum_man                               , header_comment = '1DGF Area M SUM')
		Header_Add(Cube2bPlot_1D,'FTS_LUM',Lum_Area_sum[0]                            , header_comment = '1DGF Ar2Lum SUM')
		Header_Add(Cube2bPlot_1D,'FTS_LLM',Lum_Area_sum[1]                            , header_comment = '1DGF Ar2Lum [log] SUM')

		Header_Add(Cube2bPlot_1D,'FTS_APE',0                                          , header_comment = '1DGF Amplitude Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_CTE',g_sum_var_mea                              , header_comment = '1DGF Center Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_SGE',g_sum_var_std                              , header_comment = '1DGF Sigma Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_FWE',g_sum_var_std*linewidth_fwhm(1)            , header_comment = '1DGF FWHM Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_AAE',Area_sum[1]                                , header_comment = '1DGF Area A Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_AME',Area_sum_man_err                           , header_comment = '1DGF Area M Err SUM')

		Header_Add(Cube2bPlot_1D,'FTS_ML1',lum_area_err_1_a[0]                        , header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTS_MH1',lum_area_err_1_a[1]                        , header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LL1',lum_area_err_1_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm lw lmt 15.9 pct') 
		Header_Add(Cube2bPlot_1D,'FTS_LH1',lum_area_err_1_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm hg lmt 84.1 pct') 

		Header_Add(Cube2bPlot_1D,'FTS_ML2',lum_area_err_2_a[0]                        , header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTS_MH2',lum_area_err_2_a[1]                        , header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LL2',lum_area_err_2_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LH2',lum_area_err_2_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FTS_ML3',lum_area_err_3_a[0]                        , header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTS_MH3',lum_area_err_3_a[1]                        , header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LL3',lum_area_err_3_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LH3',lum_area_err_3_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMS_ML1',lum_area_err_1_m[0]                        , header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMS_MH1',lum_area_err_1_m[1]                        , header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LL1',lum_area_err_1_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LH1',lum_area_err_1_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMS_ML2',lum_area_err_2_m[0]                        , header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMS_MH2',lum_area_err_2_m[1]                        , header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LL2',lum_area_err_2_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LH2',lum_area_err_2_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMS_ML3',lum_area_err_3_m[0]                        , header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMS_MH3',lum_area_err_3_m[1]                        , header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LL3',lum_area_err_3_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LH3',lum_area_err_3_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm hg lmt 99.8 pct')

	except (TypeError,RuntimeError):
		Header_Add(Cube2bPlot_1D,'FTS_AMP',np.nan, header_comment = '1DGF Amplitude SUM')
		Header_Add(Cube2bPlot_1D,'FTS_CTR',np.nan, header_comment = '1DGF Center SUM')
		Header_Add(Cube2bPlot_1D,'FTS_SIG',np.nan, header_comment = '1DGF Sigma SUM')
		Header_Add(Cube2bPlot_1D,'FTS_FWH',np.nan, header_comment = '1DGF FWHM SUM')
		Header_Add(Cube2bPlot_1D,'FTS_A2A',np.nan, header_comment = '1DGF Area A SUM')
		Header_Add(Cube2bPlot_1D,'FTS_A2M',np.nan, header_comment = '1DGF Area M SUM')
		Header_Add(Cube2bPlot_1D,'FTS_LUM',np.nan, header_comment = '1DGF Ar2Lum SUM')
		Header_Add(Cube2bPlot_1D,'FTS_LLM',np.nan, header_comment = '1DGF Ar2Lum [log] SUM')

		Header_Add(Cube2bPlot_1D,'FTS_APE',np.nan, header_comment = '1DGF Amplitude Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_CTE',np.nan, header_comment = '1DGF Center Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_SGE',np.nan, header_comment = '1DGF Sigma Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_FWE',np.nan, header_comment = '1DGF FWHM Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_AAE',np.nan, header_comment = '1DGF Area A Err SUM')
		Header_Add(Cube2bPlot_1D,'FTS_AME',np.nan, header_comment = '1DGF Area M Err SUM')

		Header_Add(Cube2bPlot_1D,'FTS_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTS_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTS_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTS_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FTS_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTS_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTS_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMS_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMS_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMS_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMS_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMS_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMS_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMS_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm hg lmt 99.8 pct')

		print (colored('No 1D gaussian (sum) fit performed! : ' + str(Cube2bPlot_1D),'yellow'))


	if max_rng == True and slc_nmb > 2:
		indx_in = int(slc_nmb - max_rng_val)-1
		indx_fn = int(slc_nmb + max_rng_val)+1
		indx_FLX_SUM = np.where(FLX_SUM == max(FLX_SUM[indx_in:indx_fn]))[0][0]
		plt.text(VEL_AXS[indx_FLX_SUM], max(FLX_SUM[indx_in:indx_fn]),str(indx_FLX_SUM+1) + ' ' + str(round(max(FLX_SUM),6)), ha='right', va='top'   ,color='green',family = 'serif')
	elif max_rng == False or (max_rng == True and slc_nmb <= 2):
		indx_FLX_SUM = np.where(FLX_SUM == max(FLX_SUM))[0][0]
		plt.text(VEL_AXS[indx_FLX_SUM], max(FLX_SUM),str(indx_FLX_SUM+1) + ' ' + str(round(max(FLX_SUM),6)), ha='right', va='top'   ,color='green',family = 'serif')


	#http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
	lg=plt.legend(loc=2,prop={'size':14})
	lg.draw_frame(False)

	##########################################SAVE#####################################
	plt.savefig(PLOTFILENAME)

	if verbose == True:
		print
		print (colored('Generated Plot: ' + str(PLOTFILENAME) + ' Frequency channels: ' + str(len(XAXIS)),'cyan'))
	elif verbose ==False:
		pass
	plt.close('all')

def Cube_Mask_Nan(cube_fits_ifn,*args,**kwags):
	#BEGIN MASK FREQUENCIES FUNCTION
	hdu_list   =apfts.open(cube_fits_ifn,mode='update')
	if len(hdu_list) >1:
		scidata    = hdu_list[0].data
		scidata1   = hdu_list[1].data
	else:
		scidata    = hdu_list[0].data
		scidata1   = scidata

	scidata  = np.squeeze(scidata)
	scidata1 = np.squeeze(scidata1)

	for slice_chann in range(len(scidata1)):
		if np.all(scidata[slice_chann,:,:]==0.):
			scidata[slice_chann,:,:] = np.nan
			pass
		else:
			pass
	hdu_list.flush()
	hdu_list.close()
	return scidata
	#ENDS MASK FREQUENCIES FUNCTION

def Cube_Freq2VelAxis(Cube2bMdfyAxs_ipt):
	cube       = scspc.read(Cube2bMdfyAxs_ipt) 
	cube_vel   = cube.with_spectral_unit(u.km / u.s,velocity_convention='radio')
	cube_vel.write(Cube2bMdfyAxs_ipt,overwrite=True)

def Cube_Vel2FreqAxis(Cube2bMdfyAxs_ipt):
	restfreq   = kwargs.get(restfreq,Header_Get(Cube2bMdfyAxs_ipt,'RESTFRQ'))
	cube       = scspc.read(Cube2bMdfyAxs_ipt) 
	cube_freq  = cube.with_spectral_unit(u.GHz,velocity_convention='radio',restvalue=restfreq*u.Ghz)
	cube_freq.write(Cube2bMdfyAxs_ipt,overwrite=True)

def Cube_Header_Get(cube_header_ipt,freq_rfr,*args, **kwargs):
	verbose    = kwargs.get('verbose',False) 
	redshift   = kwargs.get('redshift',0)
	freq_step  = kwargs.get('freq_step',Header_Get(cube_header_ipt,'CDELT3'))
	freq_init  = kwargs.get('freq_init',Header_Get(cube_header_ipt,'CRVAL3'))
	freq_obs_f = kwargs.get('freq_obs_f',Header_Get(cube_header_ipt,'RESTFRQ'))
	freq_obs   = kwargs.get('freq_obs',Redshifted_freq(freq_rfr,redshift))    

	vel_step   = kwargs.get('vel_step',None)

	DIM1 = Header_Get(cube_header_ipt,'NAXIS1')
	DIM2 = Header_Get(cube_header_ipt,'NAXIS2')
	DIM3 = Header_Get(cube_header_ipt,'NAXIS3')

	RVL1 = Header_Get(cube_header_ipt,'CRVAL1')
	RVL2 = Header_Get(cube_header_ipt,'CRVAL2')
	RVL3 = Header_Get(cube_header_ipt,'CRVAL3')

	RES1 = Header_Get(cube_header_ipt,'CDELT1')
	RES2 = Header_Get(cube_header_ipt,'CDELT2')

	cube                      = scspc.read(cube_header_ipt) 
	freq_step_shifted         = Redshifted_freq(freq_step,float(redshift))
	freq_init_shifted         = Redshifted_freq(freq_init,float(redshift))
	
	vel_step                  = abs(Thermal_Dopp_vel(freq_obs_f,freq_obs_f-abs((freq_step)),redshift_freq=redshift))
	vel_step_frq_bck          = Thermal_Dopp_freq(freq_obs_f,vel_step,redshift_freq=redshift) - freq_obs_f

	cube                      = cube.with_spectral_unit(u.Hz,velocity_convention='radio',rest_value=freq_obs_f* u.Hz)
	cube_vel                  = cube.with_spectral_unit(u.km / u.s,velocity_convention='radio')

	freq_max,freq_min = cube.spectral_extrema[-1].value,cube.spectral_extrema[0].value
	vel_max,vel_min   = cube_vel.spectral_extrema[-1].value,cube_vel.spectral_extrema[0].value
	
	if verbose == True:
		print (colored('Reading cube: ' + cube_header_ipt,'yellow'))
		print
		print ('Cube                                         : ',cube_header_ipt)
		print
		print (colored(cube,'yellow'))
		print
		print ('Dimensions                                   : ',DIM1,'X',DIM2,'X',DIM3)
		print ('CII restframe                                : ',freq_rfr)
		print ('CII will be observed at (fits file)          : ',freq_obs_f)
		print ('CII will be observed at                      : ',freq_obs)
		print
		print ('Frequency step                               : ',freq_step)
		print ('Initial frequency                            : ',freq_init)
		print ('Max,Min frequencies                          : ',freq_max,freq_min)
		print ('Velocity  step from Frequency        [kms-1] : ',vel_step)
		print ('Frequency step from Velocity                 : ',vel_step_frq_bck)
		print ('Max,Min velocities                   [kms-1] : ',vel_max,vel_min)
	elif verbose == False:
		pass
	return freq_rfr,freq_obs_f,DIM1,DIM2,DIM3,RVL1,RVL2,RVL3,RES1,RES2,freq_step,vel_step,freq_max,freq_min,vel_max,vel_min,cube.spectral_axis,cube_vel.spectral_axis

def Cube_Spatial_Smooth(CSS_ifn,CSS_krnsz,*args,**kwargs):
	CSS_ofn   = img_dir_res + (CSS_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-ssm.fits'
	cube      = scspc.read(CSS_ifn)
	beam      = radio_beam.Beam(major=CSS_krnsz*u.deg, minor=CSS_krnsz*u.deg, pa=0*u.deg)
	smth_cube = cube.convolve_to(beam)
	smth_cube.write(CSS_ofn,overwrite=True)
	return CSS_ofn

def Cube_CompleteSlices(CCS_ifn,CCS_xdim,CCS_ydim,add_low,add_top,*args,**kwargs):
	dest_dir = kwargs.get('dest_dir',None)
	verbose  = kwargs.get('verbose',False)

	
	cube2bcmp      = CCS_ifn
	cube           = scspc.read(cube2bcmp) 
	cube2bcmp_data = [apgtdt(cube2bcmp,memmap=False)]
	pre            = np.zeros((add_low,CCS_xdim,CCS_ydim))
	pst            = np.zeros((add_top,CCS_xdim,CCS_ydim))
	pre.fill(np.nan)
	pst.fill(np.nan)
	new            = np.concatenate((pre,cube2bcmp_data[0],pst),axis=0)
	wcs            = apwcs(cube2bcmp)

	if dest_dir != None:
		CCS_ofn        = dest_dir    + (CCS_ifn.split('.fits',1)[0]).rsplit('/',1)[-1]+ '-cmp.fits'
	elif dest_dir == None:
		CCS_ofn        = img_dir_res + (CCS_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-cmp.fits' 
	
	cube2bwritten  = scspc(data=new, wcs=wcs)
	cube2bwritten.write(CCS_ofn,overwrite=True)

	if verbose == True:
		print (colored(CCS_ifn,'yellow'))
		print (colored(CCS_ofn,'cyan'))
	elif verbose == False:
		pass
	return CCS_ofn

def Cube_Frequency_Smooth(CFS_ifn,freq_rfr,pre_res,pst_res,*args,**kwargs):
	redshift   = kwargs.get('redshift',0)
	freq_step  = kwargs.get('freq_step',Header_Get(CFS_ifn,'CDELT3'))
	freq_init  = kwargs.get('freq_init',Header_Get(CFS_ifn,'CRVAL3'))
	freq_obs_f = kwargs.get('freq_obs_f',Header_Get(CFS_ifn,'RESTFRQ'))   
	freq_obs   = kwargs.get('freq_obs',Redshifted_freq(freq_rfr,redshift))
	dest_dir   = kwargs.get('dest_dir',None)
	verbose    = kwargs.get('verbose',False)

	if dest_dir != None:
		CFS_ofn        = dest_dir    + (CFS_ifn.split('.fits',1)[0]).rsplit('/',1)[-1]+ '-fsm.fits'
	elif dest_dir == None:
		CFS_ofn        = img_dir_res + (CFS_ifn.split('.fits',1)[0]).rsplit('/',1)[-1]+ '-fsm.fits'

	cube               = scspc.read(CFS_ifn)
	CFS_ifn_header     = Cube_Header_Get(CFS_ifn,freq_rfr,**kwargs)
	fwhm_factor        = np.sqrt(8*np.log(2))
	current_resolution = pre_res * u.km/u.s
	target_resolution  = pst_res * u.km/u.s
	pixel_scale        = CFS_ifn_header[11]* u.km/u.s
	gaussian_width     = ((target_resolution**2 - current_resolution**2)**0.5 /
						pixel_scale / fwhm_factor)
	kernel             = Gaussian1DKernel(gaussian_width)
	new_cube           = cube.spectral_smooth(kernel)
	new_cube.write(CFS_ofn,overwrite=True)
	if verbose == True:
		print
		print (colored(CFS_ifn,'yellow'))
		print (colored(CFS_ofn,'cyan'))
	elif verbose == False:
		pass
	return CFS_ofn

def Cube_Frequency_Interpolate(CFI_ifn,freq_rfr,pre_res,pst_res,*args,**kwargs):
	redshift   = kwargs.get('redshift',0)
	freq_step  = kwargs.get('freq_step',Header_Get(CFI_ifn,'CDELT3'))
	freq_init  = kwargs.get('freq_init',Header_Get(CFI_ifn,'CRVAL3'))
	freq_obs_f = kwargs.get('freq_obs_f',Header_Get(CFI_ifn,'RESTFRQ'))   
	freq_obs   = kwargs.get('freq_obs',Redshifted_freq(freq_rfr,redshift))

	dest_dir   = kwargs.get('dest_dir',None)

	if dest_dir != None:
		CFI_ofn     = dest_dir    + (CFI_ifn.split('.fits',1)[0]).rsplit('/',1)[-1]+ '-fip.fits'
	elif dest_dir == None:
		CFI_ofn     = img_dir_res + (CFI_ifn.split('.fits',1)[0]).rsplit('/',1)[-1]+ '-fip.fits'

	cube            = scspc.read(CFI_ifn)

	CFI_ifn_header  = Cube_Header_Get(CFI_ifn,freq_rfr,**kwargs)
	vel_max,vel_min = CFI_ifn_header[14],CFI_ifn_header[15]
	frq_max,frq_min = CFI_ifn_header[12],CFI_ifn_header[13]

	new_axis_vel = np.arange(vel_min,vel_max,pst_res)*u.km/u.s
	new_axis_frq = np.arange(frq_min,frq_max,pst_res)*u.Hz
	fwhm_factor  = np.sqrt(8*np.log(2))

	cube_vel = cube.with_spectral_unit(u.km / u.s,velocity_convention='radio')

	interp_Cube = cube_vel.spectral_interpolate(new_axis_vel,suppress_smooth_warning=True)
	interp_Cube = interp_Cube.with_spectral_unit(u.Hz,velocity_convention='radio',rest_value=freq_obs* u.Hz)
	interp_Cube.write(CFI_ofn,overwrite=True)
	print

	if verbose == True:
		print (colored(CFI_ifn,'yellow'))
		print (colored(CFI_ofn,'cyan'))
	elif verbose == False:
		pass
	return CFI_ofn

def Cube_Spatial_Cut(CSC_ifn,Xbg_sz,Ybg_sz,Xsm_sz,Ysm_sz,*args,**kwargs):
	verbose  = kwargs.get('verbose' , False)
	dest_dir = kwargs.get('dest_dir',None)

	if dest_dir != None:
		CSC_ofn  = dest_dir    + (CSC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-cut.fits'
	elif dest_dir == None:
		CSC_ofn  = img_dir_res + (CSC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-cut.fits'
	
	cube     = scspc.read(CSC_ifn)

	if verbose == True:
		print 
		print (colored(CSC_ifn,'yellow'))
		print (colored(CSC_ofn,'cyan'))
	else:
		pass

	Ex, Ey   = ((Xbg_sz - Xsm_sz)/2, (Ybg_sz - Ysm_sz)/2)
	Fx, Fy   = ((Ex + Xsm_sz)      , Ey)
	Gx, Gy   = (Fx                 , (Ey+Xsm_sz))
	Hx, Hy   = (Ex                 , Gy)
	cut_cube = cube[:,Fy-1:Gy-1,Ex-1:Fx-1]
	cut_cube.write(CSC_ofn,overwrite=True)
	Header_History_Step(CSC_ifn,CSC_ofn)
	return CSC_ofn

def Cube_Spatial_Extract_Reg(CSER_ifn,X_C,Y_C,X_0,X_N,Y_0,Y_N,*args,**kwargs):
	dest_dir = kwargs.get('dest_dir',None)
	verbose  = kwargs.get('verbose',None)

	X_0, X_N  = X_C - X_0, X_C + X_N
	Y_0, Y_N  = Y_C - Y_0, Y_C + Y_N	

	if dest_dir != None:
		CSER_ofn       = dest_dir + (CSER_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-reg-'+str(X_N-X_0)+'as.fits'
	elif dest_dir == None:
		CSER_ofn       = stp_dir_res + (CSER_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-reg-'+str(X_N-X_0)+'as.fits'

	cube     = scspc.read(CSER_ifn)

	cut_cube  = cube[:,Y_0:Y_N,X_0:X_N]
	cut_cube.write(CSER_ofn,overwrite=True)
	if verbose == True:
		print 
		print (colored('Extracting region from Cube : ' + CSER_ifn,'yellow'))
		print (colored('Region subcube created      : ' + CSER_ofn,'cyan'))
	elif verbose == False:
		pass

	Header_Copy(CSER_ofn,CSER_ifn,'RESTFRQ',header_comment = 'Rest Frequency [GHz]')             
	Header_Copy(CSER_ofn,CSER_ifn,'STK_NUM',header_comment = 'Number of galaxies used for Stack')
	Header_Copy(CSER_ofn,CSER_ifn,'STZ_AVG', header_comment='Redshift Average')
	Header_Copy(CSER_ofn,CSER_ifn,'STZ_MED', header_comment='Redshift Median') 
	Header_Copy(CSER_ofn,CSER_ifn,'STL_AVG', header_comment='Redshift Median') 
	Header_Copy(CSER_ofn,CSER_ifn,'STL_MED', header_comment='Redshift Median') 
	
	return CSER_ofn

def Cube_Spatial_Extract_Circular(CSEC_ifn,X_C,Y_C,radii_px_inner,radii_as_inner,radii_px_outer,radii_as_outer,radii_px_msure,radii_as_msure,*args,**kwargs):
	dest_dir_stp  = kwargs.get('dest_dir_stp',None)
	verbose   = kwargs.get('verbose',None)
	plt_slcs  = kwargs.get('plt_slcs',False)

	x_ref     = kwargs.get('x_ref',0)
	y_ref     = kwargs.get('y_ref',0)

	z_avg     = kwargs.get('z_avg',Header_Get(CSEC_ifn,'STZ_AVG'))
	z_med     = kwargs.get('z_med',Header_Get(CSEC_ifn,'STZ_MED'))
	frq_r     = kwargs.get('frq_r',Header_Get(CSEC_ifn,'RESTFRQ'))

	z_f2l     = z_med

	prefix    = kwargs.get('prefix','')

	Splt_Hdr_Cmt_cp = kwargs.get('Splt_Hdr_Cmt_cp',None)

	if dest_dir_stp != None:
		CSEC_ofn_c_in = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_crc_in.fits'
		CSEC_ofn_d_in = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_dta_in.fits'
		CSEC_ofn_m_in = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_msk_in.fits'

		CSEC_ofn_c_ot = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_crc_ot.fits'
		CSEC_ofn_d_ot = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_dta_ot.fits'
		CSEC_ofn_m_ot = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_msk_ot.fits'

		CSEC_ofn_c_ms = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_crc_ms.fits'
		CSEC_ofn_d_ms = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_dta_ms.fits'
		CSEC_ofn_m_ms = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_msk_ms.fits'

	elif dest_dir_stp == None:
		CSEC_ofn_c_in = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_crc_in.fits'
		CSEC_ofn_d_in = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_dta_in.fits'
		CSEC_ofn_m_in = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_msk_in.fits'

		CSEC_ofn_c_ot = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_crc_ot.fits'
		CSEC_ofn_d_ot = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_dta_ot.fits'
		CSEC_ofn_m_ot = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_msk_ot.fits'

		CSEC_ofn_c_ms = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_crc_ms.fits'
		CSEC_ofn_d_ms = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_dta_ms.fits'
		CSEC_ofn_m_ms = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_msk_ms.fits'

	center    = PixCoord(X_C,Y_C)

	apt_ot    = CirclePixelRegion(center, radii_px_outer)
	mask_ot   = apt_ot.to_mask(mode='exact')

	apt_in    = CirclePixelRegion(center, radii_px_inner)
	mask_in   = apt_in.to_mask(mode='exact')

	apt_ms    = CirclePixelRegion(center, radii_px_msure)
	mask_ms   = apt_ms.to_mask(mode='exact')

	mask_in_amp = mask_in

	dif_row = mask_ot.data.shape[0]-mask_in_amp.data.shape[0]
	dif_col = mask_ot.data.shape[1]-mask_in_amp.data.shape[1]

	complete_row = dif_row / 2
	complete_col = dif_col / 2

	mask_in_amp.data = np.vstack((mask_in_amp.data, np.zeros((int(complete_row), mask_in_amp.data.shape[1]))))
	mask_in_amp.data = np.vstack((np.zeros((int(complete_row), mask_in_amp.data.shape[1])), mask_in_amp.data))

	mask_in_amp.data = np.hstack((mask_in_amp.data, np.zeros((mask_in_amp.data.shape[0],int(complete_col)))))
	mask_in_amp.data = np.hstack((np.zeros((mask_in_amp.data.shape[0],int(complete_col))), mask_in_amp.data))

	mask_ot.data = mask_ot.data - mask_in_amp.data

	hdu       =apfts.open(CSEC_ifn)[0]

	msk_crc_in     = []
	cub_crc_in     = []
	cub_crc_wgt_in = []

	msk_crc_ot     = []
	cub_crc_ot     = []
	cub_crc_wgt_ot = []

	msk_crc_ms     = []
	cub_crc_ms     = []
	cub_crc_wgt_ms = []

	f,x,y = hdu.shape

	freq_num,y_num,x_num = f,x,y
	nx_f2DG, ny_f2DG     = x_num,y_num 
	nx,ny                = nx_f2DG,ny_f2DG

	X0_f2DG   = kwargs.get('X0_f2DG',radii_px_inner)
	Y0_f2DG   = kwargs.get('Y0_f2DG',radii_px_inner)

	apt_in    = CirclePixelRegion(center, radii_px_inner)
	mask_in   = apt_in.to_mask(mode='exact')

	apt_ms    = CirclePixelRegion(center, radii_px_msure)
	mask_ms   = apt_ms.to_mask(mode='exact')

	mask_in.data[mask_in.data == 0] = np.nan
	mask_ot.data[mask_ot.data == 0] = np.nan

	for slice_freq in range(f):
		hdu_frq    =  hdu.data[slice_freq]

		data_in    = mask_in.cutout(hdu_frq)
		wgt_in     = mask_in.multiply(hdu_frq)

		data_ot    = mask_ot.cutout(hdu_frq)
		wgt_ot     = mask_ot.multiply(hdu_frq)

		data_ms    = mask_ms.cutout(hdu_frq)
		wgt_ms     = mask_ms.multiply(hdu_frq)

		msk_crc_in.append(mask_in.data)
		cub_crc_in.append(data_in)
		cub_crc_wgt_in.append(wgt_in)

		msk_crc_ot.append(mask_ot.data)
		cub_crc_ot.append(data_ot)
		cub_crc_wgt_ot.append(wgt_ot)

		msk_crc_ms.append(mask_ms.data)
		cub_crc_ms.append(data_ms)
		cub_crc_wgt_ms.append(wgt_ms)

	msk_crc_in     = np.asarray(msk_crc_in)
	cub_crc_in     = np.asarray(cub_crc_in)
	cub_crc_wgt_in = np.asarray(cub_crc_wgt_in)

	msk_crc_ot     = np.asarray(msk_crc_ot)
	cub_crc_ot     = np.asarray(cub_crc_ot)
	cub_crc_wgt_ot = np.asarray(cub_crc_wgt_ot)

	msk_crc_ms     = np.asarray(msk_crc_ms)
	cub_crc_ms     = np.asarray(cub_crc_ms)
	cub_crc_wgt_ms = np.asarray(cub_crc_wgt_ms)

	wcs            = apwcs(CSEC_ifn)

	cube2bwritten  = scspc(data=msk_crc_in, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_c_in,overwrite=True)

	cube2bwritten  = scspc(data=cub_crc_in, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_d_in,overwrite=True)

	cube2bwritten  = scspc(data=cub_crc_wgt_in, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_m_in,overwrite=True)

	cube2bwritten  = scspc(data=msk_crc_ot, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_c_ot,overwrite=True)

	cube2bwritten  = scspc(data=cub_crc_ot, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_d_ot,overwrite=True)

	cube2bwritten  = scspc(data=cub_crc_wgt_ot, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_m_ot,overwrite=True)

	cube2bwritten  = scspc(data=msk_crc_ms, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_c_ms,overwrite=True)

	cube2bwritten  = scspc(data=cub_crc_ms, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_d_ms,overwrite=True)

	cube2bwritten  = scspc(data=cub_crc_wgt_ms, wcs=wcs)
	cube2bwritten.write(CSEC_ofn_m_ms,overwrite=True)

	FLS_in=[CSEC_ofn_c_in,CSEC_ofn_d_in,CSEC_ofn_m_in]
	FLS_ot=[CSEC_ofn_c_ot,CSEC_ofn_d_ot,CSEC_ofn_m_ot]
	FLS_ms=[CSEC_ofn_c_ms,CSEC_ofn_d_ms,CSEC_ofn_m_ms]

	xmin_in,xmax_in,ymin_in,ymax_in = mask_in.bbox.extent
	xmin_ot,xmax_ot,ymin_ot,ymax_ot = mask_ot.bbox.extent
	xmin_ms,xmax_ms,ymin_ms,ymax_ms = mask_ms.bbox.extent

	for cubefile in FLS_in:
		Header_Add(cubefile,'XXT_MIN',xmin_in       , header_comment = 'Image Extent X MIN')
		Header_Add(cubefile,'XXT_MAX',xmax_in       , header_comment = 'Image Extent X MAX')
		Header_Add(cubefile,'YXT_MIN',ymin_in       , header_comment = 'Image Extent Y MIN')
		Header_Add(cubefile,'YXT_MAX',ymax_in       , header_comment = 'Image Extent Y MAX')
		Header_Add(cubefile,'XCT_FIT',X_C           , header_comment = 'Image Center X ')
		Header_Add(cubefile,'YCT_FIT',Y_C           , header_comment = 'Image Center Y ')
		Header_Add(cubefile,'RAD_EXT',radii_as_inner, header_comment = 'Image Extent Radii')
	
	for cubefile in FLS_ot:
		Header_Add(cubefile,'XXT_MIN',xmin_ot       , header_comment = 'Image Extent X MIN')
		Header_Add(cubefile,'XXT_MAX',xmax_ot       , header_comment = 'Image Extent X MAX')
		Header_Add(cubefile,'YXT_MIN',ymin_ot       , header_comment = 'Image Extent Y MIN')
		Header_Add(cubefile,'YXT_MAX',ymax_ot       , header_comment = 'Image Extent Y MAX')
		Header_Add(cubefile,'XCT_FIT',X_C           , header_comment = 'Image Center X ')
		Header_Add(cubefile,'YCT_FIT',Y_C           , header_comment = 'Image Center Y ')
		Header_Add(cubefile,'RAD_EXT',radii_as_outer, header_comment = 'Image Extent Radii')

	for cubefile in FLS_ms:
		Header_Add(cubefile,'XXT_MIN',xmin_ms       , header_comment = 'Image Extent X MIN')
		Header_Add(cubefile,'XXT_MAX',xmax_ms       , header_comment = 'Image Extent X MAX')
		Header_Add(cubefile,'YXT_MIN',ymin_ms       , header_comment = 'Image Extent Y MIN')
		Header_Add(cubefile,'YXT_MAX',ymax_ms       , header_comment = 'Image Extent Y MAX')
		Header_Add(cubefile,'XCT_FIT',X_C           , header_comment = 'Image Center X ')
		Header_Add(cubefile,'YCT_FIT',Y_C           , header_comment = 'Image Center Y ')
		Header_Add(cubefile,'RAD_EXT',radii_as_msure, header_comment = 'Image Extent Radii')

	csec_ofns = [CSEC_ofn_c_in,CSEC_ofn_d_in,CSEC_ofn_m_in,CSEC_ofn_c_ot,CSEC_ofn_d_ot,CSEC_ofn_m_ot,CSEC_ofn_c_ms,CSEC_ofn_d_ms,CSEC_ofn_m_ms]

	CDELT3_ORG = Header_Get(CSEC_ifn,'CDELT3')
	CUNIT3_ORG = Header_Get(CSEC_ifn,'CUNIT3')
	CRVAL3_ORG = Header_Get(CSEC_ifn,'CRVAL3')

	[Header_Updt(csec_ofns_cube,'CDELT3',CDELT3_ORG) for csec_ofns_cube in csec_ofns]
	[Header_Updt(csec_ofns_cube,'CUNIT3',CUNIT3_ORG) for csec_ofns_cube in csec_ofns]
	[Header_Updt(csec_ofns_cube,'CRVAL3',CRVAL3_ORG) for csec_ofns_cube in csec_ofns]


	[Header_Copy(csec_ofns_cube,CSEC_ifn,'RESTFRQ',header_comment = 'Rest Frequency [GHz]')              for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STK_NUM',header_comment = 'Number of galaxies used for Stack') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_AVG',header_comment = 'Redshift Average')                  for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_MED',header_comment = 'Redshift Median')                   for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_1SL',header_comment = 'Redshift 1 sgm lw lmt 15.9 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_1SH',header_comment = 'Redshift 1 sgm hg lmt 84.1 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_2SL',header_comment = 'Redshift 2 sgm lw lmt 2.30 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_2SH',header_comment = 'Redshift 2 sgm hg lmt 97.7 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_3SL',header_comment = 'Redshift 3 sgm lw lmt 0.20 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_3SH',header_comment = 'Redshift 3 sgm hg lmt 99.8 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_P25',header_comment = 'Redshift 25 pct')                   for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_P75',header_comment = 'Redshift 75 pct')                   for csec_ofns_cube in csec_ofns]

	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_AVG',header_comment = str(Splt_Hdr_Cmt_cp) + ' Average')               for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_MED',header_comment = str(Splt_Hdr_Cmt_cp) + ' Median')                for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_1SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 1 sgm lw lmt 15.9 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_1SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 1 sgm hg lmt 84.1 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_2SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 2 sgm lw lmt 2.30 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_2SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 2 sgm hg lmt 97.7 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_3SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 3 sgm lw lmt 0.20 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_3SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 3 sgm hg lmt 99.8 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_P25',header_comment = str(Splt_Hdr_Cmt_cp) + ' 25 pct')                for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_P75',header_comment = str(Splt_Hdr_Cmt_cp) + ' 75 pct')                for csec_ofns_cube in csec_ofns]

	if verbose == True:
		print 
		print (colored('Center: '+str(X_C)+','+str(Y_C),'yellow'))
		print (colored('Center: '+str(center),'yellow'))
		print (colored(CSEC_ifn,'cyan'))
		print (colored(CSEC_ofn_c_in,'yellow'))
		print (colored(CSEC_ofn_d_in,'yellow'))
		print (colored(CSEC_ofn_m_in,'yellow'))
		print (colored(CSEC_ofn_c_ot,'yellow'))
		print (colored(CSEC_ofn_d_ot,'yellow'))
		print (colored(CSEC_ofn_m_ot,'yellow'))
		print (colored(CSEC_ofn_c_ms,'yellow'))
		print (colored(CSEC_ofn_d_ms,'yellow'))
		print (colored(CSEC_ofn_m_ms,'yellow'))
		print
	elif verbose == False:
		pass

	return CSEC_ofn_c_in,CSEC_ofn_d_in,CSEC_ofn_m_in,CSEC_ofn_c_ot,CSEC_ofn_d_ot,CSEC_ofn_m_ot

def Cube_Spatial_Extract_Circular_2D(CSEC_ifn,X_C,Y_C,radii_px_inner,radii_as_inner,radii_px_outer,radii_as_outer,radii_px_msure,radii_as_msure,*args,**kwargs):
	dest_dir_stp  = kwargs.get('dest_dir_stp',None)
	verbose   = kwargs.get('verbose',None)

	x_ref     = kwargs.get('x_ref',0)
	y_ref     = kwargs.get('y_ref',0)

	z_avg     = kwargs.get('z_avg',Header_Get(CSEC_ifn,'STZ_AVG'))
	z_med     = kwargs.get('z_med',Header_Get(CSEC_ifn,'STZ_MED'))
	frq_r     = kwargs.get('frq_r',Header_Get(CSEC_ifn,'RESTFRQ'))
	z_f2l     = z_med

	prefix    = kwargs.get('prefix','')

	Splt_Hdr_Cmt_cp = kwargs.get('Splt_Hdr_Cmt_cp',None)
	if dest_dir_stp != None:
		CSEC_ofn_c_in = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_crc_in.fits'
		CSEC_ofn_d_in = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_dta_in.fits'
		CSEC_ofn_m_in = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_msk_in.fits'

		CSEC_ofn_c_ot = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_crc_ot.fits'
		CSEC_ofn_d_ot = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_dta_ot.fits'
		CSEC_ofn_m_ot = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_msk_ot.fits'

		CSEC_ofn_c_ms = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_crc_ms.fits'
		CSEC_ofn_d_ms = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_dta_ms.fits'
		CSEC_ofn_m_ms = dest_dir_stp + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_msk_ms.fits'

	elif dest_dir_stp == None:
		CSEC_ofn_c_in = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_crc_in.fits'
		CSEC_ofn_d_in = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_dta_in.fits'
		CSEC_ofn_m_in = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_inner)+'as_msk_in.fits'

		CSEC_ofn_c_ot = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_crc_ot.fits'
		CSEC_ofn_d_ot = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_dta_ot.fits'
		CSEC_ofn_m_ot = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_outer)+'as_msk_ot.fits'

		CSEC_ofn_c_ms = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_crc_ms.fits'
		CSEC_ofn_d_ms = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_dta_ms.fits'
		CSEC_ofn_m_ms = stp_dir_res + prefix + (CSEC_ifn.split('.fits',1)[0]).rsplit('/',1)[-1] + '-crc-'+str(radii_as_msure)+'as_msk_ms.fits'

	center    = PixCoord(X_C,Y_C)

	apt_ot    = CirclePixelRegion(center, radii_px_outer)
	mask_ot   = apt_ot.to_mask(mode='exact')

	apt_in    = CirclePixelRegion(center, radii_px_inner)
	mask_in   = apt_in.to_mask(mode='exact')

	apt_ms    = CirclePixelRegion(center, radii_px_msure)
	mask_ms   = apt_ms.to_mask(mode='exact')

	mask_in_amp = mask_in

	dif_row = mask_ot.data.shape[0]-mask_in_amp.data.shape[0]
	dif_col = mask_ot.data.shape[1]-mask_in_amp.data.shape[1]

	complete_row = dif_row / 2
	complete_col = dif_col / 2

	mask_in_amp.data = np.vstack((mask_in_amp.data, np.zeros((complete_row, mask_in_amp.data.shape[1]))))
	mask_in_amp.data = np.vstack((np.zeros((complete_row, mask_in_amp.data.shape[1])), mask_in_amp.data))

	mask_in_amp.data = np.hstack((mask_in_amp.data, np.zeros((mask_in_amp.data.shape[0],complete_col))))
	mask_in_amp.data = np.hstack((np.zeros((mask_in_amp.data.shape[0],complete_col)), mask_in_amp.data))

	mask_ot.data = mask_ot.data - mask_in_amp.data

	hdu       =apfts.open(CSEC_ifn)[0]

	msk_crc_in     = []
	cub_crc_in     = []
	cub_crc_wgt_in = []

	msk_crc_ot     = []
	cub_crc_ot     = []
	cub_crc_wgt_ot = []

	msk_crc_ms     = []
	cub_crc_ms     = []
	cub_crc_wgt_ms = []

	x,y = hdu.shape
	f= 1

	freq_num,y_num,x_num = f,x,y
	nx_f2DG, ny_f2DG     = x_num,y_num 
	nx,ny                = nx_f2DG,ny_f2DG

	X0_f2DG   = kwargs.get('X0_f2DG',radii_px_inner)
	Y0_f2DG   = kwargs.get('Y0_f2DG',radii_px_inner)

	apt_in    = CirclePixelRegion(center, radii_px_inner)
	mask_in   = apt_in.to_mask(mode='exact')

	apt_ms    = CirclePixelRegion(center, radii_px_msure)
	mask_ms   = apt_ms.to_mask(mode='exact')

	mask_in.data[mask_in.data == 0] = np.nan
	mask_ot.data[mask_ot.data == 0] = np.nan

	hdu_frq    =  hdu.data

	data_in    = mask_in.cutout(hdu_frq)
	wgt_in     = mask_in.multiply(hdu_frq)

	data_ot    = mask_ot.cutout(hdu_frq)
	wgt_ot     = mask_ot.multiply(hdu_frq)

	data_ms    = mask_ms.cutout(hdu_frq)
	wgt_ms     = mask_ms.multiply(hdu_frq)

	msk_crc_in.append(mask_in.data)
	cub_crc_in.append(data_in)
	cub_crc_wgt_in.append(wgt_in)

	msk_crc_ot.append(mask_ot.data)
	cub_crc_ot.append(data_ot)
	cub_crc_wgt_ot.append(wgt_ot)

	msk_crc_ms.append(mask_ms.data)
	cub_crc_ms.append(data_ms)
	cub_crc_wgt_ms.append(wgt_ms)

	msk_crc_in     = np.asarray(msk_crc_in)
	cub_crc_in     = np.asarray(cub_crc_in)
	cub_crc_wgt_in = np.asarray(cub_crc_wgt_in)

	msk_crc_ot     = np.asarray(msk_crc_ot)
	cub_crc_ot     = np.asarray(cub_crc_ot)
	cub_crc_wgt_ot = np.asarray(cub_crc_wgt_ot)

	msk_crc_ms     = np.asarray(msk_crc_ms)
	cub_crc_ms     = np.asarray(cub_crc_ms)
	cub_crc_wgt_ms = np.asarray(cub_crc_wgt_ms)

	wcs            = apwcs(CSEC_ifn)

	Wrt_FITS_File(msk_crc_in,CSEC_ofn_c_in)
	Wrt_FITS_File(cub_crc_in,CSEC_ofn_d_in)
	Wrt_FITS_File(cub_crc_wgt_in,CSEC_ofn_m_in)
	Wrt_FITS_File(msk_crc_ot,CSEC_ofn_c_ot)
	Wrt_FITS_File(cub_crc_ot,CSEC_ofn_d_ot)
	Wrt_FITS_File(cub_crc_wgt_ot,CSEC_ofn_m_ot)
	Wrt_FITS_File(msk_crc_ms,CSEC_ofn_c_ms)
	Wrt_FITS_File(cub_crc_ms,CSEC_ofn_d_ms)
	Wrt_FITS_File(cub_crc_wgt_ms,CSEC_ofn_m_ms)

	FLS_in=[CSEC_ofn_c_in,CSEC_ofn_d_in,CSEC_ofn_m_in]
	FLS_ot=[CSEC_ofn_c_ot,CSEC_ofn_d_ot,CSEC_ofn_m_ot]
	FLS_ms=[CSEC_ofn_c_ms,CSEC_ofn_d_ms,CSEC_ofn_m_ms]

	xmin_in,xmax_in,ymin_in,ymax_in = mask_in.bbox.extent
	xmin_ot,xmax_ot,ymin_ot,ymax_ot = mask_ot.bbox.extent
	xmin_ms,xmax_ms,ymin_ms,ymax_ms = mask_ms.bbox.extent

	for cubefile in FLS_in:
		Header_Add(cubefile,'XXT_MIN',xmin_in       , header_comment = 'Image Extent X MIN')
		Header_Add(cubefile,'XXT_MAX',xmax_in       , header_comment = 'Image Extent X MAX')
		Header_Add(cubefile,'YXT_MIN',ymin_in       , header_comment = 'Image Extent Y MIN')
		Header_Add(cubefile,'YXT_MAX',ymax_in       , header_comment = 'Image Extent Y MAX')
		Header_Add(cubefile,'XCT_FIT',X_C           , header_comment = 'Image Center X ')
		Header_Add(cubefile,'YCT_FIT',Y_C           , header_comment = 'Image Center Y ')
		Header_Add(cubefile,'RAD_EXT',radii_as_inner, header_comment = 'Image Extent Radii')
	
	for cubefile in FLS_ot:
		Header_Add(cubefile,'XXT_MIN',xmin_ot       , header_comment = 'Image Extent X MIN')
		Header_Add(cubefile,'XXT_MAX',xmax_ot       , header_comment = 'Image Extent X MAX')
		Header_Add(cubefile,'YXT_MIN',ymin_ot       , header_comment = 'Image Extent Y MIN')
		Header_Add(cubefile,'YXT_MAX',ymax_ot       , header_comment = 'Image Extent Y MAX')
		Header_Add(cubefile,'XCT_FIT',X_C           , header_comment = 'Image Center X ')
		Header_Add(cubefile,'YCT_FIT',Y_C           , header_comment = 'Image Center Y ')
		Header_Add(cubefile,'RAD_EXT',radii_as_outer, header_comment = 'Image Extent Radii')

	for cubefile in FLS_ms:
		Header_Add(cubefile,'XXT_MIN',xmin_ms       , header_comment = 'Image Extent X MIN')
		Header_Add(cubefile,'XXT_MAX',xmax_ms       , header_comment = 'Image Extent X MAX')
		Header_Add(cubefile,'YXT_MIN',ymin_ms       , header_comment = 'Image Extent Y MIN')
		Header_Add(cubefile,'YXT_MAX',ymax_ms       , header_comment = 'Image Extent Y MAX')
		Header_Add(cubefile,'XCT_FIT',X_C           , header_comment = 'Image Center X ')
		Header_Add(cubefile,'YCT_FIT',Y_C           , header_comment = 'Image Center Y ')
		Header_Add(cubefile,'RAD_EXT',radii_as_msure, header_comment = 'Image Extent Radii')

	csec_ofns = [CSEC_ofn_c_in,CSEC_ofn_d_in,CSEC_ofn_m_in,CSEC_ofn_c_ot,CSEC_ofn_d_ot,CSEC_ofn_m_ot,CSEC_ofn_c_ms,CSEC_ofn_d_ms,CSEC_ofn_m_ms]

	[Header_Copy(csec_ofns_cube,CSEC_ifn,'RESTFRQ',header_comment = 'Rest Frequency [GHz]')              for csec_ofns_cube in csec_ofns]
	
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_AVG',header_comment = 'Redshift Average')                  for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_MED',header_comment = 'Redshift Median')                   for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_1SL',header_comment = 'Redshift 1 sgm lw lmt 15.9 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_1SH',header_comment = 'Redshift 1 sgm hg lmt 84.1 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_2SL',header_comment = 'Redshift 2 sgm lw lmt 2.30 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_2SH',header_comment = 'Redshift 2 sgm hg lmt 97.7 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_3SL',header_comment = 'Redshift 3 sgm lw lmt 0.20 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_3SH',header_comment = 'Redshift 3 sgm hg lmt 99.8 pct')    for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_P25',header_comment = 'Redshift 25 pct')                   for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STZ_P75',header_comment = 'Redshift 75 pct')                   for csec_ofns_cube in csec_ofns]

	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_AVG',header_comment = str(Splt_Hdr_Cmt_cp) + ' Average')               for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_MED',header_comment = str(Splt_Hdr_Cmt_cp) + ' Median')                for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_1SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 1 sgm lw lmt 15.9 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_1SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 1 sgm hg lmt 84.1 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_2SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 2 sgm lw lmt 2.30 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_2SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 2 sgm hg lmt 97.7 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_3SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 3 sgm lw lmt 0.20 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_3SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 3 sgm hg lmt 99.8 pct') for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_P25',header_comment = str(Splt_Hdr_Cmt_cp) + ' 25 pct')                for csec_ofns_cube in csec_ofns]
	[Header_Copy(csec_ofns_cube,CSEC_ifn,'STS_P75',header_comment = str(Splt_Hdr_Cmt_cp) + ' 75 pct')                for csec_ofns_cube in csec_ofns]

	if verbose == True:
		print 
		print (colored('Center: '+str(X_C)+','+str(Y_C),'yellow'))
		print (colored('Center: '+str(center),'yellow'))
		print (colored(CSEC_ifn,'cyan'))
		print (colored(CSEC_ofn_c_in,'yellow'))
		print (colored(CSEC_ofn_d_in,'yellow'))
		print (colored(CSEC_ofn_m_in,'yellow'))
		print (colored(CSEC_ofn_c_ot,'yellow'))
		print (colored(CSEC_ofn_d_ot,'yellow'))
		print (colored(CSEC_ofn_m_ot,'yellow'))
		print (colored(CSEC_ofn_c_ms,'yellow'))
		print (colored(CSEC_ofn_d_ms,'yellow'))
		print (colored(CSEC_ofn_m_ms,'yellow'))
		print
	elif verbose == False:
		pass

	return CSEC_ofn_c_in,CSEC_ofn_d_in,CSEC_ofn_m_in,CSEC_ofn_c_ot,CSEC_ofn_d_ot,CSEC_ofn_m_ot

def Cube_Slice(cube_slice_ipt,width,freq_rfr,*args, **kwargs):
	verbose    = kwargs.get('verbose'   ,False) 
	redshift   = kwargs.get('redshift'  ,0)
	freq_obs   = kwargs.get('freq_obs'  ,Redshifted_freq(freq_rfr,redshift))
	freq_step  = kwargs.get('freq_step' ,Header_Get(cube_slice_ipt,'CDELT3'))
	freq_init  = kwargs.get('freq_init' ,Header_Get(cube_slice_ipt,'CRVAL3'))
	freq_obs_f = kwargs.get('freq_obs_f',Header_Get(cube_slice_ipt,'RESTFRQ'))
	vel_step   = kwargs.get('vel_step'  ,None)

	dest_dir_plt = kwargs.get('dest_dir_plt'    ,None)

	if dest_dir_plt != None:
		cube_slice_opt = dest_dir_plt    + (cube_slice_ipt.split('.fits',1)[0]).rsplit('/',1)[-1]+ '-slc.fits'
	elif dest_dir_plt == None:
		cube_slice_opt = img_dir_res + (cube_slice_ipt.split('.fits',1)[0]).rsplit('/',1)[-1]+ '-slc.fits'

	wcs                       = (apwcs(cube_slice_ipt))
	cube                      = scspc.read(cube_slice_ipt)
	freq_step_shifted         = Redshifted_freq(freq_step,float(redshift))
	freq_init_shifted         = Redshifted_freq(freq_init,float(redshift))
	
	vel_step                  = abs(Thermal_Dopp_vel(freq_obs,freq_obs-abs((freq_step)),redshift_freq=redshift))
	vel_step_frq_bck          = Thermal_Dopp_freq(freq_obs,vel_step,redshift_freq=redshift) - freq_obs

	vel_step_shifted          = abs(Thermal_Dopp_vel(freq_obs,freq_obs-abs(freq_step_shifted),redshift_freq=redshift))
	vel_step_frq_bck_shifted  = Thermal_Dopp_freq(freq_obs,vel_step_shifted,redshift_freq=redshift) - freq_obs

	subcube_frequency         = Thermal_Dopp_freq(freq_obs,subcube_width,redshift_freq=redshift)-freq_obs
	freq_min                  = freq_obs + subcube_frequency
	freq_max                  = freq_obs - subcube_frequency

	subcube_frequency_shifted = Thermal_Dopp_freq(freq_obs,subcube_width,redshift_freq=redshift)-freq_obs
	freq_min_shifted          = freq_obs + subcube_frequency
	freq_max_shifted          = freq_obs - subcube_frequency

	cube                      = cube.with_spectral_unit(u.Hz,velocity_convention='radio',rest_value=freq_obs* u.Hz)
	slab_frq                  = cube.spectral_slab(freq_min * u.Hz, freq_max * u.Hz) 

	cube_vel                  = cube.with_spectral_unit(u.km / u.s,velocity_convention='radio')
	slab_vel                  = cube_vel.spectral_slab(-subcube_width * u.km / u.s, subcube_width * u.km / u.s) 

	slab_vel.write(cube_slice_opt,overwrite=True)

	freq_step_subcube         = Header_Get(cube_slice_opt,'CDELT3')
	freq_init_subcube         = Header_Get(cube_slice_opt,'CRVAL3')
	sbcb                      = scspc.read(cube_slice_opt)
	sbcb_vel                  = sbcb.with_spectral_unit(u.km / u.s,velocity_convention='radio')

	if verbose == True:

		print
		print (colored('Reading cube: ' + cube_slice_ipt,'yellow'))

		print
		print ('CII restframe                                : ',freq_rfr)
		print ('CII will be observed at (fits file)          : ',freq_obs_f)
		print ('CII will be observed at                      : ',freq_obs)
		print
		print ('Initial frequency       (fits file)          : ',freq_init)
		print ('Delta frequency step    (fits file)          : ',freq_step)
		print ('Velocity  from Frequency             [kms-1] : ',vel_step)
		print ('Frequency from Velocity                      : ',vel_step_frq_bck)
		print
		print ('Initial frequency         shifted            : ',freq_init_shifted)
		print ('Delta frequency step      shifted            : ',freq_step_shifted)
		print ('Velocity  from Frequency  shifted    [kms-1] : ',vel_step_shifted)
		print ('Frequency from Velocity   shifted            : ',vel_step_frq_bck_shifted)
		print
		print ('The Subcube width                            : ')
		print ('Velocity width                       [kms-1] : ',subcube_width)
		print ('Frequency width                              : ',subcube_frequency)
		print ('The Subcube frequency limits                 : ',freq_min,freq_max)

		print ('Frequency width shifted                      : ',subcube_frequency_shifted)
		print ('The Subcube frequency limits shifted         : ',freq_min_shifted,freq_max_shifted)
		print
		print ('Subcube')
		print( sbcb)

		print
		print ('Original Cube                                : ',cube_slice_ipt)
		print ('Number of slices                             : ',len(cube.spectral_axis))
		print ('Frequency step      (original cube)          : ',freq_step)
		print ('Initial frequency   (original cube)          : ',freq_init)
		print ('Max,Min frequencies (cube)                   : ',cube.spectral_extrema[-1].value,cube.spectral_extrema[0].value)
		print
		print (cube.spectral_axis[0],cube.spectral_axis[1],'....',cube.spectral_axis[-2],cube.spectral_axis[-1])
		print
		print ('SubCube                                      : ',cube_slice_opt)
		print ('Number of slices    (frequency)              : ',len(sbcb.spectral_axis),len(slab_frq.spectral_axis))
		print ('Number of slices    (velocity)               : ',len(slab_vel.spectral_axis),len(sbcb_vel.spectral_axis))
		print ('Frequency step      (subcube wrt)            : ',freq_step_subcube)
		print ('Initial frequency   (subcube wrt)            : ',freq_init_subcube)
		print ('Max,Min frequencies (subcube)                : ',slab_frq.spectral_extrema[-1].value,slab_frq.spectral_extrema[0].value)
		print ('Max,Min frequencies (subcube wrt)            : ',sbcb.spectral_extrema[-1].value,sbcb.spectral_extrema[0].value)
		print ('Max,Min velocities  (subcube)                : ',slab_vel.spectral_extrema[-1].value,slab_vel.spectral_extrema[0].value)
		print ('Max,Min velocities  (subcube wrt)            : ',sbcb_vel.spectral_extrema[-1].value,sbcb_vel.spectral_extrema[0].value)
		print (slab_frq.spectral_axis[0],slab_frq.spectral_axis[1],'....',slab_frq.spectral_axis[-2],slab_frq.spectral_axis[-1])
		print (slab_vel.spectral_axis[0],slab_vel.spectral_axis[1],'....',slab_vel.spectral_axis[-2],slab_vel.spectral_axis[-1])
		print
		print (sbcb.spectral_axis[0],sbcb.spectral_axis[1],'....',sbcb.spectral_axis[-2],sbcb.spectral_axis[-1])
		print (sbcb_vel.spectral_axis[0],sbcb_vel.spectral_axis[1],'....',sbcb_vel.spectral_axis[-2],sbcb_vel.spectral_axis[-1])
		print
		print (colored('Subcube fits file: '+ str(cube_slice_opt),'cyan'))
	elif verbose == False:
		pass

	Header_Updt(cube_slice_ipt,'CDELT3',freq_step)
	Header_Get_Add(cube_slice_ipt,'h_s_0'  ,str((cube_slice_ipt.rsplit('.',1)[0]).rsplit('/',1)[-1]),header_comment='History Step Init')
	Header_Get_Add(cube_slice_ipt,'h_s_c'  ,0       ,header_comment='History Step Last')
	Header_Get_Add(cube_slice_ipt,'z'      ,redshift,header_comment='Redshift')
	Header_Get_Add(cube_slice_ipt,'FRG_RFR',freq_rfr,header_comment='Restframe Frequency')
	Header_Get_Add(cube_slice_ipt,'FRG_OBS',freq_obs,header_comment='Observed Frequency')
	Header_Get_Add(cube_slice_opt,'z'      ,redshift,header_comment='Redshift')
	Header_Get_Add(cube_slice_opt,'FRG_RFR',freq_rfr,header_comment='Restframe Frequency')
	Header_Get_Add(cube_slice_opt,'FRG_OBS',freq_obs,header_comment='Observed Frequency')
	Header_History_Step(cube_slice_ipt,cube_slice_opt)
	print (colored(cube_slice_opt,'yellow'))
	return cube_slice_opt

def Cube_fit_2D_Gaussian(Cube2bFit,*args,**kwargs):
	slc_nmb        = kwargs.get('slc_nmb' ,None)
	dest_dir       = kwargs.get('dest_dir',None)
	verbose        = kwargs.get('verbose' ,None)
	clp_fnc        = kwargs.get('clp_fnc' ,'sum')
	sgm_fnc        = kwargs.get('sgm_fnc' ,'avg')
	circular       = kwargs.get('circular',True)
	x_ref          = kwargs.get('x_ref',0)
	y_ref          = kwargs.get('y_ref',0)

	dest_dir_plt   = kwargs.get('dest_dir_plt',None)
	dest_dir_clp   = kwargs.get('dest_dir_clp',None)

	z_avg          = kwargs.get('z_avg',Header_Get(Cube2bFit,'STZ_AVG'))
	z_med          = kwargs.get('z_med',Header_Get(Cube2bFit,'STZ_MED'))
	frq_r          = kwargs.get('frq_r',restframe_frequency)
	z_f2l          = z_med

	sgm_wgth_tms   = kwargs.get('sgm_wgth_tms','5sgm')
	Cube2bFit_Err  = kwargs.get('Cube2bFit_Err', None)
	fit_type       = kwargs.get('fit_type','scipy')

	ref_wdt_lne    = kwargs.get('ref_wdt_lne',False)
	ref_wdt_fle    = kwargs.get('ref_wdt_fle',None)

	src_sze_fxd    = kwargs.get('src_sze_fxd',True)

	Splt_Hdr_Cmt_cp = kwargs.get('Splt_Hdr_Cmt_cp',None)

	if ref_wdt_lne == True:
		Cube2bFit_Hdr = ref_wdt_fle
	elif ref_wdt_lne == False:
		Cube2bFit_Hdr = Cube2bFit

	if slc_nmb == None:
		if sgm_fnc == 'avg':
			slice_nmbr   = (Header_Get(Cube2bFit_Hdr,'MAX_SNA'))
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTA_SIG'))
			slice_fwhm   = (Header_Get(Cube2bFit_Hdr,'FTA_FWH'))
			slice_cct    = (Header_Get(Cube2bFit_Hdr,'FTA_CCT'))
			slice_1sg    = (Header_Get(Cube2bFit_Hdr,'FTA_1SG'))
			slice_2sg    = (Header_Get(Cube2bFit_Hdr,'FTA_2SG'))
			slice_3sg    = (Header_Get(Cube2bFit_Hdr,'FTA_3SG'))
			slice_4sg    = (Header_Get(Cube2bFit_Hdr,'FTA_4SG'))
			slice_5sg    = (Header_Get(Cube2bFit_Hdr,'FTA_5SG'))
			slice_1fw    = (Header_Get(Cube2bFit_Hdr,'FTA_1FW'))
			slice_2fw    = (Header_Get(Cube2bFit_Hdr,'FTA_2FW'))
			slice_sigm_err = (Header_Get(Cube2bFit_Hdr,'FTA_SGE'))
			slice_fwhm_err = sigma2fwhm(slice_sigm_err)
		elif sgm_fnc == 'med':
			slice_nmbr   = (Header_Get(Cube2bFit_Hdr,'MAX_SNM'))
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTM_SIG'))
			slice_fwhm   = (Header_Get(Cube2bFit_Hdr,'FTM_FWH'))
			slice_cct    = (Header_Get(Cube2bFit_Hdr,'FTM_CCT'))
			slice_1sg    = (Header_Get(Cube2bFit_Hdr,'FTM_1SG'))
			slice_2sg    = (Header_Get(Cube2bFit_Hdr,'FTM_2SG'))
			slice_3sg    = (Header_Get(Cube2bFit_Hdr,'FTM_3SG'))
			slice_4sg    = (Header_Get(Cube2bFit_Hdr,'FTM_4SG'))
			slice_5sg    = (Header_Get(Cube2bFit_Hdr,'FTM_5SG'))
			slice_1fw    = (Header_Get(Cube2bFit_Hdr,'FTM_1FW'))
			slice_2fw    = (Header_Get(Cube2bFit_Hdr,'FTM_2FW'))
			slice_sigm_err = (Header_Get(Cube2bFit_Hdr,'FTM_SGE'))
			slice_fwhm_err = sigma2fwhm(slice_sigm_err)
		elif sgm_fnc == 'sum':
			slice_nmbr   = (Header_Get(Cube2bFit_Hdr,'MAX_SNS'))
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTS_SIG'))
			slice_fwhm   = (Header_Get(Cube2bFit_Hdr,'FTS_FWH'))
			slice_cct    = (Header_Get(Cube2bFit_Hdr,'FTS_CCT'))
			slice_1sg    = (Header_Get(Cube2bFit_Hdr,'FTS_1SG'))
			slice_2sg    = (Header_Get(Cube2bFit_Hdr,'FTS_2SG'))
			slice_3sg    = (Header_Get(Cube2bFit_Hdr,'FTS_3SG'))
			slice_4sg    = (Header_Get(Cube2bFit_Hdr,'FTS_4SG'))
			slice_5sg    = (Header_Get(Cube2bFit_Hdr,'FTS_5SG'))
			slice_1fw    = (Header_Get(Cube2bFit_Hdr,'FTS_1FW'))
			slice_2fw    = (Header_Get(Cube2bFit_Hdr,'FTS_2FW'))
			slice_sigm_err = (Header_Get(Cube2bFit_Hdr,'FTS_SGE'))
			slice_fwhm_err = sigma2fwhm(slice_sigm_err)

		if dest_dir_clp != None:
			Cube2bclp_2D_opt = dest_dir_clp  + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+clp_fnc+'.fits'
		elif dest_dir_clp == None:
			Cube2bclp_2D_opt = stp_dir_res + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+clp_fnc+'.fits'
		if dest_dir_plt != None:
			PLOTFILENAME     = str(dest_dir_plt) + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'.pdf'
			PLOTFILENAME_FMR = str(dest_dir_plt) + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-RSD.pdf'

			FITSFILENAME_MDL = dest_dir_clp + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-MDL.fits'
			FITSFILENAME_RSD = dest_dir_clp + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-RSD.fits'
		elif dest_dir_plt == None:
			PLOTFILENAME     = ana_dir_plt + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'.pdf'
			PLOTFILENAME_FMR = ana_dir_plt + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-RSD.pdf'

			FITSFILENAME_MDL = stp_dir_res + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-MDL.fits'
			FITSFILENAME_RSD = stp_dir_res + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-RSD.fits'

	elif slc_nmb != None:
		slice_nmbr = 'CSL'
		if dest_dir_clp != None:
			Cube2bclp_2D_opt = dest_dir_clp   + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+slice_nmbr+str(slc_nmb)+'.fits'
		elif dest_dir_clp == None:
			Cube2bclp_2D_opt = stp_dir_res + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+slice_nmbr+str(slc_nmb)+'.fits'
		if dest_dir_plt != None:
			PLOTFILENAME     = str(dest_dir_plt) + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+slice_nmbr+str(slc_nmb)+'.pdf'
			PLOTFILENAME_FMR = str(dest_dir_plt) + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+slice_nmbr+str(slc_nmb)+'-RSD.pdf'

			FITSFILENAME_MDL = dest_dir_clp + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-MDL.fits'
			FITSFILENAME_RSD = dest_dir_clp + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-RSD.fits'
		elif dest_dir_plt == None:
			PLOTFILENAME     = ana_dir_plt + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+slice_nmbr+str(slc_nmb)+'.pdf'
			PLOTFILENAME_FMR = ana_dir_plt + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+slice_nmbr+str(slc_nmb)+'-RSD.pdf'

			FITSFILENAME_MDL = stp_dir_res + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-MDL.fits'
			FITSFILENAME_RSD = stp_dir_res + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-RSD.fits'


	Cube_Info    = Cube_Header_Get(Cube2bFit,frq_r* u.Hz)
	FRQ_AXS      = Cube_Info[16].value
	VEL_AXS      = Cube_Info[17].value

	scale_deg    = Header_Get(Cube2bFit,'CDELT2')
	scale_arcsec = scale_deg*3600

	cube_data    = np.asarray(apgtdt(Cube2bFit,memmap=False))
	slice_cwdt   = (Header_Get(Cube2bFit_Hdr,'STT_VEL'))

	if slc_nmb == None:
		if sgm_wgth_tms == '1sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*1)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*1))
			slice_nblw = int(slice_cct - slice_1sg)
			slice_nbhg = int(slice_cct + slice_1sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vlct = VEL_AXS[int(slice_cct)]
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_1sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*1
			tlt_ch_nmb = (dlt_ch_nmb*2)+1

		elif sgm_wgth_tms == '2sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*2)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*2))
			slice_nblw = int(slice_cct - slice_2sg)
			slice_nbhg = int(slice_cct + slice_2sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vlct = VEL_AXS[int(slice_cct)]
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_2sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*2
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
		elif sgm_wgth_tms == '3sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*3)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*3))
			slice_nblw = int(slice_cct - slice_3sg)
			slice_nbhg = int(slice_cct + slice_3sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vlct = VEL_AXS[int(slice_cct)]
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_3sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*3
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
		elif sgm_wgth_tms == '4sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*4)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*4))
			slice_nblw = int(slice_cct - slice_4sg)
			slice_nbhg = int(slice_cct + slice_4sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vlct = VEL_AXS[int(slice_cct)]
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_4sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*4
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
		elif sgm_wgth_tms == '5sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*5)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*5))
			slice_nblw = int(slice_cct - slice_5sg)
			slice_nbhg = int(slice_cct + slice_5sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vlct = VEL_AXS[int(slice_cct)]
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_5sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*5
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
		elif sgm_wgth_tms == 'slice_1fw':
			print
			print (colored('Velocity width interval (fwhm) : ' + str(sgm_wgth_tms),'yellow'))
			print (colored('Cube fitting 1Xsigm (avg) km/s : ' + str(fwhm2sigma(slice_fwhm)),'yellow'))
			print (colored('Cube fitting 1Xfwhm (avg) km/s : ' + str(slice_fwhm),'yellow'))
			print

			slice_nblw = int(slice_cct - slice_1fw)
			slice_nbhg = int(slice_cct + slice_1fw)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vlct = VEL_AXS[int(slice_cct)]
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_1fw)
			slice_wdnb = dlt_ch_nmb
			slice_sig  = fwhm2sigma(slice_fwhm)
			vl_wdt_clp = slice_fwhm
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
		elif sgm_wgth_tms == 'slice_2fw':
			print
			print (colored('Velocity width interval (fwhm) : ' + str(sgm_wgth_tms),'yellow'))
			print (colored('Cube fitting 2Xsigm (avg) km/s : ' + str(fwhm2sigma(slice_fwhm*2)),'yellow'))
			print (colored('Cube fitting 2Xfwhm (avg) km/s : ' + str(slice_fwhm*2),'yellow'))
			print			
			slice_nblw = int(slice_cct - slice_2fw)
			slice_nbhg = int(slice_cct + slice_2fw)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vlct = VEL_AXS[int(slice_cct)]
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_2fw)
			slice_wdnb = dlt_ch_nmb
			slice_sigm = fwhm2sigma(slice_fwhm*2)
			vl_wdt_clp = slice_fwhm
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
	elif slc_nmb != None:
		slice_nblw = slc_nmb
		slice_nbhg = slc_nmb
		slice_vlct = 0
		slice_vllw = 0
		slice_vlhg = 0
		dlt_ch_nmb = 1
		slice_wdnb = dlt_ch_nmb
		vl_wdt_clp = slice_cwdt
		tlt_ch_nmb = 1
		if sgm_fnc == 'avg':
			slice_sigm     = (Header_Get(Cube2bFit_Hdr,'FTA_SIG'))
			slice_sigm_err = (Header_Get(Cube2bFit_Hdr,'FTA_SGE'))
			slice_fwhm_err = sigma2fwhm(slice_sigm_err)
		elif sgm_fnc == 'med':
			slice_sigm     = (Header_Get(Cube2bFit_Hdr,'FTM_SIG'))
			slice_sigm_err = (Header_Get(Cube2bFit_Hdr,'FTM_SGE'))
			slice_fwhm_err = sigma2fwhm(slice_sigm_err)
		elif sgm_fnc == 'sum':
			slice_sigm     = (Header_Get(Cube2bFit_Hdr,'FTS_SIG'))
			slice_sigm_err = (Header_Get(Cube2bFit_Hdr,'FTS_SGE'))
			slice_fwhm_err = sigma2fwhm(slice_sigm_err)
		slice_sigm = slice_sigm
		pass

	print
	print (colored('Channels                       : ' + str(slice_nblw)+'-'+str(slice_nbhg),'yellow'))
	print
	if slc_nmb != None:
		data_2b_plot = cube_data[slc_nmb]
		Message1 = 'Fitting gaussian with slice number : ' + str(slc_nmb+1)
		Message2 = 'For datacube : ' + Cube2bFit
		plt_tlt = 'Slice: ' + str(slc_nmb+1) + '-' + str(round(VEL_AXS[slc_nmb],0)) + ' km/s'
		clp_hdr = 'C'  
		clp_hdc = 'CSL' 
		data_2b_plt = np.asarray(apgtdt(Cube2bFit,memmap=False))
		data_2b_plt_clp = data_2b_plt[slc_nmb]
		data_2b_plot    = data_2b_plt_clp

	elif slc_nmb == None:
		Message1 = 'Fitting gaussian through cube collapse ('+str(clp_fnc)+')'
		Message2 = 'For datacube : ' + Cube2bFit
		plt_tlt = '2D Collapse ('+str(clp_fnc).upper()+') VW:' + str(int(vl_wdt_clp)) + 'km/s ('+sgm_wgth_tms+') ['+ str(int(slice_nmbr+1))+ '$\pm$' + str(int((dlt_ch_nmb))) + ']'
		data_2b_plt = np.asarray(apgtdt(Cube2bFit,memmap=False))
		if slice_nblw!=slice_nbhg:
			data_2b_plt_clp = data_2b_plt[slice_nblw:slice_nbhg]
			skp_clp = False
		elif slice_nblw==slice_nbhg:
			data_2b_plt_clp = data_2b_plt[int(slice_nmbr)]
			skp_clp = True
		if clp_fnc == 'sum' and skp_clp == False:
			data_2b_plot = np.asarray(np.nansum(np.array(data_2b_plt_clp)   , axis=0))
			clp_hdr = 'S'  
			clp_hdc = 'SUM'  			
		elif clp_fnc == 'med' and skp_clp == False:
			data_2b_plot = np.asarray(np.nanmedian(np.array(data_2b_plt_clp), axis=0))
			clp_hdr = 'M'  
			clp_hdc = 'MED' 
		elif clp_fnc == 'avg' and skp_clp == False:
			data_2b_plot = np.asarray(np.nanmean(np.array(data_2b_plt_clp)  , axis=0))
			clp_hdr = 'A'
			clp_hdc = 'AVG'
		elif clp_fnc == 'sum' and skp_clp == True:
			data_2b_plot = data_2b_plt_clp
			clp_hdr = 'S'  
			clp_hdc = 'SUM'
		elif clp_fnc == 'med' and skp_clp == True:
			data_2b_plot = data_2b_plt_clp
			clp_hdr = 'M'  
			clp_hdc = 'MED'
		elif clp_fnc == 'avg' and skp_clp == True:
			data_2b_plot = data_2b_plt_clp
			clp_hdr = 'A'
			clp_hdc = 'AVG'
			
	print
	print (colored(Message1,'yellow'))
	print (colored(Message2,'yellow'))
	print
	Wrt_FITS_File(data_2b_plot,Cube2bclp_2D_opt)

	CUBES_2DGF = [Cube2bclp_2D_opt,Cube2bFit]	

	if slc_nmb == None:
		[Header_Add(CUBE, clp_hdr + 'CL_TYP',sgm_wgth_tms,header_comment = 'Criteria used for Collapse ' + clp_hdc) for CUBE in CUBES_2DGF]
		[Header_Add(CUBE, clp_hdr + 'CL_CNN',tlt_ch_nmb  ,header_comment = 'Number Chns used for Collapse ' + clp_hdc) for CUBE in CUBES_2DGF]
		[Header_Add(CUBE, clp_hdr + 'CL_CNC',slice_cct   ,header_comment = 'Central Chn used for Collapse Num ' + clp_hdc) for CUBE in CUBES_2DGF]
		[Header_Add(CUBE, clp_hdr + 'CL_CNI',slice_nblw  ,header_comment = 'Initial Chn used for Collapse Num ' + clp_hdc) for CUBE in CUBES_2DGF]
		[Header_Add(CUBE, clp_hdr + 'CL_CNL',slice_nbhg  ,header_comment = 'Last Chn used for Collapse Num ' + clp_hdc) for CUBE in CUBES_2DGF]
		[Header_Add(CUBE, clp_hdr + 'CL_CVC',slice_vlct  ,header_comment = 'Central Chn used for Collapse Vel ' + clp_hdc) for CUBE in CUBES_2DGF]
		[Header_Add(CUBE, clp_hdr + 'CL_CVI',slice_vllw  ,header_comment = 'Initial Chn used for Collapse Vel ' + clp_hdc) for CUBE in CUBES_2DGF]
		[Header_Add(CUBE, clp_hdr + 'CL_CVL',slice_vlhg  ,header_comment = 'Last Chn used for Collapse Vel ' + clp_hdc) for CUBE in CUBES_2DGF]
		[Header_Add(CUBE, clp_hdr + 'CL_VLW',vl_wdt_clp  ,header_comment = 'Velocity width according to collapse criteria ' + clp_hdc) for CUBE in CUBES_2DGF]
	elif slc_nmb != None:
		pass
	try:
		[Header_Copy(CUBE,Cube2bFit,'XXT_MIN',header_comment = 'Image Extent X MIN')             for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'XXT_MAX',header_comment = 'Image Extent X MAX')             for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'YXT_MIN',header_comment = 'Image Extent Y MIN')             for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'YXT_MAX',header_comment = 'Image Extent Y MAX')             for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'XCT_FIT',header_comment = 'Image Center X ')                for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'YCT_FIT',header_comment = 'Image Center Y ')                for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'RAD_EXT',header_comment = 'Image Extent Radii')             for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STK_NUM',header_comment = 'Number of galaxies used for Stack') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_AVG',header_comment = 'Redshift Average ')              for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_MED',header_comment = 'Redshift Median ')               for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_1SL',header_comment = 'Redshift 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_1SH',header_comment = 'Redshift 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_2SL',header_comment = 'Redshift 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_2SH',header_comment = 'Redshift 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_3SL',header_comment = 'Redshift 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_3SH',header_comment = 'Redshift 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_P25',header_comment = 'Redshift 25 pct')                for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STZ_P75',header_comment = 'Redshift 75 pct')                for CUBE in CUBES_2DGF]
	except KeyError:
		pass

	try:
		[Header_Copy(CUBE,Cube2bFit,'AMS_AVG',header_comment = 'Synthetic Cubes Source Amplitude AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMN_AVG',header_comment = 'Synthetic Cubes Noise Amplitude  AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SNR_AVG',header_comment = 'Synthetic Cubes SNR Amplitude    AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGN_AVG',header_comment = 'Synthetic Cubes Sigma chan num   AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGV_AVG',header_comment = 'Synthetic Cubes Sigma Vel kms-1  AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FWH_AVG',header_comment = 'Synthetic Cubes FWHM Vel kms-1   AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'OFS_AVG',header_comment = 'Synthetic Cubes Offset           AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMS_MED',header_comment = 'Synthetic Cubes Source Amplitude MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMN_MED',header_comment = 'Synthetic Cubes Noise Amplitude  MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SNR_MED',header_comment = 'Synthetic Cubes SNR Amplitude    MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGN_MED',header_comment = 'Synthetic Cubes Sigma chan num   MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGV_MED',header_comment = 'Synthetic Cubes Sigma Vel kms-1  MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FWH_MED',header_comment = 'Synthetic Cubes FWHM Vel kms-1   MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'OFS_MED',header_comment = 'Synthetic Cubes Offset           MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMS_STD',header_comment = 'Synthetic Cubes Source Amplitude STD') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMN_STD',header_comment = 'Synthetic Cubes Noise Amplitude  STD') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SNR_STD',header_comment = 'Synthetic Cubes SNR Amplitude    STD') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGN_STD',header_comment = 'Synthetic Cubes Sigma chan num   STD') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGV_STD',header_comment = 'Synthetic Cubes Sigma Vel kms-1  STD') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FWH_STD',header_comment = 'Synthetic Cubes FWHM Vel kms-1   STD') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'OFS_STD',header_comment = 'Synthetic Cubes Offset           STD') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMS_MIN',header_comment = 'Synthetic Cubes Source Amplitude MIN') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMN_MIN',header_comment = 'Synthetic Cubes Noise Amplitude  MIN') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SNR_MIN',header_comment = 'Synthetic Cubes SNR Amplitude    MIN') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGN_MIN',header_comment = 'Synthetic Cubes Sigma chan num   MIN') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGV_MIN',header_comment = 'Synthetic Cubes Sigma Vel kms-1  MIN') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FWH_MIN',header_comment = 'Synthetic Cubes FWHM Vel kms-1   MIN') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'OFS_MIN',header_comment = 'Synthetic Cubes Offset           MIN') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMS_MAX',header_comment = 'Synthetic Cubes Source Amplitude MAX') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'AMN_MAX',header_comment = 'Synthetic Cubes Noise Amplitude  MAX') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SNR_MAX',header_comment = 'Synthetic Cubes SNR Amplitude    MAX') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGN_MAX',header_comment = 'Synthetic Cubes Sigma chan num   MAX') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'SGV_MAX',header_comment = 'Synthetic Cubes Sigma Vel kms-1  MAX') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FWH_MAX',header_comment = 'Synthetic Cubes FWHM Vel kms-1   MAX') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'OFS_MAX',header_comment = 'Synthetic Cubes Offset           MAX') for CUBE in CUBES_2DGF]
	except KeyError:
		pass

	try:
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_AVG',header_comment = str(Splt_Hdr_Cmt_cp) + ' Average')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_MED',header_comment = str(Splt_Hdr_Cmt_cp) + ' Median')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_1SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_1SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_2SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_2SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_3SL',header_comment = str(Splt_Hdr_Cmt_cp) + ' 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_3SH',header_comment = str(Splt_Hdr_Cmt_cp) + ' 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_P25',header_comment = str(Splt_Hdr_Cmt_cp) + ' 25 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STS_P75',header_comment = str(Splt_Hdr_Cmt_cp) + ' 75 pct')

		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_AVG',header_comment = 'FIR Lum Average')                   
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_MED',header_comment = 'FIR Lum Median')                    
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_1SL',header_comment = 'FIR Lum 1 sgm lw lmt 15.9 pct')  
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_1SH',header_comment = 'FIR Lum 1 sgm hg lmt 84.1 pct') 
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_2SL',header_comment = 'FIR Lum 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_2SH',header_comment = 'FIR Lum 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_3SL',header_comment = 'FIR Lum 3 sgm lw lmt 0.20 pct')   
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_3SH',header_comment = 'FIR Lum 3 sgm hg lmt 99.8 pct') 
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_P25',header_comment = 'FIR Lum 25 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STF_P75',header_comment = 'FIR Lum 75 pct')
	except KeyError:
		pass

	try:
		[Header_Copy(CUBE,Cube2bFit,'STT_VEL',header_comment = 'CbeWth [km/s]') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_FLS',header_comment = 'TFlx SUM All Chns') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_TFL',header_comment = 'TFlx SUM All Chns * CbeWth') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LMS',header_comment = 'TFlx SUM All CH 2 Lum') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LLM',header_comment = 'TFlx SUM All CH 2 Lum [log]') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LMT',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LLT',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log]') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_FLE',header_comment = 'TFlx SUM All Chns Err') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_TFE',header_comment = 'TFlx SUM All Chns * CbeWth Err') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_SL1',header_comment = 'TFlx SUM All CH 2 Lum Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_SH1',header_comment = 'TFlx SUM All CH 2 Lum Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LL1',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LH1',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_ML1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_MH1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_TL1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_TH1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_SL2',header_comment = 'TFlx SUM All CH 2 Lum Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_SH2',header_comment = 'TFlx SUM All CH 2 Lum Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LL2',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LH2',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_ML2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_MH2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_TL2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_TH2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_SL3',header_comment = 'TFlx SUM All CH 2 Lum Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_SH3',header_comment = 'TFlx SUM All CH 2 Lum Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LL3',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_LH3',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_ML3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_MH3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_TL3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'STT_TH3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_SNS',header_comment = 'Vel Prf Max Chn Location SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_SVS',header_comment = 'Vel Prf Max Frequency Location [Hz] SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_VLS',header_comment = 'Vel Prf Max Chn Value SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_SNA',header_comment = 'Vel Prf Max Chn Location AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_SVA',header_comment = 'Vel Prf Max Frequency Location [Hz] AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_VLA',header_comment = 'Vel Prf Max Chn Value AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_SNM',header_comment = 'Vel Prf Max Chn Location MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_SVM',header_comment = 'Vel Prf Max Frequency Location [Hz] MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'MAX_VLM',header_comment = 'Vel Prf Max Chn Value MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_AMP',header_comment = '1DGF Amplitude AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_CTR',header_comment = '1DGF Center AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_SIG',header_comment = '1DGF Sigma AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_FWH',header_comment = '1DGF FWHM  AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_A2A',header_comment = '1DGF Area A AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_A2M',header_comment = '1DGF Area M AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_CH2',header_comment = '1DGF Chi2 AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_CHR',header_comment = '1DGF Chi2 Reduced AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_LUM',header_comment = '1DGF Ar2Lum AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_LLM',header_comment = '1DGF Ar2Lum [log] AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_APE',header_comment = '1DGF Amplitude Err AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_CTE',header_comment = '1DGF Center Err AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_SGE',header_comment = '1DGF Sigma Err AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_FWE',header_comment = '1DGF FWHM  Err AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_AAE',header_comment = '1DGF Area A Err AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_AME',header_comment = '1DGF Area M Err AVG') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_ML1',header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_MH1',header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_LL1',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_LH1',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_ML2',header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_MH2',header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_LL2',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_LH2',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_ML3',header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_MH3',header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_LL3',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_LH3',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_ML1',header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_MH1',header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_LL1',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_LH1',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_ML2',header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_MH2',header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_LL2',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_LH2',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_ML3',header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_MH3',header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_LL3',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMA_LH3',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_CCT',header_comment = 'Channel # Central AVG (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_1SG',header_comment = 'Channel Dif Limit 1SG AVG (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_2SG',header_comment = 'Channel Dif Limit 2SG AVG (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_3SG',header_comment = 'Channel Dif Limit 3SG AVG (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_4SG',header_comment = 'Channel Dif Limit 4SG AVG (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_5SG',header_comment = 'Channel Dif Limit 5SG AVG (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_1FW',header_comment = 'Channel Dif Limit 1FW AVG (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTA_2FW',header_comment = 'Channel Dif Limit 2FW AVG (PyId)') for CUBE in CUBES_2DGF]
	except KeyError:
		pass
	try:
		[Header_Copy(CUBE,Cube2bFit,'FTM_AMP',header_comment = '1DGF Amplitude MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_CTR',header_comment = '1DGF Center MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_SIG',header_comment = '1DGF Sigma MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_FWH',header_comment = '1DGF FWHM  MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_A2A',header_comment = '1DGF Area A MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_A2M',header_comment = '1DGF Area M MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_CH2',header_comment = '1DGF Chi2 MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_CHR',header_comment = '1DGF Chi2 Reduced MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_LUM',header_comment = '1DGF Ar2Lum MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_LLM',header_comment = '1DGF Ar2Lum [log] MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_APE',header_comment = '1DGF Amplitude Err MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_CTE',header_comment = '1DGF Center Err MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_SGE',header_comment = '1DGF Sigma Err MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_FWE',header_comment = '1DGF FWHM  Err MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_AAE',header_comment = '1DGF Area A Err MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_AME',header_comment = '1DGF Area M Err MED') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_ML1',header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_MH1',header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_LL1',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_LH1',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_ML2',header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_MH2',header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_LL2',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_LH2',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_ML3',header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_MH3',header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_LL3',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_LH3',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_ML1',header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_MH1',header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_LL1',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_LH1',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_ML2',header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_MH2',header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_LL2',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_LH2',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_ML3',header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_MH3',header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_LL3',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMM_LH3',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_CCT',header_comment = 'Channel # Central MED (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_1SG',header_comment = 'Channel Dif Limit 1SG MED (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_2SG',header_comment = 'Channel Dif Limit 2SG MED (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_3SG',header_comment = 'Channel Dif Limit 3SG MED (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_4SG',header_comment = 'Channel Dif Limit 4SG MED (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_5SG',header_comment = 'Channel Dif Limit 5SG MED (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_1FW',header_comment = 'Channel Dif Limit 1FW MED (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTM_2FW',header_comment = 'Channel Dif Limit 2FW MED (PyId)') for CUBE in CUBES_2DGF]
	except KeyError:
		pass

	try:
		[Header_Copy(CUBE,Cube2bFit,'FTS_AMP',header_comment = '1DGF Amplitude SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_CTR',header_comment = '1DGF Center SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_SIG',header_comment = '1DGF Sigma SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_FWH',header_comment = '1DGF FWHM  SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_A2A',header_comment = '1DGF Area A SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_A2M',header_comment = '1DGF Area M SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_CH2',header_comment = '1DGF Chi2 SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_CHR',header_comment = '1DGF Chi2 Reduced M SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_LUM',header_comment = '1DGF Ar2Lum SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_LLM',header_comment = '1DGF Ar2Lum [log] SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_APE',header_comment = '1DGF Amplitude Err SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_CTE',header_comment = '1DGF Center Err SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_SGE',header_comment = '1DGF Sigma Err SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_FWE',header_comment = '1DGF FWHM Err SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_AAE',header_comment = '1DGF Area A Err SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_AME',header_comment = '1DGF Area M Err SUM') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_ML1',header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_MH1',header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_LL1',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_LH1',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_ML2',header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_MH2',header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_LL2',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_LH2',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_ML3',header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_MH3',header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_LL3',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_LH3',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_ML1',header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_MH1',header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_LL1',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_LH1',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_ML2',header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_MH2',header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_LL2',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_LH2',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_ML3',header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_MH3',header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_LL3',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FMS_LH3',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_CCT',header_comment = 'Channel # Central SUM (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_1SG',header_comment = 'Channel Dif Limit 1SG SUM (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_2SG',header_comment = 'Channel Dif Limit 2SG SUM (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_3SG',header_comment = 'Channel Dif Limit 3SG SUM (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_4SG',header_comment = 'Channel Dif Limit 4SG SUM (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_5SG',header_comment = 'Channel Dif Limit 5SG SUM (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_1FW',header_comment = 'Channel Dif Limit 1FW SUM (PyId)') for CUBE in CUBES_2DGF]
		[Header_Copy(CUBE,Cube2bFit,'FTS_2FW',header_comment = 'Channel Dif Limit 2FW SUM (PyId)') for CUBE in CUBES_2DGF]
	except KeyError:
		pass

	print
	print (colored('Input Cube for collapse : ' + Cube2bFit,'magenta'))
	print (colored('Resulting Collapsed Cube: ' + Cube2bclp_2D_opt,'yellow'))

	nx_f2DG, ny_f2DG = data_2b_plot.shape
	nx,ny            = nx_f2DG,ny_f2DG

	X0_f2DG     = kwargs.get('X0_f2DG',int(np.ceil(nx_f2DG/2)))
	Y0_f2DG     = kwargs.get('Y0_f2DG',int(np.ceil(ny_f2DG/2)))
	A_f2DG      = kwargs.get('A_f2DG',1)
	SIGMAX_f2DG = kwargs.get('SIGMAX_f2DG',1)
	SIGMAY_f2DG = kwargs.get('SIGMAY_f2DG',1)
	THETA_f2DG  = kwargs.get('THETA_f2DG',0)
	OFS_f2DG    = kwargs.get('OFS_f2DG',0)
	displ_s_f   = kwargs.get('displ_s_f',False)
	verbose     = kwargs.get('verbose',False)

	# Create x and y indices
	x    = np.linspace(0, nx_f2DG, nx_f2DG)-0.5
	y    = np.linspace(0, ny_f2DG, ny_f2DG)-0.5
	x, y = np.meshgrid(x, y)

	A_f2DG = np.amax(data_2b_plot)

	data = data_2b_plot

	initial_guess = (X0_f2DG,Y0_f2DG,A_f2DG,SIGMAX_f2DG,SIGMAY_f2DG,THETA_f2DG,OFS_f2DG)

	xdata       = np.vstack((x.ravel(),y.ravel()))
	ydata       = data.ravel()

	brt_flx_sum = np.sum(ydata)
	brt_flx_std = np.std(ydata)
	brt_flx_med = np.median(ydata)
	brt_flx_avg = np.mean(ydata)
	
	try:
		if fit_type == 'scipy':
			print
			print (colored('2D Gaussian Fit Mode Choosen: Scipy','yellow'))
			print
			if src_sze_fxd == True:
				print
				print (colored('Source size fixed','yellow'))
				print (colored(str(SIGMAX_f2DG),'yellow'))
				print				
				popt, pcov  = scpopt.curve_fit(func_2D_Gaussian, xdata, ydata, 
							p0=initial_guess,absolute_sigma=True,
							bounds=([X0_f2DG-1,Y0_f2DG-1,-np.inf,SIGMAX_f2DG-0.01,SIGMAX_f2DG-0.01,-np.inf,-np.inf],
									[X0_f2DG+1,Y0_f2DG+1, np.inf,SIGMAY_f2DG+0.01,SIGMAY_f2DG+0.01, np.inf, np.inf]),
							)
			elif src_sze_fxd == False:
				print
				print (colored('Source size NOT fixed','yellow'))
				print				
				popt, pcov  = scpopt.curve_fit(func_2D_Gaussian, xdata, ydata, 
							p0=initial_guess,absolute_sigma=True,
							bounds=([X0_f2DG-1,Y0_f2DG-1,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
									[X0_f2DG+1,Y0_f2DG+1, np.inf, np.inf, np.inf, np.inf, np.inf]))
			perr        = np.sqrt(np.diag(pcov))
			data_fitted = func_2D_Gaussian((x, y), *popt, circular=circular)
			fit_res     = 'OK'
			X0_F        = np.round(popt[0],0)
			Y0_F        = np.round(popt[1],0)
			X_DIF       = np.round(X0_F,0) - X0_f2DG
			Y_DIF       = np.round(Y0_F,0) - Y0_f2DG

			X0_F         = np.round(popt[0],0)
			Y0_F         = np.round(popt[1],0)
			A_F          = np.round(popt[2],9)
			SIGMAX_F     = np.round(popt[3],9)
			SIGMAY_F     = np.round(popt[4],9)
			THETA_F      = np.round(popt[5],9)
			OFFSET_F     = np.round(popt[6],9)

			X0_E         = np.round(perr[0],9)
			Y0_E         = np.round(perr[1],9)
			A_E          = np.round(perr[2],9)
			SIGMAX_E     = np.round(perr[3],9)
			SIGMAY_E     = np.round(perr[4],9)
			THETA_E      = np.round(perr[5],9)
			OFFSET_E     = np.round(perr[6],9)
		elif fit_type == 'lmfit':
			import lmfit
			print
			print (colored('2D Gaussian Fit Mode Choosen: Lmfit','yellow'))
			print
			mod     = lmfit.models.ExpressionModel(
						'func_2D_Gaussian_lmfit(X_0, Y_0, Amplitude, Sigma_x, Sigma_y, Theta, Offset',
						independent_vars=['X,Y'])
			params  = mod.make_params(X_0= X0_f2DG, Y_0= Y0_f2DG, Amplitude= A_f2DG, Sigma_x= SIGMAX_f2DG, Sigma_y= SIGMAY_f2DG, Theta= THETA_f2DG, Offset=OFS_f2DG,)
			FLX_AVG = FLX_AVG.astype(float)
			XAXIS   = XAXIS.astype(float)
			result  = mod.fit(ydata, params, x=xdata)

			X0_F         = result.params['X_0'].value
			Y0_F         = result.params['Y_0'].value
			A_F          = result.params['Amplitude'].value
			SIGMAX_F     = result.params['Sigma_x'].value
			SIGMAY_F     = result.params['Sigma_y'].value
			THETA_F      = result.params['Theta'].value
			OFFSET_F     = result.params['Offset'].value

			X0_E         = result.params['X_0'].stderr
			Y0_E         = result.params['Y_0'].stderr
			A_E          = result.params['Amplitude'].stderr
			SIGMAX_E     = result.params['Sigma_x'].stderr
			SIGMAY_E     = result.params['Sigma_y'].stderr
			THETA_E      = result.params['Theta'].stderr
			OFFSET_E     = result.params['Offset'].stderr

			X0_C         = result.params['X_0'].correl
			Y0_C         = result.params['Y_0'].correl
			A_C          = result.params['Amplitude'].correl
			SIGMAX_C     = result.params['Sigma_x'].correl
			SIGMAY_C     = result.params['Sigma_y'].correl
			THETA_C      = result.params['Theta'].correl
			OFFSET_C     = result.params['Offset'].correl

	except RuntimeError:
		popt, pcov  = [0,0,0,0,0,0,0],[0,0,0,0,0,0,0]
		perr        = [0,0,0,0,0,0,0]
		X0_F        = 0
		Y0_F        = 0
		X_DIF       = 0
		Y_DIF       = 0

		X0_F        = 0
		Y0_F        = 0
		X_DIF       = 0
		Y_DIF       = 0
					
		X0_F        = 0
		Y0_F        = 0
		A_F         = 0
		SIGMAX_F    = 0
		SIGMAY_F    = 0
		THETA_F     = 0
		OFFSET_F    = 0
					
		X0_E        = 0
		Y0_E        = 0
		A_E         = 0
		SIGMAX_E    = 0
		SIGMAY_E    = 0
		THETA_E     = 0
		OFFSET_E    = 0

		data_fitted = func_2D_Gaussian((x, y), *popt, circular=circular)
		fit_res     = 'ERR'
		print("Error - curve_fit failed")
	if circular ==True:
		SIGMAY_F = SIGMAX_F
		SIGMAY_E = SIGMAX_E
	elif circular == False:
		pass
	########################CHI-SQUARE########################
	if sgm_fnc == 'avg':
		header_amp_guess = 'FTA_AMP'
		sigma_fwhm_2d    = SIGMAX_F
	elif sgm_fnc == 'med':
		header_amp_guess = 'FTM_AMP'
		sigma_fwhm_2d    = SIGMAX_F
	elif sgm_fnc == 'sum':
		header_amp_guess = 'FTS_AMP'
		sigma_fwhm_2d    = SIGMAX_F
	g_obs = data_2b_plot
	g_exp = func_2D_Gaussian((x, y),X0_f2DG,Y0_f2DG,Header_Get(Cube2bFit,header_amp_guess),sigma_fwhm_2d,sigma_fwhm_2d,THETA_F,OFFSET_F,circular=circular)

	if Cube2bFit_Err == None:

		chi2      = sum((g_obs-g_exp)**2/g_exp)
		chi2_red  = chi2/float(len(ydata)-1)

		g_obs_sgm = g_obs
		g_exp_sgm = func_2D_Gaussian((x, y),X0_f2DG,Y0_f2DG,Header_Get(Cube2bFit,header_amp_guess),SIGMAX_F,SIGMAY_F,THETA_F,OFFSET_F,circular=circular)
		chi2_sgm      = sum((g_obs_sgm-g_exp_sgm)**2/g_exp_sgm)
		chi2_red_sgm  = chi2_sgm/float(len(ydata)-1)

		g_obs_amp = g_obs
		g_exp_amp = func_2D_Gaussian((x, y),X0_f2DG,Y0_f2DG,A_F,sigma_fwhm_2d,sigma_fwhm_2d,THETA_F,OFFSET_F,circular=circular)
		chi2_amp      = sum((g_obs_amp-g_exp_amp)**2/g_exp_amp)
		chi2_red_amp  = chi2_amp/float(len(ydata)-1)

	elif Cube2bFit_Err != None:
		print
		print (colored('Using Collapsed Datacube for Errors: ','yellow'))
		print (colored(Cube2bFit_Err,'cyan'))
		print
		cube_data_err    = np.asarray(apgtdt(Cube2bFit_Err,memmap=False) )
		g_err = cube_data_err
		g_obs = np.asarray(g_obs)
		g_exp = (np.asarray(g_exp)).reshape(nx, ny)

		chi2          = np.nansum(((g_obs-g_exp)**2)/g_err,axis=None)
		chi2_red      = chi2/float(len(ydata)-1)

		g_obs_sgm     = g_obs
		g_exp_sgm     = func_2D_Gaussian((x, y),X0_f2DG,Y0_f2DG,Header_Get(Cube2bFit,header_amp_guess),SIGMAX_F,SIGMAY_F,THETA_F,OFFSET_F,circular=circular)
		g_exp_sgm     = (np.asarray(g_exp_sgm)).reshape(nx, ny)
		chi2_sgm      = np.sum(((g_obs_sgm-g_exp_sgm)**2)/g_err,axis=None)
		chi2_red_sgm  = chi2_sgm/float(len(ydata)-1)

		g_obs_amp     = g_obs
		g_exp_amp     = func_2D_Gaussian((x, y),X0_f2DG,Y0_f2DG,A_F,sigma_fwhm_2d,sigma_fwhm_2d,THETA_F,OFFSET_F,circular=circular)
		g_exp_amp     = (np.asarray(g_exp_amp)).reshape(nx, ny)
		chi2_amp      = np.sum(((g_obs_amp-g_exp_amp)**2)/g_err,axis=None)
		chi2_red_amp  = chi2_amp/float(len(ydata)-1)
	########################CHI-SQUARE########################

	if (abs((SIGMAX_F/SIGMAX_f2DG)) * abs((SIGMAX_F/SIGMAX_f2DG))) < 1:
		SIGMAX_F = SIGMAX_f2DG
		SIGMAY_F = SIGMAY_f2DG
		print (colored('Source size smaller than beam size!','cyan'))
		print (colored(str(abs((SIGMAX_F/SIGMAX_f2DG)) * abs((SIGMAX_F/SIGMAX_f2DG))),'cyan'))
		print (colored('Assuming source size equal to beam size: '+str(SIGMAX_f2DG),'cyan'))
		print (abs((SIGMAX_F/SIGMAX_f2DG)) * abs((SIGMAX_F/SIGMAX_f2DG)))
		print 
		src_sze_sml_bem_sze = True
	else:
		src_sze_sml_bem_sze = False
		pass	
	[Header_Add(CUBE, clp_hdr + '2G_XCT',X0_F        ,header_comment = '2DGF X ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_YCT',Y0_F        ,header_comment = '2DGF Y ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_AMP',A_F         ,header_comment = '2DGF Amplitude ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_SGX',SIGMAX_F    ,header_comment = '2DGF Sigma X ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_SGY',SIGMAY_F    ,header_comment = '2DGF Sigma Y ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_THT',THETA_F     ,header_comment = '2DGF Theta ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_OFS',OFFSET_F    ,header_comment = '2DGF Offset ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_XCE',X0_E        ,header_comment = '2DGF X Err ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_YCE',Y0_E        ,header_comment = '2DGF Y Err ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_AME',A_E         ,header_comment = '2DGF Amplitude Err ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_SXE',SIGMAX_E    ,header_comment = '2DGF Sigma X Err ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_SYE',SIGMAY_E    ,header_comment = '2DGF Sigma Y Err ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_THE',THETA_E     ,header_comment = '2DGF Theta Err ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_OFE',OFFSET_E    ,header_comment = '2DGF Offset Err ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_CH2',chi2        ,header_comment = '2DGF Chi2 ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_CHR',chi2_red    ,header_comment = '2DGF Chi2 Reduced ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_C2S',chi2_sgm    ,header_comment = '2DGF Chi2 sgm '  + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_CRS',chi2_red_sgm,header_comment = '2DGF Chi2 Reduced sgm ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_C2A',chi2_amp    ,header_comment = '2DGF Chi2 Amp ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE, clp_hdr + '2G_CRA',chi2_red_amp,header_comment = '2DGF Chi2 Reduced Amp ' + clp_hdc) for CUBE in CUBES_2DGF]

	SIGMAF_E = slice_sigm_err

	#################IN THIS WAY FLUX COMPUTATIONS ARE IN Jy*km/s and NOT in Jy*km/s*1/beam#################
	###################BY NORMALIZING THE FIT SIZE BY THE BEAM SIZE IN SIGMA-PIXEL-UNITS####################
	##################################NOTICE THAT THE 2PI FACTOR CANCELS ###################################
	print
	print (colored('IN THIS WAY FLUX COMPUTATIONS ARE IN Jy*km/s and NOT in Jy*km/s*1/beam','yellow'))
	print (colored('BY NORMALIZING THE FIT SIZE BY THE BEAM SIZE IN SIGMA-PIXEL-UNITS','yellow'))
	print (colored('NOTICE THAT THE 2PI FACTOR CANCELS OUT','yellow'))
	print
	DGF_vlm = np.sqrt(2 * np.pi) * abs((SIGMAX_F/SIGMAX_f2DG)) * abs((SIGMAX_F/SIGMAX_f2DG)) * A_F 
	DGF_vx1 = np.sqrt(2 * np.pi) * abs((SIGMAX_F/SIGMAX_f2DG)) * abs((SIGMAX_F/SIGMAX_f2DG)) * A_F * slice_sigm
	DGF_vx2 = np.sqrt(2 * np.pi) * abs((SIGMAX_F/SIGMAX_f2DG)) * abs((SIGMAX_F/SIGMAX_f2DG)) * A_F * vl_wdt_clp
	DGF_vx3 = np.sqrt(2 * np.pi) * abs((SIGMAX_F/SIGMAX_f2DG)) * abs((SIGMAX_F/SIGMAX_f2DG)) * A_F 
	if src_sze_fxd == True:
		###########################FLUX ERROR ONLY HAS CONTRIBUTION FROM THE AMP-ERROR##########################
		###################################DUE TO SOURCE SIZE-FIXED=BEAM SIZE###################################
		###########################THIS ASSUMPTION IS VALID IF POINT SOURCES ARE FIT!###########################
		print (colored('THE FLUX ERROR ONLY HAS CONTRIBUTION FROM THE AMP-ERROR','yellow'))
		print (colored('THIS ASSUMPTION IS VALID IF POINT SOURCES ARE FIT!','yellow'))
		print
		DGF_vle = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2))))
		DGF_v1e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2))))
		DGF_v2e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2))))
		DGF_v3e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2))))
	elif src_sze_fxd == False and src_sze_sml_bem_sze == False:
		#########################FLUX ERROR HAS CONTRIBUTION FROM THE AMP-ERROR-SRCE-SZE########################
		###########################THIS ASSUMPTION IS VALID IF POINT SOURCES ARE FIT!###########################
		print (colored('THE FLUX ERROR INCLUDES CONTRIBUTION FROM THE AMP-ERROR AND SIGMA-ERROR','yellow'))
		print (colored('THIS ASSUMPTION IS VALID IF POINT SOURCES ARE FIT!','yellow'))
		print
		DGF_vle = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2)) + (4 * (A_F**2) * (SIGMAX_E**2))))
		DGF_v1e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2)) + (4 * (A_F**2) * (SIGMAX_E**2)))) * slice_sigm
		DGF_v2e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2)) + (4 * (A_F**2) * (SIGMAX_E**2)))) * vl_wdt_clp
		DGF_v3e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2)) + (4 * (A_F**2) * (SIGMAX_E**2)))) 
	elif src_sze_fxd == False and src_sze_sml_bem_sze == True:
		###########################FLUX ERROR ONLY HAS CONTRIBUTION FROM THE AMP-ERROR##########################
		################################DUE TO SOURCE SIZE SMALLER THAN BEAM SIZE###############################
		###########################THIS ASSUMPTION IS VALID IF POINT SOURCES ARE FIT!###########################
		print (colored('THE FLUX ERROR ONLY HAS CONTRIBUTION FROM THE AMP-ERROR','yellow'))
		print (colored('THIS ASSUMPTION IS VALID IF POINT SOURCES ARE FIT!','yellow'))
		print
		DGF_vle = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2))))
		DGF_v1e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2))))
		DGF_v2e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2))))
		DGF_v3e = (((SIGMAX_F/SIGMAX_f2DG)**1)) * (np.sqrt((((SIGMAX_F/SIGMAX_f2DG)**2) * (A_E**2))))

	#################IN THIS WAY FLUX COMPUTATIONS ARE IN Jy*km/s and NOT in Jy*km/s*1/beam#################
	###################BY NORMALIZING THE FIT SIZE BY THE BEAM SIZE IN SIGMA-PIXEL-UNITS####################
	##################################NOTICE THAT THE 2PI FACTOR CANCELS ###################################

	DGF_vol = FluxToLum(DGF_vlm,z_f2l,frq_r)
	DGF_v1l = FluxToLum(DGF_vx1,z_f2l,frq_r)
	DGF_v2l = FluxToLum(DGF_vx2,z_f2l,frq_r)
	DGF_v3l = FluxToLum(DGF_vx3,z_f2l,frq_r)

	BSF_vlm = brt_flx_sum 
	BSF_vx1 = brt_flx_sum  * slice_sigm
	BSF_vx2 = brt_flx_sum  * 2 * vl_wdt_clp
	BSF_vx3 = brt_flx_sum  * slice_cwdt * tlt_ch_nmb

	BSF_vle = brt_flx_std
	BSF_v1e = brt_flx_std * slice_sigm
	BSF_v2e = brt_flx_std * 2 * vl_wdt_clp
	BSF_v3e = brt_flx_std * slice_cwdt * tlt_ch_nmb

	BSF_vol = FluxToLum(BSF_vlm,z_f2l,frq_r)
	BSF_v1l = FluxToLum(BSF_vx1,z_f2l,frq_r)
	BSF_v2l = FluxToLum(BSF_vx2,z_f2l,frq_r)
	BSF_v3l = FluxToLum(BSF_vx3,z_f2l,frq_r)

	redshift_inf_1 = Header_Get(Cube2bFit,'STZ_1SL')
	redshift_sup_1 = Header_Get(Cube2bFit,'STZ_1SH')
	redshift_inf_2 = Header_Get(Cube2bFit,'STZ_2SL')
	redshift_sup_2 = Header_Get(Cube2bFit,'STZ_2SH')
	redshift_inf_3 = Header_Get(Cube2bFit,'STZ_3SL')
	redshift_sup_3 = Header_Get(Cube2bFit,'STZ_3SH')
	
	[Header_Add(CUBE,clp_hdr + '2G_FLS',DGF_vlm   ,header_comment = '2DGF Vol ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_FT1',DGF_vx1   ,header_comment = '2DGF Vol X 1DSGM ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_FT2',DGF_vx2   ,header_comment = '2DGF Vol X 1DNXSGM ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_FT3',DGF_vx3   ,header_comment = '2DGF Vol X CbeWthXNCHN ' + clp_hdc) for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + '2G_FSE',DGF_vle   ,header_comment = '2DGF Vol ' + clp_hdc + ' Err') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_F1E',DGF_v1e   ,header_comment = '2DGF Vol X 1DSGM ' + clp_hdc + 'Err') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_F2E',DGF_v2e   ,header_comment = '2DGF Vol X 1DNXSGM ' + clp_hdc + 'Err') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_F3E',DGF_v3e   ,header_comment = '2DGF Vol X CbeWthXNCHN ' + clp_hdc + 'Err') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + '2G_LMS',DGF_vol[0],header_comment = '2DGF Vol Lum ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_LLS',DGF_vol[1],header_comment = '2DGF Vol Lum [log] ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_LM1',DGF_v1l[0],header_comment = '2DGF Vol X 1DSGM Lum ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_LL1',DGF_v1l[1],header_comment = '2DGF Vol X 1DSGM Lum [log] ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_LM2',DGF_v2l[0],header_comment = '2DGF Vol X 1DNXSGM Lum ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_LL2',DGF_v2l[1],header_comment = '2DGF Vol X 1DNXSGM Lum [log] ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_LM3',DGF_v3l[0],header_comment = '2DGF Vol X CbeWthXNCHN Lum ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2G_LL3',DGF_v3l[1],header_comment = '2DGF Vol X CbeWthXNCHN Lum [log] ' + clp_hdc) for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + 'BS_FLS',BSF_vlm   ,header_comment = 'BSF Vol ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_FT1',BSF_vx1   ,header_comment = 'BSF Vol X 1DSGM ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_FT2',BSF_vx2   ,header_comment = 'BSF Vol X 1DNXSGM ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_FT3',BSF_vx3   ,header_comment = 'BSF Vol X CbeWthXNCHN ' + clp_hdc) for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + 'BS_FSE',BSF_vle   ,header_comment = 'BSF Vol ' + clp_hdc + ' Err') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_F1E',BSF_v1e   ,header_comment = 'BSF Vol X 1DSGM ' + clp_hdc + 'Err') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_F2E',BSF_v2e   ,header_comment = 'BSF Vol X 1DNXSGM ' + clp_hdc + 'Err') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_F3E',BSF_v3e   ,header_comment = 'BSF Vol X CbeWthXNCHN ' + clp_hdc + 'Err') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + 'BS_LMS',BSF_vol[0],header_comment = 'BSF Vol Lum ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_LLS',BSF_vol[1],header_comment = 'BSF Vol Lum [log] ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_LM1',BSF_v1l[0],header_comment = 'BSF Vol X 1DSGM Lum ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_LL1',BSF_v1l[1],header_comment = 'BSF Vol X 1DSGM Lum [log] ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_LM2',BSF_v2l[0],header_comment = 'BSF Vol X 1DNXSGM Lum ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_LL2',BSF_v2l[1],header_comment = 'BSF Vol X 1DNXSGM Lum [log] ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_LM3',BSF_v3l[0],header_comment = 'BSF Vol X CbeWthXNCHN Lum ' + clp_hdc) for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BS_LL3',BSF_v3l[1],header_comment = 'BSF Vol X CbeWthXNCHN Lum [log] ' + clp_hdc) for CUBE in CUBES_2DGF]

	lum_area_err_1_s = Luminosity_Error(DGF_vol[0],redshift_inf_1,redshift_sup_1,DGF_vle,frq_r=frq_r)
	lum_area_err_2_s = Luminosity_Error(DGF_vol[0],redshift_inf_2,redshift_sup_2,DGF_vle,frq_r=frq_r)
	lum_area_err_3_s = Luminosity_Error(DGF_vol[0],redshift_inf_3,redshift_sup_3,DGF_vle,frq_r=frq_r)

	lum_area_err_1_1 = Luminosity_Error(DGF_v1l[0],redshift_inf_1,redshift_sup_1,DGF_v1e,frq_r=frq_r)
	lum_area_err_2_1 = Luminosity_Error(DGF_v1l[0],redshift_inf_2,redshift_sup_2,DGF_v1e,frq_r=frq_r)
	lum_area_err_3_1 = Luminosity_Error(DGF_v1l[0],redshift_inf_3,redshift_sup_3,DGF_v1e,frq_r=frq_r)

	lum_area_err_1_2 = Luminosity_Error(DGF_v2l[0],redshift_inf_1,redshift_sup_1,DGF_v2e,frq_r=frq_r)
	lum_area_err_2_2 = Luminosity_Error(DGF_v2l[0],redshift_inf_2,redshift_sup_2,DGF_v2e,frq_r=frq_r)
	lum_area_err_3_2 = Luminosity_Error(DGF_v2l[0],redshift_inf_3,redshift_sup_3,DGF_v2e,frq_r=frq_r)

	lum_area_err_1_3 = Luminosity_Error(DGF_v3l[0],redshift_inf_1,redshift_sup_1,DGF_v3e,frq_r=frq_r)
	lum_area_err_2_3 = Luminosity_Error(DGF_v3l[0],redshift_inf_2,redshift_sup_2,DGF_v3e,frq_r=frq_r)
	lum_area_err_3_3 = Luminosity_Error(DGF_v3l[0],redshift_inf_3,redshift_sup_3,DGF_v3e,frq_r=frq_r)

	lum_area_brt_err_1_s = Luminosity_Error(BSF_vol[0],redshift_inf_1,redshift_sup_1,BSF_vle,frq_r=frq_r)
	lum_area_brt_err_2_s = Luminosity_Error(BSF_vol[0],redshift_inf_2,redshift_sup_2,BSF_vle,frq_r=frq_r)
	lum_area_brt_err_3_s = Luminosity_Error(BSF_vol[0],redshift_inf_3,redshift_sup_3,BSF_vle,frq_r=frq_r)

	lum_area_brt_err_1_1 = Luminosity_Error(BSF_v1l[0],redshift_inf_1,redshift_sup_1,BSF_v1e,frq_r=frq_r)
	lum_area_brt_err_2_1 = Luminosity_Error(BSF_v1l[0],redshift_inf_2,redshift_sup_2,BSF_v1e,frq_r=frq_r)
	lum_area_brt_err_3_1 = Luminosity_Error(BSF_v1l[0],redshift_inf_3,redshift_sup_3,BSF_v1e,frq_r=frq_r)

	lum_area_brt_err_1_2 = Luminosity_Error(BSF_v2l[0],redshift_inf_1,redshift_sup_1,BSF_v2e,frq_r=frq_r)
	lum_area_brt_err_2_2 = Luminosity_Error(BSF_v2l[0],redshift_inf_2,redshift_sup_2,BSF_v2e,frq_r=frq_r)
	lum_area_brt_err_3_2 = Luminosity_Error(BSF_v2l[0],redshift_inf_3,redshift_sup_3,BSF_v2e,frq_r=frq_r)

	lum_area_brt_err_1_3 = Luminosity_Error(BSF_v3l[0],redshift_inf_1,redshift_sup_1,BSF_v3e,frq_r=frq_r)
	lum_area_brt_err_2_3 = Luminosity_Error(BSF_v3l[0],redshift_inf_2,redshift_sup_2,BSF_v3e,frq_r=frq_r)
	lum_area_brt_err_3_3 = Luminosity_Error(BSF_v3l[0],redshift_inf_3,redshift_sup_3,BSF_v3e,frq_r=frq_r)

	[Header_Add(CUBE,clp_hdr + '2L_S1L',lum_area_err_1_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_S1H',lum_area_err_1_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_S1L',lum_area_err_1_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_S1H',lum_area_err_1_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_S2L',lum_area_err_2_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_S2H',lum_area_err_2_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_S2L',lum_area_err_2_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_S2H',lum_area_err_2_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_S3L',lum_area_err_3_s[0], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_S3H',lum_area_err_3_s[1], header_comment = '2DGF Vol Lum S ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_S3L',lum_area_err_3_s[2], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_S3H',lum_area_err_3_s[3], header_comment = '2DGF Vol Lum S [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + '2L_11L',lum_area_err_1_1[0], header_comment = '2DGF Vol Lum T X 1DSGM ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_11H',lum_area_err_1_1[1], header_comment = '2DGF Vol Lum T X 1DSGM ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_11L',lum_area_err_1_1[2], header_comment = '2DGF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_11H',lum_area_err_1_1[3], header_comment = '2DGF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_12L',lum_area_err_2_1[0], header_comment = '2DGF Vol Lum T X 1DSGM ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_12H',lum_area_err_2_1[1], header_comment = '2DGF Vol Lum T X 1DSGM ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_12L',lum_area_err_2_1[2], header_comment = '2DGF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_12H',lum_area_err_2_1[3], header_comment = '2DGF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_13L',lum_area_err_3_1[0], header_comment = '2DGF Vol Lum T X 1DSGM ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_13H',lum_area_err_3_1[1], header_comment = '2DGF Vol Lum T X 1DSGM ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_13L',lum_area_err_3_1[2], header_comment = '2DGF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_13H',lum_area_err_3_1[3], header_comment = '2DGF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + '2L_21L',lum_area_err_1_2[0], header_comment = '2DGF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_21H',lum_area_err_1_2[1], header_comment = '2DGF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_21L',lum_area_err_1_2[2], header_comment = '2DGF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_21H',lum_area_err_1_2[3], header_comment = '2DGF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_22L',lum_area_err_2_2[0], header_comment = '2DGF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_22H',lum_area_err_2_2[1], header_comment = '2DGF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_22L',lum_area_err_2_2[2], header_comment = '2DGF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_22H',lum_area_err_2_2[3], header_comment = '2DGF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_23L',lum_area_err_3_2[0], header_comment = '2DGF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_23H',lum_area_err_3_2[1], header_comment = '2DGF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_23L',lum_area_err_3_2[2], header_comment = '2DGF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_23H',lum_area_err_3_2[3], header_comment = '2DGF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + '2L_31L',lum_area_err_1_3[0], header_comment = '2DGF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_31H',lum_area_err_1_3[1], header_comment = '2DGF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_31L',lum_area_err_1_3[2], header_comment = '2DGF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_31H',lum_area_err_1_3[3], header_comment = '2DGF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_32L',lum_area_err_2_3[0], header_comment = '2DGF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_32H',lum_area_err_2_3[1], header_comment = '2DGF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_32L',lum_area_err_2_3[2], header_comment = '2DGF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_32H',lum_area_err_2_3[3], header_comment = '2DGF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_33L',lum_area_err_3_3[0], header_comment = '2DGF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2L_33H',lum_area_err_3_3[1], header_comment = '2DGF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_33L',lum_area_err_3_3[2], header_comment = '2DGF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + '2M_33H',lum_area_err_3_3[3], header_comment = '2DGF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + 'BL_S1L',lum_area_brt_err_1_s[0], header_comment = 'BSF Vol Lum S ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_S1H',lum_area_brt_err_1_s[1], header_comment = 'BSF Vol Lum S ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_S1L',lum_area_brt_err_1_s[2], header_comment = 'BSF Vol Lum S [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_S1H',lum_area_brt_err_1_s[3], header_comment = 'BSF Vol Lum S [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_S2L',lum_area_brt_err_2_s[0], header_comment = 'BSF Vol Lum S ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_S2H',lum_area_brt_err_2_s[1], header_comment = 'BSF Vol Lum S ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_S2L',lum_area_brt_err_2_s[2], header_comment = 'BSF Vol Lum S [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_S2H',lum_area_brt_err_2_s[3], header_comment = 'BSF Vol Lum S [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_S3L',lum_area_brt_err_3_s[0], header_comment = 'BSF Vol Lum S ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_S3H',lum_area_brt_err_3_s[1], header_comment = 'BSF Vol Lum S ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_S3L',lum_area_brt_err_3_s[2], header_comment = 'BSF Vol Lum S [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_S3H',lum_area_brt_err_3_s[3], header_comment = 'BSF Vol Lum S [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + 'BL_11L',lum_area_brt_err_1_1[0], header_comment = 'BSF Vol Lum T X 1DSGM ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_11H',lum_area_brt_err_1_1[1], header_comment = 'BSF Vol Lum T X 1DSGM ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_11L',lum_area_brt_err_1_1[2], header_comment = 'BSF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_11H',lum_area_brt_err_1_1[3], header_comment = 'BSF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_12L',lum_area_brt_err_2_1[0], header_comment = 'BSF Vol Lum T X 1DSGM ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_12H',lum_area_brt_err_2_1[1], header_comment = 'BSF Vol Lum T X 1DSGM ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_12L',lum_area_brt_err_2_1[2], header_comment = 'BSF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_12H',lum_area_brt_err_2_1[3], header_comment = 'BSF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_13L',lum_area_brt_err_3_1[0], header_comment = 'BSF Vol Lum T X 1DSGM ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_13H',lum_area_brt_err_3_1[1], header_comment = 'BSF Vol Lum T X 1DSGM ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_13L',lum_area_brt_err_3_1[2], header_comment = 'BSF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_13H',lum_area_brt_err_3_1[3], header_comment = 'BSF Vol Lum T X 1DSGM [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + 'BL_21L',lum_area_brt_err_1_2[0], header_comment = 'BSF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_21H',lum_area_brt_err_1_2[1], header_comment = 'BSF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_21L',lum_area_brt_err_1_2[2], header_comment = 'BSF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_21H',lum_area_brt_err_1_2[3], header_comment = 'BSF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_22L',lum_area_brt_err_2_2[0], header_comment = 'BSF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_22H',lum_area_brt_err_2_2[1], header_comment = 'BSF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_22L',lum_area_brt_err_2_2[2], header_comment = 'BSF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_22H',lum_area_brt_err_2_2[3], header_comment = 'BSF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_23L',lum_area_brt_err_3_2[0], header_comment = 'BSF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_23H',lum_area_brt_err_3_2[1], header_comment = 'BSF Vol Lum T X 1DNXSGM ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_23L',lum_area_brt_err_3_2[2], header_comment = 'BSF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_23H',lum_area_brt_err_3_2[3], header_comment = 'BSF Vol Lum T X 1DNXSGM [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]

	[Header_Add(CUBE,clp_hdr + 'BL_31L',lum_area_brt_err_1_3[0], header_comment = 'BSF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_31H',lum_area_brt_err_1_3[1], header_comment = 'BSF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_31L',lum_area_brt_err_1_3[2], header_comment = 'BSF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 1 sgm lw lmt 15.9 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_31H',lum_area_brt_err_1_3[3], header_comment = 'BSF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 1 sgm hg lmt 84.1 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_32L',lum_area_brt_err_2_3[0], header_comment = 'BSF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_32H',lum_area_brt_err_2_3[1], header_comment = 'BSF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_32L',lum_area_brt_err_2_3[2], header_comment = 'BSF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 2 sgm lw lmt 2.30 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_32H',lum_area_brt_err_2_3[3], header_comment = 'BSF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 2 sgm hg lmt 97.7 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_33L',lum_area_brt_err_3_3[0], header_comment = 'BSF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BL_33H',lum_area_brt_err_3_3[1], header_comment = 'BSF Vol Lum T X CbeWthXNCHN ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_33L',lum_area_brt_err_3_3[2], header_comment = 'BSF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 3 sgm lw lmt 0.20 pct') for CUBE in CUBES_2DGF]
	[Header_Add(CUBE,clp_hdr + 'BM_33H',lum_area_brt_err_3_3[3], header_comment = 'BSF Vol Lum T X CbeWthXNCHN [log] ' + clp_hdc +' Err 3 sgm hg lmt 99.8 pct') for CUBE in CUBES_2DGF]

	####################MODEL AND RESIDUAL####################
	g_obs_img     = g_obs#func_2D_Gaussian((x, y),X0_F   ,Y0_F   ,A_F                            ,sigma_fwhm_2d,sigma_fwhm_2d,THETA_F,OFFSET_F,circular=circular)
	g_mdl_img     = func_2D_Gaussian((x, y),X0_F,Y0_F,A_F,SIGMAX_F,SIGMAY_F,THETA_F,OFFSET_F,circular=circular)
	g_obs_img     = (np.asarray(g_obs_img)).reshape(nx, ny)
	g_mdl_img     = (np.asarray(g_mdl_img)).reshape(nx, ny)
	g_res_img     = (g_obs_img - g_mdl_img)#abs
	g_res_img     = (np.asarray(g_res_img)).reshape(nx, ny)
	spec_file_sum     = Wrt_FITS_File(g_mdl_img,FITSFILENAME_MDL)#,overwrite=True)
	spec_file_sum     = Wrt_FITS_File(g_res_img,FITSFILENAME_RSD)#,overwrite=True)
	####################MODEL AND RESIDUAL####################

	if verbose == True:
		print
		print ('Initial Guess:')
		print ('X0_G         : ',X0_f2DG)
		print ('Y0_G         : ',Y0_f2DG)
		print ('A_G          : ',A_f2DG)
		print ('SIGMAX_G     : ',SIGMAX_f2DG)
		print ('SIGMAY_G     : ',SIGMAY_f2DG)
		print ('THETA_G      : ',THETA_f2DG)
		print ('OFFSET_G     : ',OFS_f2DG)
		print 
		print (colored(Message1,'yellow'))
		print (colored(Message2,'yellow'))

		print
		print ('Fit Values   :')
		print ('X0_F         : ',X0_F    ,' +- ',X0_E    )
		print ('Y0_F         : ',Y0_F    ,' +- ',Y0_E    )
		print ('A_F          : ',A_F     ,' +- ',A_E     )
		print ('SIGMAX_F     : ',SIGMAX_F,' +- ',SIGMAX_E)
		print ('SIGMAY_F     : ',SIGMAY_F,' +- ',SIGMAY_E)
		print ('THETA_F      : ',THETA_F ,' +- ',THETA_E )
		print ('OFFSET_F     : ',OFFSET_F,' +- ',OFFSET_E)
		print ('Area_F       : ',DGF_vlm ,' +- ',DGF_vle)
		print ('Volume_F     : ',DGF_vx1 ,' +- ',DGF_v1e)
		print
		print ('Shift from the X coordinate center:',X_DIF)
		print ('Shift from the Y coordinate center:',Y_DIF)
		print
		print (colored('Generated Plot: ' + str(PLOTFILENAME) ,'cyan'))
		print (colored('Generated Plot Residuals : ' + str(PLOTFILENAME_FMR) ,'cyan'))
		print
		print (colored('Generated FITS Model     : ' + str(FITSFILENAME_MDL) ,'green'))
		print (colored('Generated FITS Residuals : ' + str(FITSFILENAME_RSD) ,'green'))
		print
	elif verbose == False:
		pass
		
	if displ_s_f == True:
		##########################################FIT##########################################		
		fxsize=9
		fysize=8
		f = plt.figure(num=None, figsize=(fxsize, fysize), dpi=180, facecolor='w',
			edgecolor='k')
		plt.subplots_adjust(
			left 	= (25/25.4)/fxsize,    
			bottom 	= (16/25.4)/fysize,    
			right 	= 1 - (15/25.4)/fxsize,
			top 	= 1 - (15/25.4)/fysize)
		plt.subplots_adjust(hspace=0)

		gs0 = gridspec.GridSpec(1, 1)
		gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
		ax110 = plt.Subplot(f, gs11[0,0])
		f.add_subplot(ax110)

		ax110.set_rasterization_zorder(1)
		plt.autoscale(enable=True, axis='y', tight=False)

		plt.title(plt_tlt + ' (' +  str(x_ref) + ','+str(y_ref)+')',family='serif')
		plt.xlabel('X',fontsize=20,family = 'serif')
		plt.ylabel('Y',fontsize=20,family = 'serif')
		plt.tick_params(which='both', width=1.0)
		plt.tick_params(which='major', length=10)
		plt.tick_params(which='minor', length=5)
		ax110.minorticks_on()

		if ('_ms.' in Cube2bFit) or ('dta_in.' in Cube2bFit) or ('dta_ot.' in Cube2bFit):
			tick_color = 'white'
		elif ('msk_in.' in Cube2bFit) or ('crc.' in Cube2bFit) or ('msk_ot.' in Cube2bFit):
			tick_color = 'black'
		else:
			tick_color = 'white'

		ax110.xaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)
		ax110.yaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)

		cbar_min = np.amin(data_2b_plot)
		cbar_max = np.amax(data_2b_plot)

		plt.imshow(ydata.reshape(nx, ny), cmap=plt.cm.viridis, origin='lower',
		    extent=(x.min(), x.max(), y.min(), y.max()),
		    vmin = cbar_min, vmax = cbar_max)
		divider = make_axes_locatable(ax110)
		cax  = divider.append_axes("right", size="5%", pad=0.05)	
		cbar = plt.colorbar(cax=cax)
		cbar.set_label('S [Jy]', rotation=270,family = 'serif')

		min_y, max_y = ax110.get_ylim()
		min_x, max_x = ax110.get_xlim()	

		plt.text(0,max_y-(max_y/10),
				'1 pix = '  + str(scale_arcsec) + ' arcsec',  
				ha='left' , va='baseline',color='white',fontsize=20,
				family = 'serif')

		X0,Y0 = X0_F,Y0_F
		sigx  = SIGMAX_F
		sigy  = SIGMAY_F
		theta = THETA_F
		try:
			colors=['white','white','white','white','white']
			for j in xrange(1, 4):
			    ell = Ellipse(xy=(X0, Y0),
			        width=sigx*2*j, height=sigy*2*j,
			        angle=theta,
			        edgecolor=colors[j])
			    ell.set_facecolor('none')
			    ax110.add_artist(ell)
			
			x_text_coor = -20##IN TERMS OF COLORBAR NEW AXIS CAX
			y_text_coor = 0  ##IN TERMS OF COLORBAR NEW AXIS CAX

			if '-5as_' in Cube2bFit:
				step = .10*.5*abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))
			elif '-10as_' in Cube2bFit:
				step = .10*.3*abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))
			elif '-15as_' in Cube2bFit:
				step = .10*.2*abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))

			plt.text(x_text_coor,y_text_coor+(step*8),
				'Fit:',
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*7),
				'X$_{0}$: '           + str(x_ref+X_DIF)               + ', Y$_{0}$ '+ str(y_ref+Y_DIF)  + ', '
				'A : '                + str(np.round(popt[2],5))       + ' $\pm$ '   + str(np.round(A_E,5)),
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif',)
			plt.text(x_text_coor,y_text_coor+(step*6),
				'$\sigma_{x}$ : '     + str(np.round(abs(SIGMAX_F),5)) + ' $\pm$ '   + str(np.round(SIGMAX_E,5)),
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*5),
				'$\sigma_{y}$ : '     + str(np.round(abs(SIGMAY_F),5)) + ' $\pm$ '   + str(np.round(SIGMAY_E,5)),
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*4),
				'S(A) : '             + str(np.round(DGF_vlm,5))       + ' $\pm$ '   + str(np.round(DGF_vle,5)),
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*3),
				'S(V) : '             + str(np.round(DGF_vx1,5))       + ' $\pm$ '   + str(np.round(DGF_v1e,5)),
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*2),
				'S(V) : '             + str(np.round(DGF_vx2,5))       + ' $\pm$ '   + str(np.round(DGF_v2e,5)),
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*1),
				'$\sigma$ : '           + str(np.round(slice_sigm,5))  ,
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*0),
				'$\Delta$v : '           + str(np.round(vl_wdt_clp,5))  ,
				ha='left' , va='bottom',color='white',fontsize=20,
				family = 'serif')
		except ValueError:
			pass

		plt.scatter(X0_F    + 0.0, Y0_F    + 0.0, s=25, c='white', marker='x')
		plt.scatter(X0_f2DG + 0.0, Y0_f2DG + 0.0, s=25, c='black', marker='+')

		ax110.xaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-X0_f2DG+x_ref ))
		ax110.yaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-Y0_f2DG+y_ref ))

		plt.savefig(PLOTFILENAME)
		##########################################FIT##########################################

		#########################################IMAGE#########################################
		fxsize=11
		fysize=4
		f = plt.figure(num=None, figsize=(fxsize, fysize), dpi=180, facecolor='w',
			edgecolor='k')
		plt.subplots_adjust(
			left 	= (8/25.4)/fxsize,    
			bottom 	= (2/25.4)/fysize,    
			right 	= 1 - (2/25.4)/fxsize, 
			top 	= 1 - (4/25.4)/fysize) 
		plt.subplots_adjust(wspace=0)

		gs0 = gridspec.GridSpec(1, 3)
		ax110 = plt.Subplot(f, gs0[0])
		f.add_subplot(ax110)

		ax110.set_rasterization_zorder(1)
		plt.autoscale(enable=True, axis='y', tight=False)

		plt.title('FIT',family='serif')
		plt.tick_params(which='both', width=1.0)
		plt.tick_params(which='major', length=10)
		plt.tick_params(which='minor', length=5)
		ax110.minorticks_on()

		if ('_ms.' in Cube2bFit) or ('dta_in.' in Cube2bFit) or ('dta_ot.' in Cube2bFit):
			tick_color = 'white'
		elif ('msk_in.' in Cube2bFit) or ('crc.' in Cube2bFit) or ('msk_ot.' in Cube2bFit):
			tick_color = 'black'
		else:
			tick_color = 'white'

		ax110.xaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,
									bottom=True,top=True,left=True,right=True,
									labelleft=False, labeltop=False, labelright=False, labelbottom=False)
		ax110.yaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,
									bottom=True,top=True,left=True,right=True,
									labelleft=False, labeltop=False, labelright=False, labelbottom=False)

		plt.imshow(g_obs_img, cmap=plt.cm.viridis, origin='lower',
		    extent=(x.min(), x.max(), y.min(), y.max()),
		    vmin = cbar_min, vmax = cbar_max)

		min_y, max_y = ax110.get_ylim()
		min_x, max_x = ax110.get_xlim()	

		X0,Y0 = X0_F,Y0_F
		sigx  = SIGMAX_F
		sigy  = SIGMAY_F
		theta = THETA_F
		try:
			colors=['white','white','white','white','white']
			for j in xrange(1, 4):
			    ell = Ellipse(xy=(X0, Y0),
			        width=sigx*2*j, height=sigy*2*j,
			        angle=theta,
			        edgecolor=colors[j])
			    ell.set_facecolor('none')
			    ax110.add_artist(ell)
			x_text_coor = 0
			y_text_coor = 0
			step = abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))

			x_text_coor =   0   
			y_text_coor =   0   
			if '-5as_' in Cube2bFit:
				step = 1*.5*abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))
			elif '-10as_' in Cube2bFit:
				step = 1*.5*abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))
			elif '-15as_' in Cube2bFit:
				step = 1*.5*abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))

			plt.text(x_text_coor,y_text_coor+(step*9),
				'Fit:',
				ha='left' , va='bottom',color='white',fontsize=10,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*8),
				'X$_{0}$: '       + str(x_ref+X_DIF)          + ', Y$_{0}$ '+ str(y_ref+Y_DIF)  + ', '
				'A : '            + str(np.round(popt[2],3))  + ' $\pm$ '   + str(np.round(A_E,5)),
				family = 'serif',
				ha='left' , va='bottom',color='white',fontsize=10)
			plt.text(x_text_coor,y_text_coor+(step*7),
				'A$_{0}$: : '       + str(np.round(OFFSET_F,5))  + ' $\pm$ '   + str(np.round(OFFSET_E,5)),
				ha='left' , va='bottom',color='white',fontsize=10,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*6),
				'$\sigma_{x}$ : ' + str(np.round(abs(SIGMAX_F),3)) + ' $\pm$ ' + str(np.round(SIGMAX_E,3)),
				ha='left' , va='bottom',color='white',fontsize=10,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*5),
				'$\sigma_{y}$ : ' + str(np.round(abs(SIGMAY_F),3)) + ' $\pm$ ' + str(np.round(SIGMAY_E,3)),
				ha='left' , va='bottom',color='white',fontsize=10,
				family = 'serif')
			plt.text(x_text_coor,y_text_coor+(step*4),
				'S(A) : '           + str(np.round(DGF_vlm,3))  + ' $\pm$ ' + str(np.round(DGF_vle,3)),
				ha='left' , va='bottom',color='white',fontsize=10,
				family = 'serif')
			if 'IM0' not in Cube2bFit:
				plt.text(x_text_coor,y_text_coor+(step*3),
					'S(V) : '           + str(np.round(DGF_vx1,3))  + ' $\pm$ ' + str(np.round(DGF_v1e,3)),
					ha='left' , va='bottom',color='white',fontsize=10,
					family = 'serif')
				plt.text(x_text_coor,y_text_coor+(step*2),
					'S(V) : '           + str(np.round(DGF_vx2,3))  + ' $\pm$ ' + str(np.round(DGF_v2e,3)),
					ha='left' , va='bottom',color='white',fontsize=10,
					family = 'serif')
				plt.text(x_text_coor,y_text_coor+(step*1),
					'$\sigma$ : '           + str(np.round(slice_sigm,3))  ,
					ha='left' , va='bottom',color='white',fontsize=10,
					family = 'serif')
				plt.text(x_text_coor,y_text_coor+(step*0),
					'$\Delta$v : '           + str(np.round(vl_wdt_clp,3))  ,
					ha='left' , va='bottom',color='white',fontsize=10,
					family = 'serif')
			elif 'IM0' in Cube2bFit:
				plt.text(x_text_coor,y_text_coor+(step*3),
					'$\sigma$ : '           + str(np.round(slice_sigm,3))  ,
					ha='left' , va='bottom',color='white',fontsize=10,
					family = 'serif')
				plt.text(x_text_coor,y_text_coor+(step*2),
					'$\Delta$v : '           + str(np.round(vl_wdt_clp,3))  ,
					ha='left' , va='bottom',color='white',fontsize=10,
					family = 'serif')
		except ValueError:
			pass

		plt.scatter(X0_F    + 0.0, Y0_F    + 0.0, s=25, c='white', marker='x')
		plt.scatter(X0_f2DG + 0.0, Y0_f2DG + 0.0, s=25, c='black', marker='+')

		ax110.xaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-X0_f2DG + x_ref )) 
		ax110.yaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-Y0_f2DG + y_ref )) 

		min_x,max_x = ax110.get_xlim()
		min_y,max_y = ax110.get_ylim()
		print
		print (colored('Scale values:','yellow'))
		print ('pix: ',abs(max_x-(abs(min_x-max_x)*.10)) - (max_x-(abs(min_x-max_x)*.265)))
		print ('as : ',0.5*(abs(max_x-(abs(min_x-max_x)*.10)) - (max_x-(abs(min_x-max_x)*.265))))
		plt.plot([max_x-(abs(min_x-max_x)*.265), max_x-(abs(min_x-max_x)*.10)],
				 [max_y-(abs(min_y-max_y)*.100), max_y-(abs(min_y-max_y)*.10)],
				 color='white', lw=1, alpha=0.6,ls='-',label='5 as')
		plt.text(max_x-(abs(min_x-max_x)*.18),max_y-(abs(min_y-max_y)*.18),
			str('5 as'),
			ha='center' , va='bottom',color='white',fontsize=12,
			family = 'serif')		
		#########################################IMAGE#########################################
		#########################################MODEL#########################################
		ax120 = plt.Subplot(f, gs0[1])
		f.add_subplot(ax120)

		ax120.set_rasterization_zorder(1)
		plt.autoscale(enable=True, axis='y', tight=False)

		plt.title('MODEL',family='serif')
		plt.tick_params(which='both', width=1.0)
		plt.tick_params(which='major', length=10)
		plt.tick_params(which='minor', length=5)
		ax120.minorticks_on()

		if ('_ms.' in Cube2bFit) or ('dta_in.' in Cube2bFit) or ('dta_ot.' in Cube2bFit):
			tick_color = 'white'
		elif ('msk_in.' in Cube2bFit) or ('crc.' in Cube2bFit) or ('msk_ot.' in Cube2bFit):
			tick_color = 'black'
		else:
			tick_color = 'white'

		ax120.xaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,
									bottom=True,top=True,left=True,right=True,
									labelleft=False, labeltop=False, labelright=False, labelbottom=False)
		ax120.yaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,
									bottom=True,top=True,left=True,right=True,
									labelleft=False, labeltop=False, labelright=False, labelbottom=False)

		plt.imshow(g_mdl_img, cmap=plt.cm.viridis, origin='lower',
		    extent=(x.min(), x.max(), y.min(), y.max()),
		    vmin = cbar_min, vmax = cbar_max)

		min_y, max_y = ax120.get_ylim()
		min_x, max_x = ax120.get_xlim()	

		X0,Y0 = X0_F,Y0_F

		sigx  = SIGMAX_F
		sigy  = SIGMAY_F
		theta = THETA_F


		plt.scatter(X0_F    + 0.0, Y0_F    + 0.0, s=25, c='white', marker='x')
		plt.scatter(X0_f2DG + 0.0, Y0_f2DG + 0.0, s=25, c='black', marker='+')

		plt.plot([max_x-(abs(min_x-max_x)*.265), max_x-(abs(min_x-max_x)*.10)],
				 [max_y-(abs(min_y-max_y)*.100), max_y-(abs(min_y-max_y)*.10)],
				 color='white', lw=1, alpha=0.6,ls='-',label='5 as')
		plt.text(max_x-(abs(min_x-max_x)*.18),max_y-(abs(min_y-max_y)*.18),
			str('5 as'),
			ha='center' , va='bottom',color='white',fontsize=12,
			family = 'serif')
		#########################################MODEL#########################################
		#######################################RESIDUAL########################################
		ax130 = plt.Subplot(f, gs0[2])
		f.add_subplot(ax130)

		ax130.set_rasterization_zorder(1)
		plt.autoscale(enable=True, axis='y', tight=False)

		plt.title('RESIDUAL',family='serif')
		plt.tick_params(which='both', width=1.0)
		plt.tick_params(which='major', length=10)
		plt.tick_params(which='minor', length=5)
		ax130.minorticks_on()

		if ('_ms.' in Cube2bFit) or ('dta_in.' in Cube2bFit) or ('dta_ot.' in Cube2bFit):
			tick_color = 'white'
		elif ('msk_in.' in Cube2bFit) or ('crc.' in Cube2bFit) or ('msk_ot.' in Cube2bFit):
			tick_color = 'black'
		else:
			tick_color = 'white'

		ax130.xaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,
									bottom=True,top=True,left=True,right=True,
									labelleft=False, labeltop=False, labelright=False, labelbottom=False)
		ax130.yaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,
									bottom=True,top=True,left=True,right=True,
									labelleft=False, labeltop=False, labelright=False, labelbottom=False)

		plt.imshow(g_res_img, cmap=plt.cm.viridis, origin='lower',
		    extent=(x.min(), x.max(), y.min(), y.max()),
		    vmin = cbar_min, vmax = cbar_max)
		min_y, max_y = ax130.get_ylim()
		min_x, max_x = ax130.get_xlim()	

		X0,Y0 = X0_F,Y0_F
		sigx  = SIGMAX_F
		sigy  = SIGMAY_F
		theta = THETA_F

		plt.scatter(X0_F    + 0.0, Y0_F    + 0.0, s=25, c='white', marker='x')
		plt.scatter(X0_f2DG + 0.0, Y0_f2DG + 0.0, s=25, c='black', marker='+')
		plt.plot([max_x-(abs(min_x-max_x)*.265), max_x-(abs(min_x-max_x)*.10)],
				 [max_y-(abs(min_y-max_y)*.100), max_y-(abs(min_y-max_y)*.10)],
				 color='white', lw=1, alpha=0.6,ls='-',label='5 as')
		plt.text(max_x-(abs(min_x-max_x)*.18),max_y-(abs(min_y-max_y)*.18),
			str('5 as'),
			ha='center' , va='bottom',color='white',fontsize=12,
			family = 'serif')

		plt.savefig(PLOTFILENAME_FMR)
		#######################################RESIDUAL########################################		
	elif displ_s_f == False:
		pass
	return popt,pcov,perr,fit_res,X_DIF,Y_DIF

def Cube_fit_2D_Gaussian_Noise(Cube2bFit,*args,**kwargs):
	slc_nmb      = kwargs.get('slc_nmb' ,None)
	dest_dir     = kwargs.get('dest_dir',None)
	verbose      = kwargs.get('verbose' ,None)
	clp_fnc      = kwargs.get('clp_fnc' ,'sum')
	sgm_fnc      = kwargs.get('sgm_fnc' ,'avg')
	circular     = kwargs.get('circular',True)
	x_ref        = kwargs.get('x_ref',0)
	y_ref        = kwargs.get('y_ref',0)

	dest_dir_plt = kwargs.get('dest_dir_plt',None)
	dest_dir_clp = kwargs.get('dest_dir_clp',None)

	frq_r        = kwargs.get('frq_r',restframe_frequency)
	z_avg        = kwargs.get('z_avg',Header_Get(Cube2bFit,'STZ_AVG'))
	z_med        = kwargs.get('z_med',Header_Get(Cube2bFit,'STZ_MED'))
	z_f2l        = z_med

	sgm_wgth_tms = kwargs.get('sgm_wgth_tms','5sgm')

	ref_wdt_lne    = kwargs.get('ref_wdt_lne',False)
	ref_wdt_fle    = kwargs.get('ref_wdt_fle',None)


	if ref_wdt_lne == True:
		Cube2bFit_Hdr = ref_wdt_fle
	elif ref_wdt_lne == False:
		Cube2bFit_Hdr = Cube2bFit

	if slc_nmb == None:
		if sgm_fnc == 'avg':
			slice_nmbr   = (Header_Get(Cube2bFit_Hdr,'MAX_SNS'))
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTA_SIG'))
			slice_fwhm   = (Header_Get(Cube2bFit_Hdr,'FTA_FWH'))
			slice_cct    = (Header_Get(Cube2bFit_Hdr,'FTA_CCT'))
			slice_1sg    = (Header_Get(Cube2bFit_Hdr,'FTA_1SG'))
			slice_2sg    = (Header_Get(Cube2bFit_Hdr,'FTA_2SG'))
			slice_3sg    = (Header_Get(Cube2bFit_Hdr,'FTA_3SG'))
			slice_4sg    = (Header_Get(Cube2bFit_Hdr,'FTA_4SG'))
			slice_5sg    = (Header_Get(Cube2bFit_Hdr,'FTA_5SG'))
			slice_1fw    = (Header_Get(Cube2bFit_Hdr,'FTA_1FW'))
			slice_2fw    = (Header_Get(Cube2bFit_Hdr,'FTA_2FW'))

		elif sgm_fnc == 'med':
			slice_nmbr   = (Header_Get(Cube2bFit_Hdr,'MAX_SNM'))
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTM_SIG'))
			slice_fwhm   = (Header_Get(Cube2bFit_Hdr,'FTM_FWH'))
			slice_cct    = (Header_Get(Cube2bFit_Hdr,'FTM_CCT'))
			slice_1sg    = (Header_Get(Cube2bFit_Hdr,'FTM_1SG'))
			slice_2sg    = (Header_Get(Cube2bFit_Hdr,'FTM_2SG'))
			slice_3sg    = (Header_Get(Cube2bFit_Hdr,'FTM_3SG'))
			slice_4sg    = (Header_Get(Cube2bFit_Hdr,'FTM_4SG'))
			slice_5sg    = (Header_Get(Cube2bFit_Hdr,'FTM_5SG'))
			slice_1fw    = (Header_Get(Cube2bFit_Hdr,'FTM_1FW'))
			slice_2fw    = (Header_Get(Cube2bFit_Hdr,'FTM_2FW'))

		elif sgm_fnc == 'sum':
			slice_nmbr   = (Header_Get(Cube2bFit_Hdr,'MAX_SNA'))
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTS_SIG'))
			slice_fwhm   = (Header_Get(Cube2bFit_Hdr,'FTS_FWH'))
			slice_cct    = (Header_Get(Cube2bFit_Hdr,'FTS_CCT'))
			slice_1sg    = (Header_Get(Cube2bFit_Hdr,'FTS_1SG'))
			slice_2sg    = (Header_Get(Cube2bFit_Hdr,'FTS_2SG'))
			slice_3sg    = (Header_Get(Cube2bFit_Hdr,'FTS_3SG'))
			slice_4sg    = (Header_Get(Cube2bFit_Hdr,'FTS_4SG'))
			slice_5sg    = (Header_Get(Cube2bFit_Hdr,'FTS_5SG'))
			slice_1fw    = (Header_Get(Cube2bFit_Hdr,'FTS_1FW'))
			slice_2fw    = (Header_Get(Cube2bFit_Hdr,'FTS_2FW'))

		if dest_dir_clp != None:
			Cube2bclp_2D_opt = str(dest_dir_clp)  + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+clp_fnc+'-nse.fits'
		elif dest_dir_clp == None:
			Cube2bclp_2D_opt = stp_dir_res + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+clp_fnc+'-nse.fits'
		if dest_dir_plt != None:
			PLOTFILENAME     = str(dest_dir_plt) + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-nse.pdf'
		elif dest_dir_plt == None:
			PLOTFILENAME     = ana_dir_plt + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+clp_fnc+'-nse.pdf'

	elif slc_nmb != None:
		slice_nmbr = 'CSL'
		if dest_dir_clp != None:
			Cube2bclp_2D_opt = str(dest_dir_clp)   + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+slice_nmbr+'-nse.fits'
		elif dest_dir_clp == None:
			Cube2bclp_2D_opt = stp_dir_res + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+slice_nmbr+'-nse.fits'
		if dest_dir_plt != None:
			PLOTFILENAME     = str(dest_dir_plt) + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+slice_nmbr+str(slc_nmb)+'-nse.pdf'
		elif dest_dir_plt == None:
			PLOTFILENAME     = ana_dir_plt + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+slice_nmbr+str(slc_nmb)+'-nse.pdf'

	Cube_Info    = Cube_Header_Get(Cube2bFit,frq_r* u.Hz)
	FRQ_AXS      = Cube_Info[16].value
	VEL_AXS      = Cube_Info[17].value

	scale_deg    = Header_Get(Cube2bFit,'CDELT2')
	scale_arcsec = scale_deg*3600#0.00027777778

	cube_data    = np.asarray(apgtdt(Cube2bFit,memmap=False))
	slice_cwdt   = (Header_Get(Cube2bFit_Hdr,'STT_VEL'))

	if slc_nmb == None:
		if sgm_wgth_tms == '1sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*1)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*1))
			slice_nblw = int(slice_cct - slice_1sg)
			slice_nbhg = int(slice_cct + slice_1sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_1sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*1
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
			print ('Center : '+str(slice_cct))
			print ('Width  : '+str(slice_1fw))
			print ('Lower  : '+str(slice_nblw))
			print ('Upper  : '+str(slice_nbhg))
			print
		elif sgm_wgth_tms == '2sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*2)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*2))
			slice_nblw = int(slice_cct - slice_2sg)
			slice_nbhg = int(slice_cct + slice_2sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_2sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*2
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
			print ('Center : '+str(slice_cct))
			print ('Width  : '+str(slice_1fw))
			print ('Lower  : '+str(slice_nblw))
			print ('Upper  : '+str(slice_nbhg))
			print			
		elif sgm_wgth_tms == '3sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*3)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*3))
			slice_nblw = int(slice_cct - slice_3sg)
			slice_nbhg = int(slice_cct + slice_3sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_3sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*3
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
			print ('Center : '+str(slice_cct))
			print ('Width  : '+str(slice_1fw))
			print ('Lower  : '+str(slice_nblw))
			print ('Upper  : '+str(slice_nbhg))
			print			
		elif sgm_wgth_tms == '4sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*4)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*4))
			slice_nblw = int(slice_cct - slice_4sg)
			slice_nbhg = int(slice_cct + slice_4sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_4sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*4
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
			print ('Center : '+str(slice_cct))
			print ('Width  : '+str(slice_1fw))
			print ('Lower  : '+str(slice_nblw))
			print ('Upper  : '+str(slice_nbhg))
			print			
		elif sgm_wgth_tms == '5sgm':
			print
			print (colored('Velocity width interval (sgm)  :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',slice_sigm*5)
			print ('Cube fitting fwhm (avg) km/s: ',sigma2fwhm(slice_sigm*5))
			slice_nblw = int(slice_cct - slice_5sg)
			slice_nbhg = int(slice_cct + slice_5sg)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_5sg)
			slice_wdnb = dlt_ch_nmb
			vl_wdt_clp = slice_sigm*5
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
			print ('Center : '+str(slice_cct))
			print ('Width  : '+str(slice_1fw))
			print ('Lower  : '+str(slice_nblw))
			print ('Upper  : '+str(slice_nbhg))
			print			
		elif sgm_wgth_tms == 'slice_1fw':
			print
			print (colored('Velocity width interval (fwhm) :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',fwhm2sigma(slice_fwhm*2))
			print ('Cube fitting fwhm (avg) km/s: ',slice_fwhm*2)
			slice_nblw = int(slice_cct - slice_1fw)
			slice_nbhg = int(slice_cct + slice_1fw)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_1fw)
			slice_wdnb = dlt_ch_nmb
			slice_sig = fwhm2sigma(slice_fwhm*2)
			vl_wdt_clp = slice_fwhm*2
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
			print ('Center : '+str(slice_cct))
			print ('Width  : '+str(slice_1fw))
			print ('Lower  : '+str(slice_nblw))
			print ('Upper  : '+str(slice_nbhg))
			print			
		elif sgm_wgth_tms == 'slice_2fw':
			print
			print (colored('Velocity width interval (fwhm) :  ' + sgm_wgth_tms,'yellow'))
			print ('Cube fitting sigm (avg) km/s: ',fwhm2sigma(slice_fwhm*4))
			print ('Cube fitting fwhm (avg) km/s: ',slice_fwhm*4)
			slice_nblw = int(slice_cct - slice_2fw)
			slice_nbhg = int(slice_cct + slice_2fw)
			if slice_nblw < 0:
				slice_nblw = 0
			else:
				pass
			if slice_nbhg >= len(VEL_AXS):
				slice_nbhg = -1
			else:
				pass			
			slice_vllw = VEL_AXS[slice_nblw]
			slice_vlhg = VEL_AXS[slice_nbhg]
			dlt_ch_nmb = int(slice_2fw)
			slice_wdnb = dlt_ch_nmb
			slice_sigm = fwhm2sigma(slice_fwhm*4)
			vl_wdt_clp = slice_fwhm*4
			tlt_ch_nmb = (dlt_ch_nmb*2)+1
			print ('Center : '+str(slice_cct))
			print ('Width  : '+str(slice_1fw))
			print ('Lower  : '+str(slice_nblw))
			print ('Upper  : '+str(slice_nbhg))
			print			
	elif slc_nmb != None:
		slice_nblw = slc_nmb
		slice_nbhg = slc_nmb
		slice_vllw = 0
		slice_vlhg = 0
		dlt_ch_nmb = 1
		slice_wdnb = dlt_ch_nmb
		vl_wdt_clp = slice_cwdt
		tlt_ch_nmb = 1
		if sgm_fnc == 'avg':
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTA_SIG'))
		elif sgm_fnc == 'med':
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTM_SIG'))
		elif sgm_fnc == 'sum':
			slice_sigm   = (Header_Get(Cube2bFit_Hdr,'FTS_SIG'))
		slice_sigm = slice_sigm
		pass

	if slc_nmb != None:
		data_2b_plot = cube_data[slc_nmb]
		Message1 = 'Fitting gaussian with slice number : ' + str(slc_nmb+1)
		Message2 = 'For datacube : ' + Cube2bFit
		plt_tlt = 'Slice: ' + str(slc_nmb+1) + '-' + str(round(VEL_AXS[slc_nmb],0)) + ' km/s'
		clp_hdr = 'C'  
		clp_hdc = 'CSL' 
		data_2b_plt = np.asarray(apgtdt(Cube2bFit,memmap=False))
		data_2b_plt_clp = data_2b_plt[slc_nmb]

	elif slc_nmb == None:
		Message1 = 'Collapsing (' +str(clp_fnc)+')' + ' noise errors outside line domain NO FIT IS PERFORMED! :'
		Message2 = 'For datacube : ' + Cube2bFit		
		plt_tlt = '2D Collapse ('+str(clp_fnc).upper()+') VW:' + str(int(vl_wdt_clp)) + 'km/s ('+sgm_wgth_tms+') ['+ str(int((tlt_ch_nmb))) + ']'
		data_2b_plt = np.asarray(apgtdt(Cube2bFit,memmap=False))

		#CHOOSE SLICES OUTSIDE LIMITS FOR FLUX MEASURMENTS#
		if slice_nblw!=slice_nbhg:
			indxs_clp = []
			indxs_all = np.arange(0,len(VEL_AXS),1)
			indxs_fbd = np.arange(slice_nblw,slice_nbhg+1,1)
			for j in indxs_all:
				if j not in indxs_fbd and (len(indxs_clp)<tlt_ch_nmb):
					indxs_clp.append(j)
				elif j not in indxs_fbd and (len(indxs_clp)>=tlt_ch_nmb):
					break
			data_2b_plt_clp = data_2b_plt[indxs_clp]
			skp_clp = False
			print
			print (colored('Line-width > 0','yellow'))
			print (colored('Using a channels for noise map','yellow'))
			print (indxs_fbd,len(indxs_fbd))
			print (indxs_clp)
			print
		elif slice_nblw==slice_nbhg:
			data_2b_plt_clp = data_2b_plt[-2]
			skp_clp = True
			print
			print (colored('Line-width = 0!','yellow'))
			print (colored('Using a single channel ('+str(-2)+') for noise map','yellow'))
			print
		#CHOOSE SLICES OUTSIDE LIMITS FOR FLUX MEASURMENTS#

		if clp_fnc == 'sum' and skp_clp == False:
			data_2b_plot = np.asarray(np.nansum(np.array(data_2b_plt_clp)   , axis=0))
			clp_hdr = 'S'  
			clp_hdc = 'SUM'  
			slice_nmbr = (Header_Get(Cube2bFit_Hdr,'MAX_SNS')) 
		elif clp_fnc == 'med' and skp_clp == False:
			data_2b_plot = np.asarray(np.nanmedian(np.array(data_2b_plt_clp), axis=0))
			clp_hdr = 'M'  
			clp_hdc = 'MED'  
			slice_nmbr = (Header_Get(Cube2bFit_Hdr,'MAX_SNM')) 
		elif clp_fnc == 'avg' and skp_clp == False:
			data_2b_plot = np.asarray(np.nanmean(np.array(data_2b_plt_clp)  , axis=0))
			clp_hdr = 'A'
			clp_hdc = 'AVG'
			slice_nmbr = (Header_Get(Cube2bFit_Hdr,'MAX_SNA')) 
		elif clp_fnc == 'sum' and skp_clp == True:
			data_2b_plot = data_2b_plt_clp
			clp_hdr = 'S'  
			clp_hdc = 'SUM'  
			slice_nmbr = (Header_Get(Cube2bFit_Hdr,'MAX_SNS')) 
		elif clp_fnc == 'med' and skp_clp == True:
			data_2b_plot = data_2b_plt_clp
			clp_hdr = 'M'  
			clp_hdc = 'MED'  
			slice_nmbr = (Header_Get(Cube2bFit_Hdr,'MAX_SNM')) 
		elif clp_fnc == 'avg' and skp_clp == True:
			data_2b_plot = data_2b_plt_clp
			clp_hdr = 'A'
			clp_hdc = 'AVG'
			slice_nmbr = (Header_Get(Cube2bFit_Hdr,'MAX_SNA')) 
	print
	print (colored(Message1,'yellow'))
	print (colored(Message2,'yellow'))
	print
	Wrt_FITS_File(data_2b_plot,Cube2bclp_2D_opt)

	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'XXT_MIN',header_comment = 'Image Extent X MIN')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'XXT_MAX',header_comment = 'Image Extent X MAX')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'YXT_MIN',header_comment = 'Image Extent Y MIN')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'YXT_MAX',header_comment = 'Image Extent Y MAX')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'XCT_FIT',header_comment = 'Image Center X ')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'YCT_FIT',header_comment = 'Image Center Y ')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'RAD_EXT',header_comment = 'Image Extent Radii')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STK_NUM',header_comment = 'Number of galaxies used for Stack')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_AVG',header_comment = 'Redshift Average ')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_MED',header_comment = 'Redshift Median ')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_1SL',header_comment = 'Redshift 1 sgm lw lmt 15.9 pct') 
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_1SH',header_comment = 'Redshift 1 sgm hg lmt 84.1 pct')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_2SL',header_comment = 'Redshift 2 sgm lw lmt 2.30 pct')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_2SH',header_comment = 'Redshift 2 sgm hg lmt 97.7 pct')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_3SL',header_comment = 'Redshift 3 sgm lw lmt 0.20 pct')  
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_3SH',header_comment = 'Redshift 3 sgm hg lmt 99.8 pct')
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_P25',header_comment = 'Redshift 25 pct')                   
	Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STZ_P75',header_comment = 'Redshift 75 pct') 

	try:
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMS_AVG',header_comment = 'Synthetic Cubes Source Amplitude AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMN_AVG',header_comment = 'Synthetic Cubes Noise Amplitude  AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SNR_AVG',header_comment = 'Synthetic Cubes SNR Amplitude    AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGN_AVG',header_comment = 'Synthetic Cubes Sigma chan num   AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGV_AVG',header_comment = 'Synthetic Cubes Sigma Vel kms-1  AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FWH_AVG',header_comment = 'Synthetic Cubes FWHM Vel kms-1   AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'OFS_AVG',header_comment = 'Synthetic Cubes Offset           AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMS_MED',header_comment = 'Synthetic Cubes Source Amplitude MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMN_MED',header_comment = 'Synthetic Cubes Noise Amplitude  MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SNR_MED',header_comment = 'Synthetic Cubes SNR Amplitude    MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGN_MED',header_comment = 'Synthetic Cubes Sigma chan num   MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGV_MED',header_comment = 'Synthetic Cubes Sigma Vel kms-1  MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FWH_MED',header_comment = 'Synthetic Cubes FWHM Vel kms-1   MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'OFS_MED',header_comment = 'Synthetic Cubes Offset           MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMS_STD',header_comment = 'Synthetic Cubes Source Amplitude STD')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMN_STD',header_comment = 'Synthetic Cubes Noise Amplitude  STD')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SNR_STD',header_comment = 'Synthetic Cubes SNR Amplitude    STD')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGN_STD',header_comment = 'Synthetic Cubes Sigma chan num   STD')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGV_STD',header_comment = 'Synthetic Cubes Sigma Vel kms-1  STD')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FWH_STD',header_comment = 'Synthetic Cubes FWHM Vel kms-1   STD')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'OFS_STD',header_comment = 'Synthetic Cubes Offset           STD')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMS_MIN',header_comment = 'Synthetic Cubes Source Amplitude MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMN_MIN',header_comment = 'Synthetic Cubes Noise Amplitude  MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SNR_MIN',header_comment = 'Synthetic Cubes SNR Amplitude    MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGN_MIN',header_comment = 'Synthetic Cubes Sigma chan num   MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGV_MIN',header_comment = 'Synthetic Cubes Sigma Vel kms-1  MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FWH_MIN',header_comment = 'Synthetic Cubes FWHM Vel kms-1   MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'OFS_MIN',header_comment = 'Synthetic Cubes Offset           MIN')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMS_MAX',header_comment = 'Synthetic Cubes Source Amplitude MAX')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'AMN_MAX',header_comment = 'Synthetic Cubes Noise Amplitude  MAX')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SNR_MAX',header_comment = 'Synthetic Cubes SNR Amplitude    MAX')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGN_MAX',header_comment = 'Synthetic Cubes Sigma chan num   MAX')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'SGV_MAX',header_comment = 'Synthetic Cubes Sigma Vel kms-1  MAX')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FWH_MAX',header_comment = 'Synthetic Cubes FWHM Vel kms-1   MAX')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'OFS_MAX',header_comment = 'Synthetic Cubes Offset           MAX')
	except KeyError:
		pass

	try:
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_VEL',header_comment = 'CbeWth [km/s]')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_FLS',header_comment = 'TFlx SUM All Chns')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TFL',header_comment = 'TFlx SUM All Chns * CbeWth')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LMS',header_comment = 'TFlx SUM All CH 2 Lum')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LLM',header_comment = 'TFlx SUM All CH 2 Lum [log]')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LMT',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LLT',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log]')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_FLE',header_comment = 'TFlx SUM All Chns Err')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TFE',header_comment = 'TFlx SUM All Chns * CbeWth Err')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SL1',header_comment = 'TFlx SUM All CH 2 Lum Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SH1',header_comment = 'TFlx SUM All CH 2 Lum Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LL1',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LH1',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_ML1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_MH1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TL1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TH1',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SL2',header_comment = 'TFlx SUM All CH 2 Lum Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SH2',header_comment = 'TFlx SUM All CH 2 Lum Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LL2',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LH2',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_ML2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_MH2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TL2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TH2',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SL3',header_comment = 'TFlx SUM All CH 2 Lum Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_SH3',header_comment = 'TFlx SUM All CH 2 Lum Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LL3',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_LH3',header_comment = 'TFlx SUM All CH 2 Lum [log] Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_ML3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_MH3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TL3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'STT_TH3',header_comment = 'TFlx SUM All CH X CbeWth 2 Lum [log] Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SNS',header_comment = 'Vel Prf Max Chn Location SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SVS',header_comment = 'Vel Prf Max Frequency Location [Hz] SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_VLS',header_comment = 'Vel Prf Max Chn Value SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SNA',header_comment = 'Vel Prf Max Chn Location AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SVA',header_comment = 'Vel Prf Max Frequency Location [Hz] AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_VLA',header_comment = 'Vel Prf Max Chn Value AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SNM',header_comment = 'Vel Prf Max Chn Location MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_SVM',header_comment = 'Vel Prf Max Frequency Location [Hz] MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'MAX_VLM',header_comment = 'Vel Prf Max Chn Value MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_AMP',header_comment = '1DGF Amplitude AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_CTR',header_comment = '1DGF Center AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_SIG',header_comment = '1DGF Sigma AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_FWH',header_comment = '1DGF FWHM  AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_A2A',header_comment = '1DGF Area A AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_A2M',header_comment = '1DGF Area M AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_CH2',header_comment = '1DGF Chi2 AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_CHR',header_comment = '1DGF Chi2 Reduced AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LUM',header_comment = '1DGF Ar2Lum AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LLM',header_comment = '1DGF Ar2Lum [log] AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_APE',header_comment = '1DGF Amplitude Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_CTE',header_comment = '1DGF Center Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_SGE',header_comment = '1DGF Sigma Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_FWE',header_comment = '1DGF FWHM  Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_AAE',header_comment = '1DGF Area A Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_AME',header_comment = '1DGF Area M Err AVG')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_ML1',header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_MH1',header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LL1',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LH1',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_ML2',header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_MH2',header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LL2',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LH2',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_ML3',header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_MH3',header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LL3',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_LH3',header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_ML1',header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_MH1',header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LL1',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LH1',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_ML2',header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_MH2',header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LL2',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LH2',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_ML3',header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_MH3',header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LL3',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMA_LH3',header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_CCT',header_comment = 'Channel # Central AVG (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_1SG',header_comment = 'Channel Dif Limit 1SG AVG (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_2SG',header_comment = 'Channel Dif Limit 2SG AVG (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_3SG',header_comment = 'Channel Dif Limit 3SG AVG (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_4SG',header_comment = 'Channel Dif Limit 4SG AVG (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_5SG',header_comment = 'Channel Dif Limit 5SG AVG (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_1FW',header_comment = 'Channel Dif Limit 1FW AVG (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTA_2FW',header_comment = 'Channel Dif Limit 2FW AVG (PyId)')
	except KeyError:
		pass

	try:
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_AMP',header_comment = '1DGF Amplitude MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_CTR',header_comment = '1DGF Center MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_SIG',header_comment = '1DGF Sigma MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_FWH',header_comment = '1DGF FWHM  MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_A2A',header_comment = '1DGF Area A MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_A2M',header_comment = '1DGF Area M MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_CH2',header_comment = '1DGF Chi2 MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_CHR',header_comment = '1DGF Chi2 Reduced MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LUM',header_comment = '1DGF Ar2Lum MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LLM',header_comment = '1DGF Ar2Lum [log] MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_APE',header_comment = '1DGF Amplitude Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_CTE',header_comment = '1DGF Center Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_SGE',header_comment = '1DGF Sigma Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_FWE',header_comment = '1DGF FWHM  Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_AAE',header_comment = '1DGF Area A Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_AME',header_comment = '1DGF Area M Err MED')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_ML1',header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_MH1',header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LL1',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LH1',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_ML2',header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_MH2',header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LL2',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LH2',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_ML3',header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_MH3',header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LL3',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_LH3',header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_ML1',header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_MH1',header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LL1',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LH1',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_ML2',header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_MH2',header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LL2',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LH2',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_ML3',header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_MH3',header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LL3',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMM_LH3',header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_CCT',header_comment = 'Channel # Central MED (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_1SG',header_comment = 'Channel Dif Limit 1SG MED (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_2SG',header_comment = 'Channel Dif Limit 2SG MED (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_3SG',header_comment = 'Channel Dif Limit 3SG MED (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_4SG',header_comment = 'Channel Dif Limit 4SG MED (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_5SG',header_comment = 'Channel Dif Limit 5SG MED (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_1FW',header_comment = 'Channel Dif Limit 1FW MED (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTM_2FW',header_comment = 'Channel Dif Limit 2FW MED (PyId)')
	except KeyError:
		pass

	try:

		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_AMP',header_comment = '1DGF Amplitude SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_CTR',header_comment = '1DGF Center SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_SIG',header_comment = '1DGF Sigma SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_FWH',header_comment = '1DGF FWHM  SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_A2A',header_comment = '1DGF Area A SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_A2M',header_comment = '1DGF Area M SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_CH2',header_comment = '1DGF Chi2 SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_CHR',header_comment = '1DGF Chi2 Reduced M SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LUM',header_comment = '1DGF Ar2Lum SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LLM',header_comment = '1DGF Ar2Lum [log] SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_APE',header_comment = '1DGF Amplitude Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_CTE',header_comment = '1DGF Center Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_SGE',header_comment = '1DGF Sigma Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_FWE',header_comment = '1DGF FWHM Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_AAE',header_comment = '1DGF Area A Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_AME',header_comment = '1DGF Area M Err SUM')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_ML1',header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_MH1',header_comment = '1DGF Ar2Lum Lum A SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LL1',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LH1',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_ML2',header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_MH2',header_comment = '1DGF Ar2Lum Lum A SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LL2',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LH2',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_ML3',header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_MH3',header_comment = '1DGF Ar2Lum Lum A SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LL3',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_LH3',header_comment = '1DGF Ar2Lum Lum A [log] SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_ML1',header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_MH1',header_comment = '1DGF Ar2Lum Lum M SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LL1',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm lw lmt 15.9 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LH1',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 1 sgm hg lmt 84.1 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_ML2',header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_MH2',header_comment = '1DGF Ar2Lum Lum M SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LL2',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm lw lmt 2.30 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LH2',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 2 sgm hg lmt 97.7 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_ML3',header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_MH3',header_comment = '1DGF Ar2Lum Lum M SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LL3',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm lw lmt 0.20 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FMS_LH3',header_comment = '1DGF Ar2Lum Lum M [log] SUM Err 3 sgm hg lmt 99.8 pct')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_CCT',header_comment = 'Channel # Central SUM (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_1SG',header_comment = 'Channel Dif Limit 1SG SUM (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_2SG',header_comment = 'Channel Dif Limit 2SG SUM (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_3SG',header_comment = 'Channel Dif Limit 3SG SUM (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_4SG',header_comment = 'Channel Dif Limit 4SG SUM (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_5SG',header_comment = 'Channel Dif Limit 5SG SUM (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_1FW',header_comment = 'Channel Dif Limit 1FW SUM (PyId)')
		Header_Copy(Cube2bclp_2D_opt,Cube2bFit,'FTS_2FW',header_comment = 'Channel Dif Limit 2FW SUM (PyId)')
	except KeyError:
		pass

	print
	print (colored('Input Cube for collapse : ' + Cube2bFit,'magenta'))
	print (colored('Resulting Collapsed Cube: ' + Cube2bclp_2D_opt,'yellow'))

	nx_f2DG, ny_f2DG = data_2b_plot.shape
	nx,ny            = nx_f2DG,ny_f2DG

	X0_f2DG     = kwargs.get('X0_f2DG',int(np.ceil(nx_f2DG/2)))
	Y0_f2DG     = kwargs.get('Y0_f2DG',int(np.ceil(ny_f2DG/2)))
	A_f2DG      = kwargs.get('A_f2DG',1)
	SIGMAX_f2DG = kwargs.get('SIGMAX_f2DG',1)
	SIGMAY_f2DG = kwargs.get('SIGMAY_f2DG',1)
	THETA_f2DG  = kwargs.get('THETA_f2DG',0)
	OFS_f2DG    = kwargs.get('OFS_f2DG',0)
	displ_s_f   = kwargs.get('displ_s_f',False)
	verbose     = kwargs.get('verbose',False)

	# Create x and y indices
	x    = np.linspace(0, nx_f2DG, nx_f2DG)-0.5
	y    = np.linspace(0, ny_f2DG, ny_f2DG)-0.5
	x, y = np.meshgrid(x, y)

	data = data_2b_plot

	#############################################2D-Fit#############################################
	initial_guess = (X0_f2DG,Y0_f2DG,A_f2DG,SIGMAX_f2DG,SIGMAY_f2DG,THETA_f2DG,OFS_f2DG)

	xdata       = np.vstack((x.ravel(),y.ravel()))
	ydata       = data.ravel()

	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_TYP',sgm_wgth_tms,header_comment = 'Criteria used for Collapse ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CNN',tlt_ch_nmb  ,header_comment = 'Number Chns used for Collapse ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CNI',slice_nblw  ,header_comment = 'Initial Chn used for Collapse Num ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CNL',slice_nbhg  ,header_comment = 'Last Chn used for Collapse Num ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CVI',slice_vllw  ,header_comment = 'Initial Chn used for Collapse Vel ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + 'CL_CVL',slice_vlhg  ,header_comment = 'Last Chn used for Collapse Vel ' + clp_hdc)

	if verbose == True:
		print
		print (colored(Message1,'yellow'))
		print (colored(Message2,'yellow'))
		print
		print (colored('Generated Plot: ' + str(PLOTFILENAME) ,'cyan'))
		print
	elif verbose == False:
		pass
		
	if displ_s_f == True:
		fxsize=9
		fysize=8
		f = plt.figure(num=None, figsize=(fxsize, fysize), dpi=180, facecolor='w',
			edgecolor='k')
		plt.subplots_adjust(
			left 	= (25/25.4)/fxsize,    
			bottom 	= (16/25.4)/fysize,    
			right 	= 1 - (15/25.4)/fxsize,
			top 	= 1 - (15/25.4)/fysize)
		plt.subplots_adjust(hspace=0)

		gs0 = gridspec.GridSpec(1, 1)
		gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
		ax110 = plt.Subplot(f, gs11[0,0])
		f.add_subplot(ax110)

		ax110.set_rasterization_zorder(1)
		plt.autoscale(enable=True, axis='y', tight=False)

		plt.title(plt_tlt + ' (' +  str(x_ref) + ','+str(y_ref)+')',family='serif')
		plt.xlabel('X',fontsize=20,family = 'serif')
		plt.ylabel('Y',fontsize=20,family = 'serif')
		plt.tick_params(which='both', width=1.0)
		plt.tick_params(which='major', length=10)
		plt.tick_params(which='minor', length=5)
		ax110.minorticks_on()

		if ('_ms.' in Cube2bFit) or ('dta_in.' in Cube2bFit) or ('dta_ot.' in Cube2bFit):
			tick_color = 'white'
		elif ('msk_in.' in Cube2bFit) or ('crc.' in Cube2bFit) or ('msk_ot.' in Cube2bFit):
			tick_color = 'black'
		else:
			tick_color = 'white'

		ax110.xaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)
		ax110.yaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)

		plt.imshow(ydata.reshape(nx, ny), cmap=plt.cm.viridis, origin='lower',
		    extent=(x.min(), x.max(), y.min(), y.max()))
		divider = make_axes_locatable(ax110)
		cax  = divider.append_axes("right", size="5%", pad=0.05)	
		cbar = plt.colorbar(cax=cax)
		cbar.set_label('S [Jy]', rotation=270,family = 'serif')

		min_y, max_y = ax110.get_ylim()
		min_x, max_x = ax110.get_xlim()	

		plt.text(0,max_y-(max_y/10),
				'1 pix = '  + str(scale_arcsec) + ' arcsec',  
				ha='left' , va='baseline',color='white',fontsize=20,
				family = 'serif')

		X0,Y0 = x_ref,y_ref

		plt.scatter(X0_f2DG + 0.0, Y0_f2DG + 0.0, s=25, c='black', marker='+')

		ax110.xaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-X0_f2DG+x_ref )) #X_DIF
		ax110.yaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-Y0_f2DG+y_ref )) #Y_DIF

		plt.savefig(PLOTFILENAME)
	elif displ_s_f == False:
		pass

def Cube_fit_1D_Gaussian(Cube2bPlot_1D,*args, **kwargs):
	autoaxis  = kwargs.get('autoaxis',True)
	verbose   = kwargs.get('verbose' , False)
	epssave   = kwargs.get('epssave' , False)
	showplot  = kwargs.get('showplot', False) 
	amplitude = kwargs.get('amplitude',1)
	mean      = kwargs.get('mean',0)
	stddev    = kwargs.get('stddev',1)
	cubewdthv = kwargs.get('cubewdthv',1)

	z_avg     = kwargs.get('z_avg',Header_Get(Cube2bPlot_1D,'STZ_AVG'))
	z_med     = kwargs.get('z_med',Header_Get(Cube2bPlot_1D,'STZ_MED'))
	frq_r     = kwargs.get('frq_r',restframe_frequency)
	z_f2l     = z_med

	fit_type  = kwargs.get('fit_type','scipy')

	Cube2bPlot_1D_Err  = kwargs.get('Cube2bPlot_1D_Err', None)
	fit_max_1d         = kwargs.get('fit_max_1d',False)
	dest_dir_plt       = kwargs.get('dest_dir_plt',None)

	redshift_inf_1 = Header_Get(Cube2bPlot_1D,'STZ_1SL')
	redshift_sup_1 = Header_Get(Cube2bPlot_1D,'STZ_1SH')
	redshift_inf_2 = Header_Get(Cube2bPlot_1D,'STZ_2SL')
	redshift_sup_2 = Header_Get(Cube2bPlot_1D,'STZ_2SH')
	redshift_inf_3 = Header_Get(Cube2bPlot_1D,'STZ_3SL')
	redshift_sup_3 = Header_Get(Cube2bPlot_1D,'STZ_3SH')

	twn_axs_scl    = kwargs.get('twn_axs_scl',1.5)

	FLUX2bPlot = Cube_Stat_Slice(Cube2bPlot_1D,cubewdthv=cubewdthv,frq_r=frq_r)
	if dest_dir_plt != None:
		PLOTFILENAME = str(dest_dir_plt)  + '/' + (str(Cube2bPlot_1D).split('.fits')[0]).split('/')[-1] + '-1DGF.pdf'
	elif dest_dir_plt == None:
		PLOTFILENAME = ana_dir_plt    + '/' + (str(Cube2bPlot_1D).split('.fits')[0]).split('/')[-1] + '-1DGF.pdf'
	
	Cube_Info = Cube_Header_Get(Cube2bPlot_1D,frq_r* u.Hz)
	FRQ_AXS   = Cube_Info[16].value
	VEL_AXS   = Cube_Info[17].value
	FLX_SUM   = FLUX2bPlot[0]
	FLX_AVG   = FLUX2bPlot[1]
	FLX_MED   = FLUX2bPlot[2]
	FLX_STD   = FLUX2bPlot[3]
	FLX_TOT   = FLUX2bPlot[4]
	FLX_MXC   = FLUX2bPlot[5]
	FLX_MXR   = FLUX2bPlot[6]

	XAXIS     = VEL_AXS

	label_SUM = 'SUM $\Delta$'
	label_AVG = 'AVG $\star$'
	label_MED = 'MED $\circ$'
	label_STD = 'STD'
	label_MXC = 'MXC $\\blacktriangleleft$'
	label_MXR = 'MXR $\\blacktriangleright$'

	print
	print (colored('Fits file for fitting: ','yellow'))
	print (colored(str(Cube2bPlot_1D)       ,'cyan'))
	print

	fxsize=11
	fysize=8
	f = plt.figure(num=None, figsize=(11, 8), dpi=180, facecolor='w',
		edgecolor='k')
	plt.subplots_adjust(
		left 	= (38/25.4)/fxsize,    
		bottom 	= (14/25.4)/fysize, 
		right 	= 1 - (22/25.4)/fxsize,
		top 	= 1 - (15/25.4)/fysize)
	plt.subplots_adjust(hspace=0)


	gs0 = gridspec.GridSpec(1, 1)
	##########################################SPEC-1###################################

	gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
		
	ax110 = plt.Subplot(f, gs11[0,0])
	f.add_subplot(ax110)

	ax110.set_rasterization_zorder(1)
	plt.autoscale(enable=True, axis='both', tight=False)
	ax110.xaxis.set_tick_params(which='both',labelsize=16,direction='in',color='black',bottom=True,top=True,left=True,right=True)
	ax110.yaxis.set_tick_params(which='both',labelsize=16,direction='in',color='black',bottom=True,top=True,left=True,right=True)
	ax110.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	xticklabels = ax110.get_xticklabels()
	plt.setp(xticklabels, visible=True,family='serif')
	yticklabels = ax110.get_yticklabels()
	plt.setp(yticklabels, visible=True,family='serif')

	minorLocator_x   = plt.MultipleLocator((XAXIS[1]-XAXIS[0])/2)
	majorLocator_x   = plt.MultipleLocator(XAXIS[1]-XAXIS[0])
	plt.tick_params(which='both', width=1.0)
	plt.tick_params(which='major', length=10)
	plt.tick_params(which='minor', length=5)
	ax110.minorticks_on()

	plt.xlabel('$v$ kms$^{-1}$'  ,fontsize=16,family = 'serif')
	plt.ylabel('S [Jy]',fontsize=16,family = 'serif')

	if Cube2bPlot_1D_Err == None:
		plt.scatter(XAXIS, FLX_AVG, color = 'r'   , marker = '*', alpha = 0.4)#label = label_AVG
		plt.scatter(XAXIS, FLX_MED, color = 'b'   , marker = 'o', alpha = 0.4)#label = label_MED
	elif Cube2bPlot_1D_Err != None:
		print
		print (colored('Using outer region cubes as std error for chi2 computations!','yellow'))
		print (colored(str(Cube2bPlot_1D_Err),'cyan'))
		print
		FLUX2bPlotErr = Cube_Stat_Slice(Cube2bPlot_1D_Err,cubewdthv=cubewdthv,frq_r=frq_r,verbose=False)#[3]
		FLX_ERR_AVG       = FLUX2bPlotErr[1]
		FLX_ERR_STD       = FLUX2bPlotErr[3]
		FLX_ERR_RMS       = np.sqrt(FLX_ERR_AVG**2 + FLX_ERR_STD**2)
		#plt.errorbar(XAXIS, FLX_AVG, yerr=FLX_ERR_STD ,color = 'r'   , label = label_AVG, marker = '*', alpha = 0.4, ls = 'none')
		#plt.errorbar(XAXIS, FLX_MED, yerr=FLX_ERR_STD ,color = 'b'   , label = label_MED, marker = 'o', alpha = 0.4, ls = 'none')
		plt.scatter(XAXIS, FLX_AVG, color = 'r'   , marker = '*', alpha = 0.4)#label = label_AVG
		plt.scatter(XAXIS, FLX_MED, color = 'b'   , marker = 'o', alpha = 0.4)#label = label_MED

	slc_nmb       = kwargs.get('slc_nmb',(np.ceil(len(FLX_AVG)/2)))
	max_rng_val   = kwargs.get('max_rng_val',(np.ceil(len(FLX_AVG)/10.0)))
	max_rng       = kwargs.get('max_rng',False)

	if max_rng == True and slc_nmb > 2:
		indx_in = int(slc_nmb - max_rng_val)-1
		indx_fn = int(slc_nmb + max_rng_val)+1
		indx_FLX_SUM = np.where(FLX_SUM == max(FLX_SUM[indx_in:indx_fn]))[0][0]
		indx_FLX_AVG = np.where(FLX_AVG == max(FLX_AVG[indx_in:indx_fn]))[0][0]
		indx_FLX_MED = np.where(FLX_MED == max(FLX_MED[indx_in:indx_fn]))[0][0]
		indx_FLX_MXC = np.where(FLX_MXC == max(FLX_MXC[indx_in:indx_fn]))[0][0]
		indx_FLX_MXR = np.where(FLX_MXR == max(FLX_MXR[indx_in:indx_fn]))[0][0]
		plt.text(VEL_AXS[indx_FLX_AVG], max(FLX_AVG[indx_in:indx_fn]),str(indx_FLX_AVG+1) + ' ' + str(round(max(FLX_AVG),6)), ha='left' , va='bottom',color='red' ,family = 'serif')
		plt.text(VEL_AXS[indx_FLX_MED], max(FLX_MED[indx_in:indx_fn]),str(indx_FLX_MED+1) + ' ' + str(round(max(FLX_MED),6)), ha='right', va='top'   ,color='blue',family = 'serif')
	elif max_rng == False or (max_rng == True and slc_nmb <= 2):
		indx_FLX_SUM = np.where(FLX_SUM == max(FLX_SUM))[0][0]
		indx_FLX_AVG = np.where(FLX_AVG == max(FLX_AVG))[0][0]
		indx_FLX_MED = np.where(FLX_MED == max(FLX_MED))[0][0]
		indx_FLX_MXC = np.where(FLX_MXC == max(FLX_MXC))[0][0]
		indx_FLX_MXR = np.where(FLX_MXR == max(FLX_MXR))[0][0]
		plt.text(VEL_AXS[indx_FLX_AVG], max(FLX_AVG),str(indx_FLX_AVG+1) + ' ' + str(round(max(FLX_AVG),6)), ha='left' , va='bottom',color='red' ,family = 'serif')
		plt.text(VEL_AXS[indx_FLX_MED], max(FLX_MED),str(indx_FLX_MED+1) + ' ' + str(round(max(FLX_MED),6)), ha='right', va='top'   ,color='blue',family = 'serif')

	mean_avg      = VEL_AXS[int(indx_FLX_AVG)]
	amplitude_avg = FLX_AVG[indx_FLX_AVG]

	mean_med      = VEL_AXS[int(indx_FLX_MED)]
	amplitude_med = FLX_MED[indx_FLX_MED]

	mean_sum      = VEL_AXS[int(indx_FLX_SUM)]
	amplitude_sum = FLX_SUM[indx_FLX_SUM]

	mean_mxc      = VEL_AXS[int(indx_FLX_MXC)]
	amplitude_mxc = FLX_SUM[indx_FLX_MXC]

	mean_mxr      = VEL_AXS[int(indx_FLX_MXR)]
	amplitude_mxr = FLX_SUM[indx_FLX_MXR]

	Header_Add(Cube2bPlot_1D,'MAX_SNS',int(indx_FLX_SUM)          ,header_comment='Vel Prf Max Chn Location SUM')
	Header_Add(Cube2bPlot_1D,'MAX_SVS',VEL_AXS[int(indx_FLX_SUM)] ,header_comment='Vel Prf Max Vel Location [km/s] SUM')
	Header_Add(Cube2bPlot_1D,'MAX_SVS',FRQ_AXS[int(indx_FLX_SUM)] ,header_comment='Vel Prf Max Frequency Location [Hz] SUM')
	Header_Add(Cube2bPlot_1D,'MAX_VLS',max(FLX_SUM)               ,header_comment='Vel Prf Max Chn Value SUM')
	Header_Add(Cube2bPlot_1D,'MAX_SNA',int(indx_FLX_AVG)          ,header_comment='Vel Prf Max Chn Location AVG')
	Header_Add(Cube2bPlot_1D,'MAX_SVA',VEL_AXS[int(indx_FLX_AVG)] ,header_comment='Vel Prf Max Vel Location [km/s] AVG')
	Header_Add(Cube2bPlot_1D,'MAX_SVA',FRQ_AXS[int(indx_FLX_AVG)] ,header_comment='Vel Prf Max Frequency Location [Hz] AVG')
	Header_Add(Cube2bPlot_1D,'MAX_VLA',max(FLX_AVG)               ,header_comment='Vel Prf Max Chn Value AVG')
	Header_Add(Cube2bPlot_1D,'MAX_SNM',int(indx_FLX_MED)          ,header_comment='Vel Prf Max Chn Location MED')
	Header_Add(Cube2bPlot_1D,'MAX_SVM',VEL_AXS[int(indx_FLX_MED)] ,header_comment='Vel Prf Max Vel Location [km/s] MED')
	Header_Add(Cube2bPlot_1D,'MAX_SVM',FRQ_AXS[int(indx_FLX_MED)] ,header_comment='Vel Prf Max Frequency Location [Hz] MED')
	Header_Add(Cube2bPlot_1D,'MAX_VLM',max(FLX_MED)               ,header_comment='Vel Prf Max Chn Value MED')

	Header_Add(Cube2bPlot_1D,'MAX_SMC',int(indx_FLX_MXC)          ,header_comment='Vel Prf Max Chn Location MXC')
	Header_Add(Cube2bPlot_1D,'MAX_SMC',VEL_AXS[int(indx_FLX_MXC)] ,header_comment='Vel Prf Max Vel Location [km/s] MXC')
	Header_Add(Cube2bPlot_1D,'MAX_SMC',FRQ_AXS[int(indx_FLX_MXC)] ,header_comment='Vel Prf Max Frequency Location [Hz] MXC')
	Header_Add(Cube2bPlot_1D,'MAX_VMC',max(FLX_MXC)               ,header_comment='Vel Prf Max Chn Value MXC')

	Header_Add(Cube2bPlot_1D,'MAX_SMR',int(indx_FLX_MXR)          ,header_comment='Vel Prf Max Chn Location MXR')
	Header_Add(Cube2bPlot_1D,'MAX_SMR',VEL_AXS[int(indx_FLX_MXR)] ,header_comment='Vel Prf Max Vel Location [km/s] MXR')
	Header_Add(Cube2bPlot_1D,'MAX_SMR',FRQ_AXS[int(indx_FLX_MXR)] ,header_comment='Vel Prf Max Frequency Location [Hz] MXR')
	Header_Add(Cube2bPlot_1D,'MAX_VMR',max(FLX_MXR)               ,header_comment='Vel Prf Max Chn Value MXR')

	##############################################################FIT-AVG##############################################################
	XAXIS_FIT  = np.arange(XAXIS[0],XAXIS[-1],0.01)
	std_wght   = Header_Get(Cube2bPlot_1D,'STT_VEL')
	try:
		if fit_type == 'astropy':
			print
			print (colored('1D Gaussian Fit Mode Choosen: Astropy (No-Offset)','yellow'))
			print
			#astropy FIT#
			#http://docs.astropy.org/en/stable/api/astropy.modeling.fitting.LevMarLSQFitter.html#astropy.modeling.fitting.LevMarLSQFitter
			g_wght_avg   = func_1D_Gaussian(XAXIS,mean_avg,amplitude_avg,std_wght)
			g_init_avg   = apmd.models.Gaussian1D(amplitude=amplitude_avg, mean=mean_avg, stddev=stddev)
			g_init_avg.amplitude.fixed = True
			fit_g_avg    = apmd.fitting.LevMarLSQFitter()
			g_avg        = fit_g_avg(g_init_avg, XAXIS, FLX_AVG,weights=g_wght_avg)
			g_avg_cov    = fit_g_avg.fit_info['param_cov']
			
			if (g_avg_cov is None):
				g_avg_var    = np.zeros((1,2))
				g_avg_var[:] = np.nan
				g_avg_var    = np.squeeze((g_avg_var))
			elif np.linalg.det(g_avg_cov) < 0:
				g_avg_var    = np.zeros_like(np.diag(g_avg_cov))
				g_avg_var[:] = np.nan
				g_avg_var    = np.squeeze((g_avg_var))
			else:
				g_avg_var = np.sqrt(np.diag(g_avg_cov))
				g_avg_var = np.squeeze((g_avg_var))
			
			#DEFINE ASTROPY FIT VALUES#
			g_avg_amplitude  = g_avg.amplitude[0]
			g_avg_mean       = g_avg.mean[0]
			g_avg_stddev     = abs(g_avg.stddev[0])
			
			g_avg_var_mea    = g_avg_var[0]
			g_avg_var_std    = g_avg_var[-1]
			g_avg_plt = g_avg(XAXIS_FIT)
			g_exp_avg = func_1D_Gaussian(XAXIS,g_avg_mean,g_avg_amplitude,g_avg_stddev)
			g_avg_are = g_avg
			#DEFINE ASTROPY FIT VALUES#		
			#astropy FIT#

		elif fit_type=='scipy':
			print
			print (colored('1D Gaussian Fit Mode Choosen: Scipy (Offset)','yellow'))
			print
			##scipy FIT##
			g_wght_avg_O  = func_1D_Gaussian_O(XAXIS,mean_avg,amplitude_avg,std_wght,min(FLX_AVG))
			initial_guess = (mean_avg,amplitude_avg,stddev,min(FLX_AVG))

			popt, pcov  = scpopt.curve_fit(func_1D_Gaussian_O, XAXIS, FLX_AVG, 
						p0=initial_guess,
						bounds=([mean_avg-0.001,(amplitude_avg)-(abs(amplitude_avg)*.01),-np.inf,-np.inf],
								[mean_avg+0.001,(amplitude_avg)+(abs(amplitude_avg)*.01), np.inf, np.inf])
						)
			perr        = np.sqrt(np.diag(pcov))
			data_fitted = func_1D_Gaussian_O(XAXIS, *popt)
			fit_res     = 'OK'

			#DEFINE SCIPY FIT VALUES#			
			g_avg_mean       = np.round(popt[0],10)
			g_avg_amplitude  = np.round(popt[1],10)
			g_avg_stddev     = abs(np.round(popt[2],10))
			g_avg_offset     = np.round(popt[3],10)

			g_avg_var_mea          = np.round(perr[0],10)
			g_avg_var_amplitude    = np.round(perr[1],10)
			g_avg_var_std          = np.round(perr[2],10)
			g_avg_var_offset       = np.round(perr[3],10)

			g_avg_plt = func_1D_Gaussian_O(XAXIS_FIT, *popt)
			g_exp_avg = func_1D_Gaussian_O(XAXIS,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)
			g_avg_are = g_avg_plt
			#DEFINE SCIPY FIT VALUES#
			##scipy FIT##

		elif fit_type=='lmfit':
			import lmfit
			print
			print (colored('1D Gaussian Fit Mode Choosen: Lmfit (Offset)','yellow'))
			print
			g_wght_avg_O  = func_1D_Gaussian_O(XAXIS,mean_avg,amplitude_avg,std_wght,min(FLX_AVG))
			mod     = lmfit.models.ExpressionModel('Offset+(Amplitude*exp(-(((x-X_0)**2)/(2*(Sigma**2)))))')
			params  = mod.make_params(Offset=min(FLX_AVG), Amplitude=amplitude_avg, X_0=mean_avg, Sigma=stddev)

			if min(FLX_AVG) == 0:
				min_FLX_AVG = 0.001
			else:
				min_FLX_AVG = min(FLX_AVG)
				pass
			if amplitude_avg == 0:
				amplitude_avg = 0.001
			else:
				pass
			params['Offset'].min = min_FLX_AVG - (abs(min_FLX_AVG)*.01)
			params['Offset'].max = min_FLX_AVG + (abs(min_FLX_AVG)*.01)
			params['Amplitude'].min = (amplitude_avg)-(abs(amplitude_avg)*.01)
			params['Amplitude'].max = (amplitude_avg)+(abs(amplitude_avg)*.01)
			params['X_0'].min = mean_avg-0.001
			params['X_0'].max = mean_avg+0.001

			FLX_AVG = FLX_AVG.astype(float)
			XAXIS   = XAXIS.astype(float)
			result  = mod.fit(FLX_AVG, params, x=XAXIS, weights=g_wght_avg_O)

			g_avg_mean             = result.params['X_0'].value
			g_avg_amplitude        = result.params['Amplitude'].value
			g_avg_stddev           = abs(result.params['Sigma'].value)
			g_avg_offset           = result.params['Offset'].value

			g_avg_var_mea          = result.params['X_0'].stderr
			g_avg_var_amplitude    = result.params['Amplitude'].stderr
			g_avg_var_std          = result.params['Sigma'].stderr
			g_avg_var_offset       = result.params['Offset'].stderr

			g_avg_cor_mea          = result.params['X_0'].correl
			g_avg_cor_amplitude    = result.params['Amplitude'].correl
			g_avg_cor_std          = result.params['Sigma'].correl
			g_avg_cor_offset       = result.params['Offset'].correl

			g_avg_chisqr = result.chisqr
			g_avg_redchi = result.redchi

			g_avg_plt = func_1D_Gaussian_O(XAXIS_FIT, g_avg_mean, g_avg_amplitude, g_avg_stddev, g_avg_offset )
			g_exp_avg = func_1D_Gaussian_O(XAXIS,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)
			g_avg_are = g_avg_plt

			#print(result.fit_report())   

		Area_avg         = scpint.quad(lambda x: g_avg_amplitude*np.exp(-((x-g_avg_mean)**2)/(2*g_avg_stddev**2)), -np.inf, np.inf)
		Area_avg_man     = g_avg_stddev  * amplitude_avg * np.sqrt(2*np.pi)
		Area_avg_man_err = g_avg_var_std * amplitude_avg * np.sqrt(2*np.pi)

		Lum_Area_avg     = FluxToLum(Area_avg[0],z_f2l,frq_r)
		lum_area_err_1_a = Luminosity_Error(Area_avg[0],redshift_inf_1,redshift_sup_1,Area_avg[1],frq_r=frq_r)
		lum_area_err_2_a = Luminosity_Error(Area_avg[0],redshift_inf_2,redshift_sup_2,Area_avg[1],frq_r=frq_r)
		lum_area_err_3_a = Luminosity_Error(Area_avg[0],redshift_inf_3,redshift_sup_3,Area_avg[1],frq_r=frq_r)

		lum_area_err_1_m = Luminosity_Error(Area_avg[0],redshift_inf_1,redshift_sup_1,Area_avg_man_err,frq_r=frq_r)
		lum_area_err_2_m = Luminosity_Error(Area_avg[0],redshift_inf_2,redshift_sup_2,Area_avg_man_err,frq_r=frq_r)
		lum_area_err_3_m = Luminosity_Error(Area_avg[0],redshift_inf_3,redshift_sup_3,Area_avg_man_err,frq_r=frq_r)

		g_obs_avg = FLX_AVG

		chi2_avg      = sum((g_obs_avg-g_exp_avg)**2/(FLX_ERR_STD**2))
		chi2_red_avg  = chi2_avg/float(len(XAXIS)-1)
		
		plt.plot(XAXIS_FIT, g_avg_plt, color='red',ls=':',alpha=0.4,label='')

		x_text_coor = min(XAXIS_FIT)
		y_text_coor = ax110.get_yticks(minor=True)[-1]
		step = abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))*.4

		plt.text(x_text_coor,y_text_coor-step*0,
				'Gaussian fit :'+label_AVG,
				ha='left' , va='bottom',color='red',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*1,
				'A : '        + str(np.round(g_avg_amplitude,10))        ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*2,
				'$\mu$ :'     + str(np.round(g_avg_mean,3))                   + ' $\pm$ ' + str(np.round(g_avg_var_mea,9))       ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*3,
				'$\sigma$ : ' + str(np.round(g_avg_stddev,3))                 + ' $\pm$ ' + str(np.round(g_avg_var_std,9))      ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*4,
				'fwhm : '     + str(np.round(linewidth_fwhm(g_avg_stddev),3)) + ' $\pm$ ' + str(np.round(g_avg_var_std*linewidth_fwhm(1),9)),
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*5,
				'S : '        + str(np.round(Area_avg_man,3))                 + ' $\pm$ ' + str(np.round(Area_avg_man_err,9))    ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')	
		plt.text(x_text_coor,y_text_coor-step*6,
				'L : '        + str(np.round(Lum_Area_avg[0],3))  + '-' + str(np.round(lum_area_err_1_m[0],3)) + '+' + str(str(np.round(lum_area_err_1_m[1],9))) ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*7,
				'log(L) : '   + str(np.round(Lum_Area_avg[1],3))  + '-' + str(np.round(lum_area_err_1_m[2],3)) + '+' + str(str(np.round(lum_area_err_1_m[3],9))),
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')

		k=2.355
		j=2*k
		h=max(g_avg_plt)*.05

		if fit_type == 'astropy':
			plt.plot([-0*g_avg_stddev, -0*g_avg_stddev], [0, func_1D_Gaussian(-0*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='k'     , lw=1.5, alpha=0.9, ls=':')
			plt.plot([-1*g_avg_stddev, -1*g_avg_stddev], [0, func_1D_Gaussian(-1*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='red'   , lw=1.5, alpha=0.4, ls=':')
			plt.plot([+1*g_avg_stddev, +1*g_avg_stddev], [0, func_1D_Gaussian(+1*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='red'   , lw=1.5, alpha=0.4, ls=':')

			plt.plot([-k*g_avg_stddev, -k*g_avg_stddev], [0, func_1D_Gaussian(-k*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='k'     , lw=1.5, alpha=0.9, ls='-.')
			plt.plot([+k*g_avg_stddev, +k*g_avg_stddev], [0, func_1D_Gaussian(+k*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='k'     , lw=1.5, alpha=0.9, ls='-.')
			plt.plot([-j*g_avg_stddev, -j*g_avg_stddev], [0, func_1D_Gaussian(-j*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)+h], color='k'     , lw=1.5, alpha=0.9, ls='-.')
			plt.plot([+j*g_avg_stddev, +j*g_avg_stddev], [0, func_1D_Gaussian(+j*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)+h], color='k'     , lw=1.5, alpha=0.9, ls='-.')

			plt.plot([-2*g_avg_stddev, -2*g_avg_stddev], [0, func_1D_Gaussian(-2*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='blue'  , lw=1.5, alpha=0.4, ls=':')
			plt.plot([+2*g_avg_stddev, +2*g_avg_stddev], [0, func_1D_Gaussian(+2*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='blue'  , lw=1.5, alpha=0.4, ls=':')
			plt.plot([-3*g_avg_stddev, -3*g_avg_stddev], [0, func_1D_Gaussian(-3*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='orange', lw=1.5, alpha=0.4, ls=':')
			plt.plot([+3*g_avg_stddev, +3*g_avg_stddev], [0, func_1D_Gaussian(+3*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)], color='orange', lw=1.5, alpha=0.4, ls=':')
			plt.plot([-4*g_avg_stddev, -4*g_avg_stddev], [0, func_1D_Gaussian(-4*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)+h], color='yellow', lw=1.5, alpha=0.4, ls=':')
			plt.plot([+4*g_avg_stddev, +4*g_avg_stddev], [0, func_1D_Gaussian(+4*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)+h], color='yellow', lw=1.5, alpha=0.4, ls=':')
			plt.plot([-5*g_avg_stddev, -5*g_avg_stddev], [0, func_1D_Gaussian(-5*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)+h], color='gray'  , lw=1.5, alpha=0.4, ls=':')
			plt.plot([+5*g_avg_stddev, +5*g_avg_stddev], [0, func_1D_Gaussian(+5*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev)+h], color='gray'  , lw=1.5, alpha=0.4, ls=':')

			#Upper shadow pct 83, 97, 99
			mask  = (XAXIS_FIT >= -1*g_avg_stddev) & (XAXIS_FIT <= +1*g_avg_stddev)
			a1sg  = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='red', facecolor='red',
					linewidth=0, linestyle='solid', antialiased=True, label = 'Area 1$\sigma$')
			mask  = (XAXIS_FIT>1*g_avg_stddev) & (XAXIS_FIT <= +2*g_avg_stddev)	
			a2sgl = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='blue', facecolor='blue',
					linewidth=0, linestyle='solid', antialiased=True)
			mask  = (XAXIS_FIT>-2*g_avg_stddev) & (XAXIS_FIT <= -1*g_avg_stddev)	
			a2sgh = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='blue', facecolor='blue',
					linewidth=0, linestyle='solid', antialiased=True,label = 'Area 2$\sigma$')

			mask  = (XAXIS_FIT>2*g_avg_stddev) & (XAXIS_FIT <= +3*g_avg_stddev)	
			a3sgl = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='orange', facecolor='orange',
					linewidth=0, linestyle='solid', antialiased=True)
			mask  = (XAXIS_FIT>-3*g_avg_stddev) & (XAXIS_FIT <= -2*g_avg_stddev)	
			a3sgh = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='orange', facecolor='orange',
					linewidth=0, linestyle='solid', antialiased=True,label = 'Area 3$\sigma$')

			mask  = (XAXIS_FIT>3*g_avg_stddev) & (XAXIS_FIT <= +4*g_avg_stddev)	
			a4sgl = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='yellow', facecolor='yellow',
					linewidth=0, linestyle='solid', antialiased=True)
			mask  = (XAXIS_FIT>-4*g_avg_stddev) & (XAXIS_FIT <= -3*g_avg_stddev)	
			a4sgh = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='yellow', facecolor='yellow',
					linewidth=0, linestyle='solid', antialiased=True,label = 'Area 4$\sigma$')

			mask  = (XAXIS_FIT>4*g_avg_stddev) & (XAXIS_FIT <= +5*g_avg_stddev)	
			a5sgl = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='gray', facecolor='gray',
					linewidth=0, linestyle='solid', antialiased=True)
			mask  = (XAXIS_FIT>-5*g_avg_stddev) & (XAXIS_FIT <= -4*g_avg_stddev)	
			a5sgh = plt.fill_between(XAXIS_FIT[mask], 0, g_avg_are(XAXIS_FIT[mask]),
					alpha=0.5, edgecolor='gray', facecolor='gray',
					linewidth=0, linestyle='solid', antialiased=True, label = 'Area 5$\sigma$')
			l_a1sg = plt.legend([a1sg] ,['Area 1$\sigma$'],loc=3)
			l_a2sg = plt.legend([a2sgh],['Area 2$\sigma$'],loc=3)
			l_a3sg = plt.legend([a3sgh],['Area 3$\sigma$'],loc=3)
			l_a4sg = plt.legend([a4sgh],['Area 4$\sigma$'],loc=3)
			l_a5sg = plt.legend([a5sgh],['Area 5$\sigma$'],loc=3)
		elif fit_type=='scipy' or fit_type=='lmfit':
			plt.plot([-0*g_avg_stddev, -0*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(-0*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='k'     , lw=1.5, alpha=0.9, ls=':')
			plt.plot([-1*g_avg_stddev, -1*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(-1*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='red'   , lw=1.5, alpha=0.4, ls=':')
			plt.plot([+1*g_avg_stddev, +1*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(+1*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='red'   , lw=1.5, alpha=0.4, ls=':')

			plt.plot([-k*g_avg_stddev, -k*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(-k*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='k'     , lw=1.5, alpha=0.9, ls='-.')
			plt.plot([+k*g_avg_stddev, +k*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(+k*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='k'     , lw=1.5, alpha=0.9, ls='-.')
			plt.plot([-j*g_avg_stddev, -j*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(-j*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev+h,g_avg_offset)], color='k'     , lw=1.5, alpha=0.9, ls='-.')
			plt.plot([+j*g_avg_stddev, +j*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(+j*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev+h,g_avg_offset)], color='k'     , lw=1.5, alpha=0.9, ls='-.')

			plt.plot([-2*g_avg_stddev, -2*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(-2*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='blue'  , lw=1.5, alpha=0.4, ls=':')
			plt.plot([+2*g_avg_stddev, +2*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(+2*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='blue'  , lw=1.5, alpha=0.4, ls=':')
			plt.plot([-3*g_avg_stddev, -3*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(-3*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='orange', lw=1.5, alpha=0.4, ls=':')
			plt.plot([+3*g_avg_stddev, +3*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(+3*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset)]  , color='orange', lw=1.5, alpha=0.4, ls=':')
			plt.plot([-4*g_avg_stddev, -4*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(-4*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev+h,g_avg_offset)], color='yellow', lw=1.5, alpha=0.4, ls=':')
			plt.plot([+4*g_avg_stddev, +4*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(+4*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev+h,g_avg_offset)], color='yellow', lw=1.5, alpha=0.4, ls=':')
			plt.plot([-5*g_avg_stddev, -5*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(-5*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev+h,g_avg_offset)], color='gray'  , lw=1.5, alpha=0.4, ls=':')
			plt.plot([+5*g_avg_stddev, +5*g_avg_stddev], [g_avg_offset, func_1D_Gaussian_O(+5*g_avg_stddev,g_avg_mean,g_avg_amplitude,g_avg_stddev+h,g_avg_offset)], color='gray'  , lw=1.5, alpha=0.4, ls=':')

			#Upper shadow pct 83, 97, 99
			mask  = (XAXIS_FIT >= -1*g_avg_stddev) & (XAXIS_FIT <= +1*g_avg_stddev)
			a1sg  = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='red', facecolor='red',
					linewidth=0, linestyle='solid', antialiased=True, label = 'Area 1$\sigma$')
			mask  = (XAXIS_FIT>1*g_avg_stddev) & (XAXIS_FIT <= +2*g_avg_stddev)	
			a2sgl = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='blue', facecolor='blue',
					linewidth=0, linestyle='solid', antialiased=True)
			mask  = (XAXIS_FIT>-2*g_avg_stddev) & (XAXIS_FIT <= -1*g_avg_stddev)	
			a2sgh = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='blue', facecolor='blue',
					linewidth=0, linestyle='solid', antialiased=True,label = 'Area 2$\sigma$')

			mask  = (XAXIS_FIT>2*g_avg_stddev) & (XAXIS_FIT <= +3*g_avg_stddev)	
			a3sgl = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='orange', facecolor='orange',
					linewidth=0, linestyle='solid', antialiased=True)
			mask  = (XAXIS_FIT>-3*g_avg_stddev) & (XAXIS_FIT <= -2*g_avg_stddev)	
			a3sgh = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='orange', facecolor='orange',
					linewidth=0, linestyle='solid', antialiased=True,label = 'Area 3$\sigma$')

			mask  = (XAXIS_FIT>3*g_avg_stddev) & (XAXIS_FIT <= +4*g_avg_stddev)	
			a4sgl = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='yellow', facecolor='yellow',
					linewidth=0, linestyle='solid', antialiased=True)
			mask  = (XAXIS_FIT>-4*g_avg_stddev) & (XAXIS_FIT <= -3*g_avg_stddev)	
			a4sgh = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='yellow', facecolor='yellow',
					linewidth=0, linestyle='solid', antialiased=True,label = 'Area 4$\sigma$')

			mask  = (XAXIS_FIT>4*g_avg_stddev) & (XAXIS_FIT <= +5*g_avg_stddev)	
			a5sgl = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='gray', facecolor='gray',
					linewidth=0, linestyle='solid', antialiased=True)
			mask  = (XAXIS_FIT>-5*g_avg_stddev) & (XAXIS_FIT <= -4*g_avg_stddev)	
			a5sgh = plt.fill_between(XAXIS_FIT[mask], g_avg_offset, 
					func_1D_Gaussian_O(XAXIS_FIT[mask], g_avg_mean,g_avg_amplitude,g_avg_stddev,g_avg_offset),
					alpha=0.5, edgecolor='gray', facecolor='gray',
					linewidth=0, linestyle='solid', antialiased=True, label = 'Area 5$\sigma$')
			l_a1sg = plt.legend([a1sg] ,['Area 1$\sigma$'],loc=3)
			l_a2sg = plt.legend([a2sgh],['Area 2$\sigma$'],loc=3)
			l_a3sg = plt.legend([a3sgh],['Area 3$\sigma$'],loc=3)
			l_a4sg = plt.legend([a4sgh],['Area 4$\sigma$'],loc=3)
			l_a5sg = plt.legend([a5sgh],['Area 5$\sigma$'],loc=3)
		
		lg=plt.legend(loc=4,prop={'size':14})
		lg.draw_frame(False)

		sgma0  = np.digitize(0*abs(g_avg_stddev),XAXIS,right=False)
		sgma1h = np.digitize(1*abs(g_avg_stddev),XAXIS,right=False)
		sgma2h = np.digitize(2*abs(g_avg_stddev),XAXIS,right=False)
		sgma3h = np.digitize(3*abs(g_avg_stddev),XAXIS,right=False)
		sgma4h = np.digitize(4*abs(g_avg_stddev),XAXIS,right=False)
		sgma5h = np.digitize(5*abs(g_avg_stddev),XAXIS,right=False)
		sgma6h = np.digitize(k*abs(g_avg_stddev),XAXIS,right=False)
		sgma7h = np.digitize(j*abs(g_avg_stddev),XAXIS,right=False)

		if (sgma1h) >= ((len(XAXIS)-1)):
			sgma0 =0
			sgma1h=0
			sgma2h=0
			sgma3h=0
			sgma4h=0
			sgma5h=0
			sgma6h=0
			sgma7h=0
		else:
			pass

		chn_ctr = float(sgma0)
		chn_sg1 = abs(sgma1h-chn_ctr)
		chn_sg2 = abs(sgma2h-chn_ctr)
		chn_sg3 = abs(sgma3h-chn_ctr)
		chn_sg4 = abs(sgma4h-chn_ctr)
		chn_sg5 = abs(sgma5h-chn_ctr)
		chn_fw2 = abs(sgma6h-chn_ctr)
		chn_fw4 = abs(sgma7h-chn_ctr)

		Header_Add(Cube2bPlot_1D,'FTA_AMP',np.round(g_avg_amplitude,5)                , header_comment = '1DGF Amplitude AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CTR',np.round(g_avg_mean,6)                     , header_comment = '1DGF Center AVG')
		Header_Add(Cube2bPlot_1D,'FTA_SIG',np.round(g_avg_stddev,2)                   , header_comment = '1DGF Sigma AVG')
		Header_Add(Cube2bPlot_1D,'FTA_FWH',linewidth_fwhm(np.round(g_avg_stddev,2))   , header_comment = '1DGF FWHM  AVG')
		Header_Add(Cube2bPlot_1D,'FTA_A2A',Area_avg[0]                                , header_comment = '1DGF Area A AVG')
		Header_Add(Cube2bPlot_1D,'FTA_A2M',Area_avg_man                               , header_comment = '1DGF Area M AVG')
		Header_Add(Cube2bPlot_1D,'FTA_LUM',Lum_Area_avg[0]                            , header_comment = '1DGF Ar2Lum AVG')
		Header_Add(Cube2bPlot_1D,'FTA_LLM',Lum_Area_avg[1]                            , header_comment = '1DGF Ar2Lum [log] AVG')

		Header_Add(Cube2bPlot_1D,'FTA_APE',0                                          , header_comment = '1DGF Amplitude Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CTE',g_avg_var_mea                              , header_comment = '1DGF Center Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_SGE',g_avg_var_std                              , header_comment = '1DGF Sigma Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_FWE',g_avg_var_std*linewidth_fwhm(1)            , header_comment = '1DGF FWHM  Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_AAE',Area_avg[1]                                , header_comment = '1DGF Area A Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_AME',Area_avg_man_err                           , header_comment = '1DGF Area M Err AVG')

		Header_Add(Cube2bPlot_1D,'FTA_CH2',chi2_avg                                   , header_comment = '1DGF Chi2 AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CHR',chi2_red_avg                               , header_comment = '1DGF Chi2 Reduced AVG')

		Header_Add(Cube2bPlot_1D,'FTA_ML1',lum_area_err_1_a[0]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH1',lum_area_err_1_a[1]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL1',lum_area_err_1_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH1',lum_area_err_1_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTA_ML2',lum_area_err_2_a[0]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH2',lum_area_err_2_a[1]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL2',lum_area_err_2_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH2',lum_area_err_2_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FTA_ML3',lum_area_err_3_a[0]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH3',lum_area_err_3_a[1]                        , header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL3',lum_area_err_3_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH3',lum_area_err_3_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML1',lum_area_err_1_m[0]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH1',lum_area_err_1_m[1]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL1',lum_area_err_1_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH1',lum_area_err_1_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML2',lum_area_err_2_m[0]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH2',lum_area_err_2_m[1]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL2',lum_area_err_2_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH2',lum_area_err_2_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML3',lum_area_err_3_m[0]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH3',lum_area_err_3_m[1]                        , header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL3',lum_area_err_3_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH3',lum_area_err_3_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FTA_CCT',chn_ctr                                    , header_comment = 'Channel # Central AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_1SG',chn_sg1                                    , header_comment = 'Channel Dif Limit 1SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_2SG',chn_sg2                                    , header_comment = 'Channel Dif Limit 2SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_3SG',chn_sg3                                    , header_comment = 'Channel Dif Limit 3SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_4SG',chn_sg4                                    , header_comment = 'Channel Dif Limit 4SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_5SG',chn_sg5                                    , header_comment = 'Channel Dif Limit 5SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_1FW',chn_fw2                                    , header_comment = 'Channel Dif Limit 1FW AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_2FW',chn_fw4                                    , header_comment = 'Channel Dif Limit 2FW AVG (PyId)')
		print (colored('1D gaussian (avg) fit performed OK! : ' , 'yellow'))
		print (colored(str(Cube2bPlot_1D),'yellow'))
	except (TypeError,RuntimeError):
		print
		print (colored('No 1D gaussian (avg) fit performed! : ' ,'yellow'))
		print (colored(str(Cube2bPlot_1D),'yellow'))
		print
		Header_Add(Cube2bPlot_1D,'FTA_AMP',np.nan, header_comment = '1DGF Amplitude AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CTR',np.nan, header_comment = '1DGF Center AVG')
		Header_Add(Cube2bPlot_1D,'FTA_SIG',np.nan, header_comment = '1DGF Sigma AVG')
		Header_Add(Cube2bPlot_1D,'FTA_FWH',np.nan, header_comment = '1DGF FWHM AVG')
		Header_Add(Cube2bPlot_1D,'FTA_A2A',np.nan, header_comment = '1DGF Area A AVG')
		Header_Add(Cube2bPlot_1D,'FTA_A2M',np.nan, header_comment = '1DGF Area M AVG')
		Header_Add(Cube2bPlot_1D,'FTA_LUM',np.nan, header_comment = '1DGF Ar2Lum AVG')
		Header_Add(Cube2bPlot_1D,'FTA_LLM',np.nan, header_comment = '1DGF Ar2Lum [log] AVG')

		Header_Add(Cube2bPlot_1D,'FTA_APE',np.nan, header_comment = '1DGF Amplitude Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CTE',np.nan, header_comment = '1DGF Center Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_SGE',np.nan, header_comment = '1DGF Sigma Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_FWE',np.nan, header_comment = '1DGF FWHM Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_AAE',np.nan, header_comment = '1DGF Area A Err AVG')
		Header_Add(Cube2bPlot_1D,'FTA_AME',np.nan, header_comment = '1DGF Area M Err AVG')

		Header_Add(Cube2bPlot_1D,'FTA_CH2',np.nan, header_comment = '1DGF Chi2 AVG')
		Header_Add(Cube2bPlot_1D,'FTA_CHR',np.nan, header_comment = '1DGF Chi2 Reduced AVG')

		Header_Add(Cube2bPlot_1D,'FTA_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTA_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTA_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTA_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum A AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] AVG Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMA_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum M AVG Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMA_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] AVG Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FTA_CCT',np.nan, header_comment = 'Channel # Central AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_1SG',np.nan, header_comment = 'Channel Dif Limit 1SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_2SG',np.nan, header_comment = 'Channel Dif Limit 2SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_3SG',np.nan, header_comment = 'Channel Dif Limit 3SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_4SG',np.nan, header_comment = 'Channel Dif Limit 4SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_5SG',np.nan, header_comment = 'Channel Dif Limit 5SG AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_1FW',np.nan, header_comment = 'Channel Dif Limit 1FW AVG (PyId)')
		Header_Add(Cube2bPlot_1D,'FTA_2FW',np.nan, header_comment = 'Channel Dif Limit 2FW AVG (PyId)')
	##############################################################FIT-AVG##############################################################

	##############################################################FIT-MED##############################################################
	XAXIS_FIT  = np.arange(XAXIS[0],XAXIS[-1],0.01)
	try:
		if fit_type == 'astropy':
			print
			print (colored('1D Gaussian Fit Mode Choosen: Astropy (No-Offset)','yellow'))
			print
			#astropy FIT#
			#http://docs.astropy.org/en/stable/api/astropy.modeling.fitting.LevMarLSQFitter.html#astropy.modeling.fitting.LevMarLSQFitter
			g_wght_med       = func_1D_Gaussian(XAXIS,mean_med,amplitude_med,std_wght)
			g_init_med       = apmd.models.Gaussian1D(amplitude=amplitude_med, mean=mean_med, stddev=stddev)
			g_init_med.amplitude.fixed = True
			fit_g_med        = apmd.fitting.LevMarLSQFitter()
			g_med            = fit_g_med(g_init_med, XAXIS, FLX_MED,weights=g_wght_med)
			g_med_cov        = fit_g_med.fit_info['param_cov']
			
			if (g_med_cov is None):
				g_med_var    = np.zeros((1,2))
				g_med_var[:] = np.nan
				g_med_var    = np.squeeze((g_med_var))
			elif np.linalg.det(g_med_cov) < 0:
				g_med_var    = np.zeros_like(np.diag(g_med_cov))
				g_med_var[:] = np.nan
				g_med_var    = np.squeeze((g_med_var))
			else:
				g_med_var = np.sqrt(np.diag(g_med_cov))
				g_med_var = np.squeeze((g_med_var))
			
			print (g_med_var)
			#DEFINE ASTROPY FIT VALUES#
			g_med_amplitude  = g_med.amplitude[0]
			g_med_mean       = g_med.mean[0]
			g_med_stddev     = abs(g_med.stddev[0])
			
			g_med_var_mea    = g_med_var[0]
			g_med_var_std    = g_med_var[-1]
			g_med_plt = g_med(XAXIS_FIT)
			g_exp_med = func_1D_Gaussian(XAXIS,g_med_mean,g_med_amplitude,g_med_stddev)
			#DEFINE ASTROPY FIT VALUES#		
			#astropy FIT#

		elif fit_type=='scipy':
			print
			print (colored('1D Gaussian Fit Mode Choosen: Scipy (Offset)','yellow'))
			print
			##scipy FIT##
			g_wght_med_O  = func_1D_Gaussian_O(XAXIS,mean_med,amplitude_med,std_wght,min(FLX_MED))
			initial_guess = (mean_med,amplitude_med,stddev,min(FLX_MED))
			#FLX_MED       = FLX_MED*g_wght_med_O
			popt, pcov    = scpopt.curve_fit(func_1D_Gaussian_O, XAXIS, FLX_MED, 
						p0=initial_guess,
						bounds=([mean_med-0.001,amplitude_med-0.0000001,-np.inf,-np.inf],
								[mean_med+0.001,amplitude_med+0.0000001, np.inf, np.inf])
						)
			perr        = np.sqrt(np.diag(pcov))
			data_fitted = func_1D_Gaussian_O(XAXIS, *popt)
			fit_res     = 'OK'

			#DEFINE SCIPY FIT VALUES#			
			g_med_mean       = np.round(popt[0],10)
			g_med_amplitude  = np.round(popt[1],10)
			g_med_stddev     = abs(np.round(popt[2],10))
			g_med_offset     = np.round(popt[3],10)
			
			g_med_var_mea          = np.round(perr[0],10)
			g_med_var_amplitude    = np.round(perr[1],10)
			g_med_var_std          = np.round(perr[2],10)
			g_med_var_offset       = np.round(perr[3],10)

			g_med_plt = func_1D_Gaussian_O(XAXIS_FIT, *popt)
			g_exp_med = func_1D_Gaussian_O(XAXIS,g_med_mean,g_med_amplitude,g_med_stddev,g_med_offset)

			#DEFINE SCIPY FIT VALUES#
			##scipy FIT##

		elif fit_type=='lmfit':
			print
			print (colored('1D Gaussian Fit Mode Choosen: Lmfit (Offset)','yellow'))
			print
			mod     = lmfit.models.ExpressionModel('Offset+(Amplitude*exp(-((x-X_0)**2)/(2*Sigma**2)))')
			params  = mod.make_params(Offset=min(FLX_MED), Amplitude=amplitude_med, X_0=mean_med, Sigma=stddev)

			if min(FLX_MED) == 0:
				min_FLX_MED = 0.001
			else:
				min_FLX_MED = min(FLX_MED)
				pass
			if amplitude_med == 0:
				amplitude_med = 0.001
			else:
				pass
			params['Offset'].min = min_FLX_MED - (abs(min_FLX_MED)*.01)
			params['Offset'].max = min_FLX_MED + (abs(min_FLX_MED)*.01)
			params['Amplitude'].min = (amplitude_med)-(abs(amplitude_med)*.01)
			params['Amplitude'].max = (amplitude_med)+(abs(amplitude_med)*.01)
			params['X_0'].min = mean_med-0.001
			params['X_0'].max = mean_med+0.001

			FLX_MED = FLX_MED.astype(float)
			XAXIS   = XAXIS.astype(float)
			result  = mod.fit(FLX_MED, params, x=XAXIS)

			g_med_mean             = result.params['X_0'].value
			g_med_amplitude        = result.params['Amplitude'].value
			g_med_stddev           = abs(result.params['Sigma'].value)
			g_med_offset           = result.params['Offset'].value

			g_med_var_mea          = result.params['X_0'].stderr
			g_med_var_amplitude    = result.params['Amplitude'].stderr
			g_med_var_std          = result.params['Sigma'].stderr
			g_med_var_offset       = result.params['Offset'].stderr

			g_med_cor_mea          = result.params['X_0'].correl
			g_med_cor_amplitude    = result.params['Amplitude'].correl
			g_med_cor_std          = result.params['Sigma'].correl
			g_med_cor_offset       = result.params['Offset'].correl

			g_med_chisqr = result.chisqr
			g_med_redchi = result.redchi

			g_med_plt = func_1D_Gaussian_O(XAXIS_FIT, g_med_mean, g_med_amplitude, g_med_stddev, g_med_offset )
			g_exp_med = func_1D_Gaussian_O(XAXIS,g_med_mean,g_med_amplitude,g_med_stddev,g_med_offset)
			g_avg_are = g_avg_plt


		Area_med         = scpint.quad(lambda x: g_med_amplitude*np.exp(-((x-g_med_mean)**2)/(2*g_med_stddev**2)), -np.inf, np.inf)

		Area_med_man     = g_med_stddev  * amplitude_med * np.sqrt(2*np.pi)
		Area_med_man_err = g_med_var_std * amplitude_med * np.sqrt(2*np.pi)

		Lum_Area_med     = FluxToLum(Area_med[0],z_f2l,frq_r)
		lum_area_err_1_a = Luminosity_Error(Area_med[0],redshift_inf_1,redshift_sup_1,Area_med[1],frq_r=frq_r)
		lum_area_err_2_a = Luminosity_Error(Area_med[0],redshift_inf_2,redshift_sup_2,Area_med[1],frq_r=frq_r)
		lum_area_err_3_a = Luminosity_Error(Area_med[0],redshift_inf_3,redshift_sup_3,Area_med[1],frq_r=frq_r)

		lum_area_err_1_m = Luminosity_Error(Area_med[0],redshift_inf_1,redshift_sup_1,Area_med_man_err,frq_r=frq_r)
		lum_area_err_2_m = Luminosity_Error(Area_med[0],redshift_inf_2,redshift_sup_2,Area_med_man_err,frq_r=frq_r)
		lum_area_err_3_m = Luminosity_Error(Area_med[0],redshift_inf_3,redshift_sup_3,Area_med_man_err,frq_r=frq_r)

		g_obs_med        = FLX_MED

		if Cube2bPlot_1D_Err == None:
			chi2_med      = sum((g_obs_med-g_exp_med)**2/g_exp_med)
			chi2_red_med  = chi2_med/float(len(XAXIS)-1)
		elif Cube2bPlot_1D_Err != None:
			chi2_med      = sum((g_obs_med-g_exp_med)**2/(FLX_ERR_STD**2))
			chi2_red_med  = chi2_med/float(len(XAXIS)-1)

		plt.plot(XAXIS_FIT, g_med_plt, color='blue',ls=':',alpha=0.4,label='')
		x_text_coor = min(XAXIS_FIT)
		y_text_coor = y_text_coor-step*10
		step = abs(abs(ax110.get_yticks(minor=True)[-1])-abs(ax110.get_yticks(minor=True)[-2]))*.4

		plt.text(x_text_coor,y_text_coor-step*0,
				'Gaussian fit :'+label_MED,
				ha='left' , va='bottom',color='blue',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*1,
				'A : '        + str(np.round(g_med_amplitude,10))              ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*2,
				'$\mu$ :'     + str(np.round(g_med_mean,3))                   + ' $\pm$ ' + str(np.round(g_med_var_mea,9))       ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*3,
				'$\sigma$ : ' + str(np.round(g_med_stddev,3))                 + ' $\pm$ ' + str(np.round(g_med_var_std,3))      ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*4,
				'fwhm : '     + str(np.round(linewidth_fwhm(g_med_stddev),3)) + ' $\pm$ ' + str(np.round(g_med_var_std*linewidth_fwhm(1),9)),
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*5,
				'S : '        + str(np.round(Area_med_man,3))                  + ' $\pm$ ' + str(np.round(Area_med_man_err,9))    ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*6,
				'L : '        + str(np.round(Lum_Area_med[0],3))  + '-' + str(np.round(lum_area_err_1_m[0],3)) + '+' + str(str(np.round(lum_area_err_1_m[1],9))) ,
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')
		plt.text(x_text_coor,y_text_coor-step*7,
				'log(L) : '   + str(np.round(Lum_Area_med[1],3))  + '-' + str(np.round(lum_area_err_1_m[2],3)) + '+' + str(str(np.round(lum_area_err_1_m[3],9))),
				ha='left' , va='bottom',color='black',fontsize=10,
				family = 'serif')

		k=2.355
		j=2*k
		h=max(g_med_plt*.05)

		sgma0  = np.digitize(0*abs(g_med_stddev),XAXIS,right=False)
		sgma1h = np.digitize(1*abs(g_med_stddev),XAXIS,right=False)
		sgma2h = np.digitize(2*abs(g_med_stddev),XAXIS,right=False)
		sgma3h = np.digitize(3*abs(g_med_stddev),XAXIS,right=False)
		sgma4h = np.digitize(4*abs(g_med_stddev),XAXIS,right=False)
		sgma5h = np.digitize(5*abs(g_med_stddev),XAXIS,right=False)
		sgma6h = np.digitize(k*abs(g_med_stddev),XAXIS,right=False)
		sgma7h = np.digitize(j*abs(g_med_stddev),XAXIS,right=False)

		if (sgma1h) >= ((len(XAXIS)-1)):
			sgma0 =0
			sgma1h=0
			sgma2h=0
			sgma3h=0
			sgma4h=0
			sgma5h=0
			sgma6h=0
			sgma7h=0
		else:
			pass

		chn_ctr = float(sgma0)
		chn_sg1 = abs(sgma1h-chn_ctr)
		chn_sg2 = abs(sgma2h-chn_ctr)
		chn_sg3 = abs(sgma3h-chn_ctr)
		chn_sg4 = abs(sgma4h-chn_ctr)
		chn_sg5 = abs(sgma5h-chn_ctr)
		chn_fw2 = abs(sgma6h-chn_ctr)
		chn_fw4 = abs(sgma7h-chn_ctr)
				
		Header_Add(Cube2bPlot_1D,'FTM_AMP',np.round(g_med_amplitude,5)                , header_comment = '1DGF Amplitude MED')
		Header_Add(Cube2bPlot_1D,'FTM_CTR',np.round(g_med_mean,6)                     , header_comment = '1DGF Center MED')
		Header_Add(Cube2bPlot_1D,'FTM_SIG',np.round(g_med_stddev,2)                   , header_comment = '1DGF Sigma MED')
		Header_Add(Cube2bPlot_1D,'FTM_FWH',linewidth_fwhm(np.round(g_med_stddev,2))   , header_comment = '1DGF FWHM MED')
		Header_Add(Cube2bPlot_1D,'FTM_A2A',Area_med[0]                                , header_comment = '1DGF Area MED')
		Header_Add(Cube2bPlot_1D,'FTM_A2M',Area_med_man                               , header_comment = '1DGF Area M MED')
		Header_Add(Cube2bPlot_1D,'FTM_LUM',Lum_Area_med[0]                            , header_comment = '1DGF Ar2Lum MED')
		Header_Add(Cube2bPlot_1D,'FTM_LLM',Lum_Area_med[1]                            , header_comment = '1DGF Ar2Lum [log] MED')

		Header_Add(Cube2bPlot_1D,'FTM_APE',0                                          , header_comment = '1DGF Amplitude Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_CTE',g_med_var_mea                              , header_comment = '1DGF Center Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_SGE',g_med_var_std                              , header_comment = '1DGF Sigma Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_FWE',g_med_var_std*linewidth_fwhm(1)            , header_comment = '1DGF FWHM Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_AAE',Area_med[1]                                , header_comment = '1DGF Area A Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_AME',Area_med_man_err                           , header_comment = '1DGF Area M Err MED')

		Header_Add(Cube2bPlot_1D,'FTM_CH2',chi2_med                                   , header_comment = '1DGF Chi2 MED')
		Header_Add(Cube2bPlot_1D,'FTM_CHR',chi2_red_med                               , header_comment = '1DGF Chi2 Reduced MED')
		
		Header_Add(Cube2bPlot_1D,'FTM_ML1',lum_area_err_1_a[0]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH1',lum_area_err_1_a[1]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL1',lum_area_err_1_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH1',lum_area_err_1_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTM_ML2',lum_area_err_2_a[0]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH2',lum_area_err_2_a[1]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL2',lum_area_err_2_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH2',lum_area_err_2_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FTM_ML3',lum_area_err_3_a[0]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH3',lum_area_err_3_a[1]                        , header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL3',lum_area_err_3_a[2]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH3',lum_area_err_3_a[3]                        , header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML1',lum_area_err_1_m[0]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH1',lum_area_err_1_m[1]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL1',lum_area_err_1_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH1',lum_area_err_1_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML2',lum_area_err_2_m[0]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH2',lum_area_err_2_m[1]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL2',lum_area_err_2_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH2',lum_area_err_2_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML3',lum_area_err_3_m[0]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH3',lum_area_err_3_m[1]                        , header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL3',lum_area_err_3_m[2]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH3',lum_area_err_3_m[3]                        , header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FTM_CCT',chn_ctr                                    , header_comment = 'Channel # Central MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_1SG',chn_sg1                                    , header_comment = 'Channel Dif Limit 1SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_2SG',chn_sg2                                    , header_comment = 'Channel Dif Limit 2SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_3SG',chn_sg3                                    , header_comment = 'Channel Dif Limit 3SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_4SG',chn_sg4                                    , header_comment = 'Channel Dif Limit 4SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_5SG',chn_sg5                                    , header_comment = 'Channel Dif Limit 5SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_1FW',chn_fw2                                    , header_comment = 'Channel Dif Limit 1FW MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_2FW',chn_fw4                                    , header_comment = 'Channel Dif Limit 2FW MED (PyId)')
		print
		print (colored('1D gaussian (med) fit performed OK! : ' , 'yellow'))
		print (colored(str(Cube2bPlot_1D),'yellow'))
		print
	except (TypeError,RuntimeError):
		print (colored('No 1D gaussian (med) fit performed! : ','yellow'))
		print (colored(str(Cube2bPlot_1D),'yellow'))
		print
		Header_Add(Cube2bPlot_1D,'FTM_AMP',np.nan, header_comment = '1DGF Amplitude MED')
		Header_Add(Cube2bPlot_1D,'FTM_CTR',np.nan, header_comment = '1DGF Center MED')
		Header_Add(Cube2bPlot_1D,'FTM_SIG',np.nan, header_comment = '1DGF Sigma MED')
		Header_Add(Cube2bPlot_1D,'FTM_FWH',np.nan, header_comment = '1DGF FWHM MED')
		Header_Add(Cube2bPlot_1D,'FTM_A2A',np.nan, header_comment = '1DGF Area A MED')
		Header_Add(Cube2bPlot_1D,'FTM_A2M',np.nan, header_comment = '1DGF Area M MED')
		Header_Add(Cube2bPlot_1D,'FTM_LUM',np.nan, header_comment = '1DGF Ar2Lum MED')
		Header_Add(Cube2bPlot_1D,'FTM_LLM',np.nan, header_comment = '1DGF Ar2Lum [log] MED')

		Header_Add(Cube2bPlot_1D,'FTM_APE',np.nan, header_comment = '1DGF Amplitude Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_CTE',np.nan, header_comment = '1DGF Center Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_SGE',np.nan, header_comment = '1DGF Sigma Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_FWE',np.nan, header_comment = '1DGF FWHM Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_AAE',np.nan, header_comment = '1DGF Area A Err MED')
		Header_Add(Cube2bPlot_1D,'FTM_AME',np.nan, header_comment = '1DGF Area M Err MED')

		Header_Add(Cube2bPlot_1D,'FTM_CH2',np.nan, header_comment = '1DGF Chi2 MED')
		Header_Add(Cube2bPlot_1D,'FTM_CHR',np.nan, header_comment = '1DGF Chi2 Reduced MED')

		Header_Add(Cube2bPlot_1D,'FTM_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FTM_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FTM_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTM_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum A MED Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FTM_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum A [log] MED Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML1',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH1',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 1 sgm hg lmt 84.1 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm lw lmt 15.9 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH1',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 1 sgm hg lmt 84.1 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML2',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH2',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 2 sgm hg lmt 97.7 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm lw lmt 2.30 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH2',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 2 sgm hg lmt 97.7 pct')

		Header_Add(Cube2bPlot_1D,'FMM_ML3',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMM_MH3',np.nan, header_comment = '1DGF Ar2Lum Lum M MED Err 3 sgm hg lmt 99.8 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LL3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm lw lmt 0.20 pct')
		Header_Add(Cube2bPlot_1D,'FMM_LH3',np.nan, header_comment = '1DGF Ar2Lum Lum M [log] MED Err 3 sgm hg lmt 99.8 pct')

		Header_Add(Cube2bPlot_1D,'FTM_CCT',np.nan, header_comment = 'Channel # Central MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_1SG',np.nan, header_comment = 'Channel Dif Limit 1SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_2SG',np.nan, header_comment = 'Channel Dif Limit 2SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_3SG',np.nan, header_comment = 'Channel Dif Limit 3SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_4SG',np.nan, header_comment = 'Channel Dif Limit 4SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_5SG',np.nan, header_comment = 'Channel Dif Limit 5SG MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_1FW',np.nan, header_comment = 'Channel Dif Limit 1FW MED (PyId)')
		Header_Add(Cube2bPlot_1D,'FTM_2FW',np.nan, header_comment = 'Channel Dif Limit 2FW MED (PyId)')
	##############################################################FIT-MED##############################################################
	ax110_twin = ax110.twiny()
	ax110_twin_major = (np.linspace(1,len(VEL_AXS)+1,len(VEL_AXS)+1)).astype(int)
	ax110_twin.set_xticks(ax110_twin_major)
	ax110_twin.set_xticklabels(ax110_twin_major,fontsize=16,family = 'serif')
	ax110_twin.set_xlabel('#CHN (' + str(fit_type) + ')',fontsize=12,family='serif')

	##########################################SAVE#####################################
	plt.savefig(PLOTFILENAME)

	if verbose == True:
		print
		print (colored('Generated Plot: ' + str(PLOTFILENAME) + ' Frequency channels: ' + str(len(XAXIS)),'cyan'))
	elif verbose ==False:
		pass
	if epssave == True:
		plt.savefig('Spectra.eps', rasterized=True)
		#os.system('open Spectra.eps')
	elif epssave == False:
		pass
	if showplot == True:
		#os.system('open '+str(PLOTFILENAME))
		pass
	elif showplot == False:
		pass	
	plt.close('all')

def fit_stars(synt_img,res_tbl,n_me,n_md,n_st,n_rms,output_table_res_fit,displ_fit,*args,**kwargs):
	verbose = kwargs.get('verbose',False)
	in_val = readtable_cat(res_tbl,'ascii')
	X_in   = []
	Y_in   = []
	A_in   = []
	A_M_in = []
	SX_in  = []
	SY_in  = []
	T_in   = []
	OF_in  = []
	STN_in = []

	X_out  = []
	Y_out  = []
	A_out  = []
	SX_out = []
	SY_out = []
	T_out  = []
	OF_out = []

	X_err  = []
	Y_err  = []
	A_err  = []
	SX_err = []
	SY_err = []
	T_err  = []
	OF_err = []

	NME = []
	NMD = []
	NST = []
	NRM = []

	widgets = ['Fitting Stars ('+str(n)+'):', Percentage(), ' ', Bar(marker='*',left='[',right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=n)
	pbar.start()


	for j in range(n):

		pbar.update(j)

		X0_0      = in_val[1][j]
		Y0_0      = in_val[2][j]
		A_0       = in_val[3][j]
		A_M_0     = in_val[4][j]
		SIGMAX_0  = in_val[5][j]
		SIGMAY_0  = in_val[6][j]
		THETA_0   = in_val[7][j]
		OFFSET_0  = in_val[8][j]
		N_TMS_N_0 = in_val[9][j]

		X_in.append(float(round(X0_0,5)))
		Y_in.append(float(round(Y0_0,5)))
		A_in.append(float(round(A_0,5)))
		A_M_in.append(float(round(A_M_0,5)))
		SX_in.append(float(round(SIGMAX_0,5)))
		SY_in.append(float(round(SIGMAY_0,5)))
		T_in.append(float(round(THETA_0,5)))
		OF_in.append(float(round(OFFSET_0,5)))
		STN_in.append(str(round(N_TMS_N_0,5)))

		Cube_res_fit = fit_2D_Gaussian(synt_img,nx,ny,X0_0,Y0_0,A_0,SIGMAX_0,SIGMAY_0,THETA_0,OFFSET_0,displ_fit)

		X0_F      = res_fit[0][0]
		Y0_F      = res_fit[0][1]
		A_F       = res_fit[0][2]
		SIGMAX_F  = res_fit[0][3]
		SIGMAY_F  = res_fit[0][4]
		THETA_F   = res_fit[0][5]
		OFFSET_F  = res_fit[0][6]

		X0_E      = res_fit[2][0]
		Y0_E      = res_fit[2][1]
		A_E       = res_fit[2][2]
		SIGMAX_E  = res_fit[2][3]
		SIGMAY_E  = res_fit[2][4]
		THETA_E   = res_fit[2][5]
		OFFSET_E  = res_fit[2][6]

		fit_st_res = res_fit[3]

		if fit_st_res == 'OK':

			X_out.append(float(round(X0_F,5)))
			Y_out.append(float(round(Y0_F,5)))
			A_out.append(float(round(A_F,5)))
			SX_out.append(float(round(SIGMAX_F,5)))
			SY_out.append(float(round(SIGMAY_F,5)))
			T_out.append(float(round(THETA_F,5)))
			OF_out.append(float(round(OFFSET_F,5)))

			X_err.append(float(round(X0_E,5)))
			Y_err.append(float(round(Y0_E,5)))
			A_err.append(float(round(A_E,5)))
			SX_err.append(float(round(SIGMAX_E,5)))
			SY_err.append(float(round(SIGMAY_E,5)))
			T_err.append(float(round(THETA_E,5)))
			OF_err.append(float(round(OFFSET_E,5)))

			NME.append(float(round(n_me,10)))
			NMD.append(float(round(n_md,10)))
			NST.append(float(round(n_st,10)))
			NRM.append(float(round(n_rms,10)))

		elif fit_st_res == 'ERR':

			X_out.append('ERR')
			Y_out.append('ERR')
			A_out.append('ERR')
			SX_out.append('ERR')
			SY_out.append('ERR')
			T_out.append('ERR')
			OF_out.append('ERR')

			X_err.append('ERR')
			Y_err.append('ERR')
			A_err.append('ERR')
			SX_err.append('ERR')
			SY_err.append('ERR')
			T_err.append('ERR')
			OF_err.append('ERR')

			NME.append(float(round(n_me,10)))
			NMD.append(float(round(n_md,10)))
			NST.append(float(round(n_st,10)))
			NRM.append(float(round(n_rms,10)))

		if verbose == True:
			print
			print 'Star:',j+1
			print 'Initial Values / Fit Parameters:' 
			print 'X  :',X0_0,X0_F    
			print 'Y  :',Y0_0,Y0_F    
			print 'A  :',A_0,A_F     
			print 'SX :',SIGMAX_0,SIGMAX_F
			print 'SY :',SIGMAY_0,SIGMAY_F
			print 'T  :',THETA_0,THETA_F 
			print 'OF :',OFFSET_0,OFFSET_F
			print
		elif verbose == False:
			pass
	if output_table_res_fit == 'yes':
		rt           = Table()
		rt['X0_0']   = X_in
		rt['Y0_0']   = Y_in
		rt['A_0 ']   = A_in
		rt['A_M_0']  = A_M_in
		rt['SX_0']   = SX_in
		rt['SY_0']   = SY_in
		rt['T_0']    = T_in
		rt['OFS_0']  = OF_in

		rt['X_F']    = X_out
		rt['Y_F']    = Y_out
		rt['A_F']    = A_out
		rt['SX_F']   = SX_out
		rt['SY_F']   = SY_out
		rt['T_F']    = T_out
		rt['OFS_F']  = OF_out

		rt['X_E']    = X_err
		rt['Y_E']    = Y_err
		rt['A_E']    = A_err
		rt['SX_E']   = SX_err
		rt['SY_E']   = SY_err
		rt['T_E']    = T_err
		rt['OFS_E']  = OF_err

		rt['N_ME']   = NME
		rt['N_MD']   = NMD
		rt['N_ST']   = NST
		rt['N_RMS']  = NRM

		rt.write(output_table_fit, format='ascii.fixed_width_two_line')	
		print 
		print 'Results containing fit parameters: ',output_table_fit
	elif output_table_res_fit=='no':
		pass
	pbar.finish()
	return X_in,Y_in,A_in,A_M_in,SX_in,SY_in,T_in,OF_in,STN_in,X_out,Y_out,A_out,SX_out,SY_out,T_out,OF_out,X_err,Y_err,A_err,SX_err,SY_err,T_err,OF_err,NME,NMD,NST,NRM

def Image2D_fit_2D_Gaussian(Cube2bFit,*args,**kwargs):
	dest_dir = kwargs.get('dest_dir',None)
	verbose  = kwargs.get('verbose' ,None)
	clp_fnc  = kwargs.get('clp_fnc' ,'sum')
	circular = kwargs.get('circular',True)
	x_ref    = kwargs.get('x_ref',0)
	y_ref    = kwargs.get('y_ref',0)

	dest_dir_plt = kwargs.get('dest_dir',None)
	dest_dir_clp = kwargs.get('dest_dir',None)

	slice_nmbr = 'CSL'
	if dest_dir_clp != None:
		Cube2bclp_2D_opt = dest_dir   + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+slice_nmbr+'.fits'
		PLOTFILENAME     = str(dest_dir) + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+slice_nmbr+'.pdf'
	elif dest_dir_clp == None:
		Cube2bclp_2D_opt = stp_dir_res + (Cube2bFit.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+slice_nmbr+'.fits'
		PLOTFILENAME     = plt_dir_res + '/' + (str(Cube2bFit).split('.fits')[0]).split('/')[-1] + '-2DCGF-'+slice_nmbr+'.pdf'

	scale_deg    = Header_Get(Cube2bFit,'CDELT2')
	scale_arcsec = scale_deg*3600#0.00027777778

	cube_data    = np.asarray(astropy.io.fits.getdata(Cube2bFit,memmap=False) )

	data_2b_plot = cube_data
	Message = 'Fitting gaussian with slice number : ' 
	plt_tlt = 'Slice: ' 
	clp_hdr = 'C'  
	clp_hdc = 'CSL' 
	data_2b_plt = np.asarray(astropy.io.fits.getdata(Cube2bFit,memmap=False))
	data_2b_plt_clp = data_2b_plt#[slc_nmb]

	Wrt_FITS_File(data_2b_plot,Cube2bclp_2D_opt)

	print
	print colored('Input Cube for collapse : ' + Cube2bFit,'magenta')
	print colored('Resulting Collapsed Cube: ' + Cube2bclp_2D_opt,'yellow')
	print
	data_2b_plot = np.squeeze(data_2b_plot)
	nx_f2DG, ny_f2DG = data_2b_plot.shape
	nx,ny            = nx_f2DG,ny_f2DG

	X0_f2DG     = kwargs.get('X0_f2DG',int(np.ceil(nx_f2DG/2)))
	Y0_f2DG     = kwargs.get('Y0_f2DG',int(np.ceil(ny_f2DG/2)))
	A_f2DG      = kwargs.get('A_f2DG',1)
	SIGMAX_f2DG = kwargs.get('SIGMAX_f2DG',1)
	SIGMAY_f2DG = kwargs.get('SIGMAY_f2DG',1)
	THETA_f2DG  = kwargs.get('THETA_f2DG',0)
	OFS_f2DG    = kwargs.get('OFS_f2DG',0)
	displ_s_f   = kwargs.get('displ_s_f',False)
	verbose     = kwargs.get('verbose',False)

	# Create x and y indices
	x    = np.linspace(0, nx_f2DG, nx_f2DG)-0.5
	y    = np.linspace(0, ny_f2DG, ny_f2DG)-0.5
	x, y = np.meshgrid(x, y)

	data = data_2b_plot

	initial_guess = (X0_f2DG,Y0_f2DG,A_f2DG,SIGMAX_f2DG,SIGMAY_f2DG,THETA_f2DG,OFS_f2DG)

	xdata       = np.vstack((x.ravel(),y.ravel()))
	ydata       = data.ravel()
	try:
		popt, pcov  = optimize.curve_fit(func_2D_Gaussian, xdata, ydata, 
					p0=initial_guess,
					bounds=([X0_f2DG-1,Y0_f2DG-1,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
							[X0_f2DG+1,Y0_f2DG+1, np.inf, np.inf, np.inf, np.inf, np.inf]))
		perr        = np.sqrt(np.diag(pcov))
		data_fitted = func_2D_Gaussian((x, y), *popt, circular=circular)
		fit_res     = 'OK'
		X0_F        = np.round(popt[0],0)
		Y0_F        = np.round(popt[1],0)
		X_DIF       = np.round(X0_F,0) - X0_f2DG
		Y_DIF       = np.round(Y0_F,0) - Y0_f2DG

	except RuntimeError:
		popt, pcov  = [0,0,0,0,0,0,0],[0,0,0,0,0,0,0]
		perr        = [0,0,0,0,0,0,0]
		X0_F        = 0
		Y0_F        = 0
		X_DIF       = 0
		Y_DIF       = 0
		data_fitted = func_2D_Gaussian((x, y), *popt, circular=circular)
		fit_res     = 'ERR'
		print("Error - curve_fit failed")

	X0_F         = np.round(popt[0],0)
	Y0_F         = np.round(popt[1],0)
	A_F          = np.round(popt[2],9)
	SIGMAX_F     = np.round(popt[3],9)
	SIGMAY_F     = np.round(popt[4],9)
	THETA_F      = np.round(popt[5],9)
	OFFSET_F     = np.round(popt[6],9)

	X0_E         = np.round(perr[0],9)
	Y0_E         = np.round(perr[1],9)
	A_E          = np.round(perr[2],9)
	SIGMAX_E     = np.round(perr[3],9)
	SIGMAY_E     = np.round(perr[4],9)
	THETA_E      = np.round(perr[5],9)
	OFFSET_E     = np.round(perr[6],9)

	if circular ==True:
		SIGMAY_F = SIGMAX_F
		SIGMAY_E = SIGMAX_E
	elif circular == False:
		pass

	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_XCT',X0_F       ,header_comment = '2DGF X ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_YCT',Y0_F       ,header_comment = '2DGF Y ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_AMP',A_F        ,header_comment = '2DGF Amplitude ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_SGX',SIGMAX_F   ,header_comment = '2DGF Sigma X ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_SGY',SIGMAY_F   ,header_comment = '2DGF Sigma Y ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_THT',THETA_F    ,header_comment = '2DGF Theta ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_OFS',OFFSET_F   ,header_comment = '2DGF Offset ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_XCE',X0_E       ,header_comment = '2DGF X Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_YCE',Y0_E       ,header_comment = '2DGF Y Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_AME',A_E        ,header_comment = '2DGF Amplitude Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_SXE',SIGMAX_E   ,header_comment = '2DGF Sigma X Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_SYE',SIGMAY_E   ,header_comment = '2DGF Sigma Y Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_THE',THETA_E    ,header_comment = '2DGF Theta Err ' + clp_hdc)
	Header_Add(Cube2bclp_2D_opt, clp_hdr + '2G_OFE',OFFSET_E   ,header_comment = '2DGF Offset Err ' + clp_hdc)

	DGF_vlm = abs(SIGMAX_F) * abs(SIGMAX_F) * A_F * 2 * np.pi

	DGF_vle = (2*np.pi * (SIGMAX_F**2)) * (np.sqrt(((SIGMAX_F**2) * (A_E**2)) + (4 * (A_F**2) * (SIGMAX_E**2))))
	
	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_FLS',DGF_vlm   ,header_comment = '2DGF Vol ' + clp_hdc)

	Header_Add(Cube2bclp_2D_opt,clp_hdr + '2G_FSE',DGF_vle   ,header_comment = '2DGF Vol ' + clp_hdc + ' Err')

	Header_Add(Cube2bFit, clp_hdr + '2G_XCT',X0_F       ,header_comment = '2DGF X ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_YCT',Y0_F       ,header_comment = '2DGF Y ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_AMP',A_F        ,header_comment = '2DGF Amplitude ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_SGX',SIGMAX_F   ,header_comment = '2DGF Sigma X ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_SGY',SIGMAY_F   ,header_comment = '2DGF Sigma Y ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_THT',THETA_F    ,header_comment = '2DGF Theta ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_OFS',OFFSET_F   ,header_comment = '2DGF Offset ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_XCE',X0_E       ,header_comment = '2DGF X Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_YCE',Y0_E       ,header_comment = '2DGF Y Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_AME',A_E        ,header_comment = '2DGF Amplitude Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_SXE',SIGMAX_E   ,header_comment = '2DGF Sigma X Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_SYE',SIGMAY_E   ,header_comment = '2DGF Sigma Y Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_THE',THETA_E    ,header_comment = '2DGF Theta Err ' + clp_hdc)
	Header_Add(Cube2bFit, clp_hdr + '2G_OFE',OFFSET_E   ,header_comment = '2DGF Offset Err ' + clp_hdc)

	Header_Add(Cube2bFit,clp_hdr + '2G_FLS',DGF_vlm   ,header_comment = '2DGF Vol ' + clp_hdc)

	Header_Add(Cube2bFit,clp_hdr + '2G_FSE',DGF_vle   ,header_comment = '2DGF Vol ' + clp_hdc + ' Err')

	if verbose == True:
		print
		print 'Initial Guess:'
		print 'X0_G         : ',X0_f2DG
		print 'Y0_G         : ',Y0_f2DG
		print 'A_G          : ',A_f2DG
		print 'SIGMAX_G     : ',SIGMAX_f2DG
		print 'SIGMAY_G     : ',SIGMAY_f2DG
		print 'THETA_G      : ',THETA_f2DG
		print 'OFFSET_G     : ',OFS_f2DG
		print 
		print colored(Message,'yellow')

		print
		print 'Fit Values   :'
		print 'X0_F         : ',X0_F    ,' +- ',X0_E    
		print 'Y0_F         : ',Y0_F    ,' +- ',Y0_E    
		print 'A_F          : ',A_F     ,' +- ',A_E     
		print 'SIGMAX_F     : ',SIGMAX_F,' +- ',SIGMAX_E
		print 'SIGMAY_F     : ',SIGMAY_F,' +- ',SIGMAY_E
		print 'THETA_F      : ',THETA_F ,' +- ',THETA_E 
		print 'OFFSET_F     : ',OFFSET_F,' +- ',OFFSET_E
		print 'Area_F       : ',DGF_vlm ,' +- ',DGF_vle
		print 'Volume_F     : ',DGF_vx1 ,' +- ',DGF_v1e
		print
		print 'Shift from the X coordinate center:',X_DIF
		print 'Shift from the Y coordinate center:',Y_DIF
		print
		print colored('Generated Plot: ' + str(PLOTFILENAME) ,'cyan')
		print
	elif verbose == False:
		pass
		
	if displ_s_f == True:
		fxsize=9
		fysize=8
		f = plt.figure(num=None, figsize=(fxsize, fysize), dpi=180, facecolor='w',
			edgecolor='k')
		plt.subplots_adjust(
			left 	= (16/25.4)/fxsize, 
			bottom 	= (12/25.4)/fysize, 
			right 	= 1 - (6/25.4)/fxsize, 
			top 	= 1 - (15/25.4)/fysize)
		plt.subplots_adjust(hspace=0)

		gs0 = gridspec.GridSpec(1, 1)
		gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
		ax110 = plt.Subplot(f, gs11[0,0])
		f.add_subplot(ax110)

		ax110.set_rasterization_zorder(1)
		plt.autoscale(enable=True, axis='y', tight=False)

		plt.title(plt_tlt + ' (' +  str(x_ref) + ','+str(y_ref)+')')
		plt.xlabel('X',fontsize=16)
		plt.ylabel('Y',fontsize=16)
		plt.tick_params(which='both', width=1.0)
		plt.tick_params(which='major', length=10)
		plt.tick_params(which='minor', length=5)
		ax110.minorticks_on()

		if ('_ms.' in Cube2bFit) or ('dta_in.' in Cube2bFit) or ('dta_ot.' in Cube2bFit):
			tick_color = 'white'
		elif ('msk_in.' in Cube2bFit) or ('crc.' in Cube2bFit) or ('msk_ot.' in Cube2bFit):
			tick_color = 'black'
		else:
			tick_color = 'white'

		ax110.xaxis.set_tick_params(which='both',labelsize=16,direction='in',color=tick_color,bottom='on',top='on',left='on',right='on')
		ax110.yaxis.set_tick_params(which='both',labelsize=16,direction='in',color=tick_color,bottom='on',top='on',left='on',right='on')

		plt.imshow(ydata.reshape(nx, ny), cmap=plt.cm.viridis, origin='lower',
		    extent=(x.min(), x.max(), y.min(), y.max()))
		cbar = plt.colorbar(format=mpl.ticker.FuncFormatter(fmt))
		cbar.set_label('S [mJy]', rotation=270)

		min_y, max_y = ax110.get_ylim()
		min_x, max_x = ax110.get_xlim()	

		plt.text(0,max_y-(max_y/10),
				'1 pix = '  + str(scale_arcsec) + ' arcsec',  
				ha='left' , va='baseline',color='white',fontsize=16)

		X0,Y0 = X0_F,Y0_F

		sigx  = SIGMAX_F
		sigy  = SIGMAY_F
		theta = THETA_F
		try:
			colors=['white','white','white','white','white']
			for j in xrange(1, 4):
			    ell = Ellipse(xy=(X0, Y0),
			        width=sigx*2*j, height=sigy*2*j,
			        angle=theta,
			        edgecolor=colors[j])
			    ell.set_facecolor('none')
			    ax110.add_artist(ell)
		
			plt.text(0,min_y+(3*(max_y)/30),
				'Fit:',
				ha='left' , va='bottom',color='white',fontsize=16)

			plt.text(0,min_y+(2*(max_y)/30),
				'X$_{0}$: '       + str(x_ref+X_DIF)          + ', Y$_{0}$ '+ str(y_ref+Y_DIF)  + ', '
				'A : '            + str(np.round(popt[2],3))  + ' $\pm$ '   + str(np.round(A_E,5)),
				ha='left' , va='bottom',color='white',fontsize=16)

			plt.text(0,min_y+(1*(max_y)/30),
				'$\sigma_{x,y}$ : ' + str(np.round(SIGMAX_F,3)) + ' $\pm$ ' + str(np.round(SIGMAX_E,3)) + ', '
				'S(A) : '           + str(np.round(DGF_vlm,3))  + ' $\pm$ ' + str(np.round(DGF_vle,3)),
				ha='left' , va='bottom',color='white',fontsize=16)
		except ValueError:
			pass

		plt.scatter(X0_F    + 0.0, Y0_F    + 0.0, s=25, c='white', marker='x')
		plt.scatter(X0_f2DG + 0.0, Y0_f2DG + 0.0, s=25, c='black', marker='+')

		ax110.xaxis.set_major_locator(ScaledLocator(dx=1.0,x0=-X0_f2DG + x_ref ))
		ax110.yaxis.set_major_locator(ScaledLocator(dx=1.0,x0=-Y0_f2DG + y_ref ))
		ax110.xaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-X0_f2DG+x_ref ))
		ax110.yaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-Y0_f2DG+y_ref ))

		plt.savefig(PLOTFILENAME)
	elif displ_s_f == False:
		pass

	print PLOTFILENAME
	return popt,pcov,perr,fit_res,X_DIF,Y_DIF
####Fnc_Syn_Spc####