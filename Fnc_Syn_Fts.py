from astropy.io import fits as apfts

####Fnc_Syn_Fts####
def Wrt_FITS_File(img_inpt_array,otp_img_fn,*args, **kwargs):
	hdu        = apfts.PrimaryHDU(img_inpt_array)
	hdulist    = apfts.HDUList([hdu])
	hdulist.writeto(otp_img_fn,overwrite=True)
	hdulist.close()
	return otp_img_fn

def Header_Get(image,field):
	hdulist = apfts.open(image)
	try:
		header  = float(hdulist[0].header[field])
	except:
		header  = str(hdulist[0].header[field])
	hdulist.close()
	return header

def Header_Get_All(fitsfile_in,*args, **kwargs):
	hdulist_afh  = apfts.open(fitsfile_in, mode='update')
	prihdr       = hdulist_afh[0].header
	hdulist_afh.flush()
	hdulist_afh.close()
	return prihdr

def Header_Updt(fitsfile_in,field,value,*args, **kwargs):
	header_comment = kwargs.get('header_comment',None)
	hdulist_afh  = apfts.open(fitsfile_in, mode='update')
	prihdr       = hdulist_afh[0].header
	prihdr.set(field, value,comment=header_comment)
	hdulist_afh.flush()
	hdulist_afh.close()

def Header_Add(fitsfile_in,field,value,*args, **kwargs):
	header_comment = kwargs.get('header_comment',None)
	hdulist_afh  = apfts.open(fitsfile_in, mode='update')
	prihdr       = hdulist_afh[0].header
	try:
		prihdr.set(field, value,comment=header_comment)
	except ValueError:
		prihdr.set(field, str(value),comment=header_comment)
	hdulist_afh.flush()
	hdulist_afh.close()	

def Header_Get_Add(fits_ipfn,header,value,*args, **kwargs):
	header_comment = kwargs.get('header_comment',None)
	try:
		head_val      = (Header_Get(fits_ipfn,header))
	except KeyError:
		Header_Add(fits_ipfn,header,value,header_comment=header_comment)
		head_val      = (Header_Get(fits_ipfn,header))
	return head_val	

def Header_Copy(fits_to,fits_from,header,*args,**kwargs):
	header_comment = kwargs.get('header_comment',None)
	value = Header_Get(fits_from,header)
	Header_Get_Add(fits_to,header,value,header_comment=header_comment)	

def Header_History_Init(fits_ipfn_HHI,*args,**kwargs):
	hdr_nme = kwargs.get('hdr_nme', None) 
	hdr_val = kwargs.get('hdr_val', None) 
	Header_Add(fits_ipfn_HHI,'h_s_c',0,header_comment='History Step Last')
	Header_Add(fits_ipfn_HHI,'h_s_0',str((fits_ipfn_HHI.rsplit('/',1)[1]).rsplit('.',1)[0]),header_comment='History Step 0')
	if hdr_nme != None:
		if len(hdr_nme)==len(hdr_val):
			[Header_Add(fits_ipfn_HHI,str(name),float(val)) for name,val in zip(hdr_nme,hdr_val)]
		elif len(hdr_nme)!=len(hdr_val):
			print ('Error, header and values are different')
	elif hdr_nme == None:
		pass

def Header_History_Step(fits_ipfn_prev,fits_ipfn_pst,*args,**kwargs):
	try:
		head_val      = (Header_Get(fits_ipfn_prev,'h_s_c'))
		#Header_Updt(fits_ipfn,header,value)
	except KeyError:
		head_val = 0
		Header_Add(fits_ipfn_prev,'h_s_c',0,header_comment='History Step Last')
		Header_Add(fits_ipfn_prev,'h_s_0',str((fits_ipfn_prev.rsplit('/',1)[1]).rsplit('.',1)[0]),header_comment='History Step 0')
		Header_Add(fits_ipfn_pst,'h_s_0',str((fits_ipfn_prev.rsplit('/',1)[1]).rsplit('.',1)[0]),header_comment='History Step 0')
	c_step = int(head_val + 1)
	header = 'h_s_'+str(c_step)
	Header_Add(fits_ipfn_pst,header,str((fits_ipfn_pst.rsplit('/',1)[1]).rsplit('.',1)[0]))
	Header_Updt(fits_ipfn_pst,'h_s_c',c_step)
from astropy.io import fits
####Fnc_Syn_Fts####
