import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib import colors as mcolors
import matplotlib.ticker as mticker
import scipy.integrate as integrate

plt.rcParams.update({'font.family':'serif'})

from matplotlib.backends.backend_pdf import PdfPages

from Fnc_Syn_Dir import *
from Fnc_Syn_Fts import *
from Fnc_Syn_Spc import *
from Fnc_Syn_Mth import *
from Fnc_Syn_Tbl import *

####Fnc_Syn_Plt####
def align_yaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	_, y1 = ax1.transData.transform((0, v1))
	_, y2 = ax2.transData.transform((0, v2))
	inv = ax2.transData.inverted()
	_, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
	miny, maxy = ax2.get_ylim()
	ax2.set_ylim(miny+dy, maxy+dy)

def align_xaxis(ax1, v1, ax2, v2, y2min, y2max):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1."""

    """where y2max is the maximum value in your secondary plot. I haven't
     had a problem with minimum values being cut, so haven't set this. This
     approach doesn't necessarily make for axis limits at nice near units,
     but does optimist plot space"""

    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    scale = 1
    while scale*(maxy+dy) < y2max:
        scale += 0.05

class ScaledLocator(mpl.ticker.MaxNLocator):
    """
    Locates regular intervals along an axis scaled by *dx* and shifted by
    *x0*. For example, this would locate minutes on an axis plotted in seconds
    if dx=60.  This differs from MultipleLocator in that an approriate interval
    of dx units will be chosen similar to the default MaxNLocator.
    """
    def __init__(self, dx=1.0, x0=0.0):
        self.dx = dx
        self.x0 = x0
        mpl.ticker.MaxNLocator.__init__(self, nbins=9, steps=[1, 2, 5, 10])

    def rescale(self, x):
        return x / self.dx + self.x0
    def inv_rescale(self, x):
        return  (x - self.x0) * self.dx

    def __call__(self): 
        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = self.rescale(vmin), self.rescale(vmax)
        vmin, vmax = mpl.transforms.nonsingular(vmin, vmax, expander = 0.05)
        locs = self.bin_boundaries(vmin, vmax)
        locs = self.inv_rescale(locs)
        prune = self._prune
        if prune=='lower':
            locs = locs[1:]
        elif prune=='upper':
            locs = locs[:-1]
        elif prune=='both':
            locs = locs[1:-1]
        return self.raise_if_exceeds(locs)

class ScaledFormatter(mpl.ticker.OldScalarFormatter):
    """Formats tick labels scaled by *dx* and shifted by *x0*."""
    def __init__(self, dx=1.0, x0=0.0, **kwargs):
        self.dx, self.x0 = dx, x0

    def rescale(self, x):
        return x / self.dx + self.x0

    def __call__(self, x, pos=None):
        xmin, xmax = self.axis.get_view_interval()
        xmin, xmax = self.rescale(xmin), self.rescale(xmax)
        d = abs(xmax - xmin)
        x = self.rescale(x)
        s = self.pprint_val(x, d)
        return s

def fmt(x, pos):
    #x=x/1e-3
    a, b = '{:.1f}e-3'.format(x/1e-3).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def Plot_Cube_2D(Cube2bplot_2D,*args,**kwargs):
    dest_dir_plt = kwargs.get('dest_dir_plt',None)
    dest_dir_clp = kwargs.get('dest_dir_clp',None)
    autoaxis     = kwargs.get('autoaxis',False)
    verbose      = kwargs.get('verbose' , False)
    epssave      = kwargs.get('epssave' , False)
    showplot     = kwargs.get('showplot', False) 
    slc_nmb      = kwargs.get('slc_nmb' , None) 
    clp_fnc      = kwargs.get('clp_fnc' , 'sum')

    redshift     = kwargs.get('redshift' ,'1')
    rst_frq      = kwargs.get('rst_frq'  ,'1')

    x_ref        = kwargs.get('x_ref',0)
    y_ref        = kwargs.get('y_ref',0)
    ap_size      = kwargs.get('ap_size',0)

    z_avg        = kwargs.get('z_avg',Header_Get(Cube2bplot_2D,'STZ_AVG'))
    z_med        = kwargs.get('z_med',Header_Get(Cube2bplot_2D,'STZ_MED'))
    frq_r        = kwargs.get('frq_r',Header_Get(Cube2bplot_2D,'RESTFRQ'))
    z_f2l        = z_med

    prefix       = kwargs.get('prefix','')

    dest_dir_plt = kwargs.get('dest_dir_plt',None)

    Cube_Info    = Cube_Header_Get(Cube2bplot_2D,frq_r* u.Hz)
    FRQ_AXS      = Cube_Info[16].value
    VEL_AXS      = Cube_Info[17].value

    flx_scl      = kwargs.get('flx_scl',1e-06)

    if slc_nmb != None:
        pass
    elif slc_nmb == None:
        slc_nmb = 0

    if flx_scl == 1e-06:
        scl_prfx = '$\mu$'
    elif flx_scl == 1e-03:
        scl_prfx = 'm'
    else :
        scl_prfx = ''

    if dest_dir_plt != None:
        PLOTFILENAME_2DS = str(dest_dir_plt)  + '/' + prefix + (str(Cube2bplot_2D).split('.fits')[0]).split('/')[-1] + '-2DS.pdf'
        PLOTFILENAME_CLP = str(dest_dir_plt)  + '/' + prefix + (str(Cube2bplot_2D).split('.fits')[0]).split('/')[-1] + '-2DC-'+ clp_fnc +'.pdf'
    elif dest_dir_plt == None:
        PLOTFILENAME_2DS = stm_dir_plt        + '/' + prefix + (str(Cube2bplot_2D).split('.fits')[0]).split('/')[-1] + '-2DS.pdf'
        PLOTFILENAME_CLP = stm_dir_plt        + '/' + prefix + (str(Cube2bplot_2D).split('.fits')[0]).split('/')[-1] + '-2DC-'+ clp_fnc +'.pdf'

    if dest_dir_clp != None:
        Cube2bclp_2D_opt = dest_dir_clp + (Cube2bplot_2D.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+clp_fnc+'.fits'
    elif dest_dir_clp == None:
        Cube2bclp_2D_opt = stp_dir_res + (Cube2bplot_2D.split('.fits',1)[0]).rsplit('/',1)[-1] + '-2DC-'+clp_fnc+'.fits'


    data_2b_plt = np.asarray(apgtdt(Cube2bplot_2D,memmap=False) )

    slice_fwhm = (Header_Get(Cube2bplot_2D,'FTS_FWH'))
    slice_cwdt = (Header_Get(Cube2bplot_2D,'STT_VEL')) 
    slice_nmbr = (Header_Get(Cube2bplot_2D,'MAX_SNS')) 

    slice_wdnb = int(np.ceil(slice_fwhm / slice_cwdt))
    slice_nblw = int(slice_nmbr-int(np.ceil(slice_fwhm / slice_cwdt)))
    slice_nbhg = int(slice_nmbr+int(np.ceil(slice_fwhm / slice_cwdt)))


    if (int(slice_nmbr) == slice_nblw) & (int(slice_nmbr)==slice_nbhg):
        data_2b_plt_clp = data_2b_plt[int(slice_nmbr)]
        data_2d_clp     = data_2b_plt_clp
        Message1        = 'Creating collapsed datacube ('+str(clp_fnc)+')'
        Message2        = 'FWHM      : ' + str(slice_fwhm)
        Message3        = 'Channels  : ' + str(slice_nblw)+'-'+str(slice_nbhg)
        Message4        = 'Fits File : ' + Cube2bclp_2D_opt
        print
        print (colored(Message1,'yellow'))
        print (colored(Message2,'yellow'))
        print
    else:
        data_2b_plt_clp = data_2b_plt[slice_nblw:slice_nbhg]
        if clp_fnc == 'sum':
            data_2d_clp = np.asarray(np.nansum(np.array(data_2b_plt_clp)   , axis=0))
            doublecollapsed = np.ravel(data_2d_clp)
            data_2d_clp =  data_2d_clp #*subcube_width 
        elif clp_fnc == 'med':
            data_2d_clp = np.asarray(np.nanmedian(np.array(data_2b_plt_clp), axis=0))
            data_2d_clp =  data_2d_clp #*subcube_width 
            doublecollapsed = np.ravel(data_2d_clp)
            data_2d_clp =  data_2d_clp #*subcube_width 
        elif clp_fnc == 'avg':
            data_2d_clp = np.asarray(np.nanmean(np.array(data_2b_plt_clp)  , axis=0))
            data_2d_clp =  data_2d_clp #*subcube_width 
            doublecollapsed = np.ravel(data_2d_clp)
            data_2d_clp =  data_2d_clp #*subcube_width 

    Wrt_FITS_File(data_2d_clp,Cube2bclp_2D_opt)
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_AVG')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_MED')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_1SL')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_1SH')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_2SL')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_2SH')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_3SL')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_3SH')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_P25')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STZ_P75')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_AVG')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_MED')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_1SL')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_1SH')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_2SL')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_2SH')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_3SL')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_3SH')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_P25')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'STS_P75')
    Header_Copy(Cube2bclp_2D_opt,Cube2bplot_2D,'RESTFRQ')

    F_plt,X_plt,Y_plt = data_2b_plt.shape
    X_clp,Y_clp       = data_2d_clp.shape

    nx_f2DG, ny_f2DG = X_clp,Y_clp
    nx,ny            = nx_f2DG,ny_f2DG

    X0_f2DG     = kwargs.get('X0_f2DG',nx_f2DG/2)
    Y0_f2DG     = kwargs.get('Y0_f2DG',ny_f2DG/2)
    MAX_CTR     = data_2d_clp[X_plt/2,Y_plt/2]

    ##########################################2DS###################################

    fxsize=9
    fysize=8
    f = plt.figure(num=None, figsize=(fxsize, fysize), dpi=180, facecolor='w',
        edgecolor='k')
    plt.subplots_adjust(
        left    = (25/25.4)/fxsize,       #-26 bigger 14-def
        bottom  = (16/25.4)/fysize,       #20 bigger  16-def
        right   = 1 - (34/25.4)/fxsize,   #          20-def 
        top     = 1 - (15/25.4)/fysize)   #          15-def 
    plt.subplots_adjust(hspace=0)#,wspace=0)


    #f.suptitle('An overall title', size=20)
    gs0 = gridspec.GridSpec(1, 1)
    

    gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
        
    ax110 = plt.Subplot(f, gs11[0,0])
    f.add_subplot(ax110)

    ax110.set_rasterization_zorder(1)
    plt.autoscale(enable=True, axis='y', tight=False)
    ax110.xaxis.set_tick_params(labelsize=16)
    ax110.yaxis.set_tick_params(labelsize=16)
    ax110.set_title('Slice number: '+ str(slc_nmb+1)  +  '-' + str(round(VEL_AXS[slc_nmb],0)) + ' km/s'+
                    "\n" +  'X$_{\mathrm{c}}$,Y$_{\mathrm{c}}$: ' + str(+x_ref) + ','+str(+y_ref) + ' $\\varnothing$: '+ str(ap_size) + '"',
                    family='serif',fontsize=16)
    xticklabels = ax110.get_xticklabels()
    plt.setp(xticklabels, visible=True)
    yticklabels = ax110.get_yticklabels()
    plt.setp(yticklabels, visible=True)

    #minorLocator_x   = plt.MultipleLocator(5)
    #majorLocator_x   = plt.MultipleLocator(50)
    #minorLocator_y   = plt.MultipleLocator(0.1)
    #majorLocator_y   = plt.MultipleLocator(0.5)
    #ax110.xaxis.set_minor_locator(minorLocator_x)
    #ax110.xaxis.set_major_locator(majorLocator_x)
    #ax110.yaxis.set_minor_locator(minorLocator_y)
    #ax110.yaxis.set_major_locator(majorLocator_y)
    plt.tick_params(which='both', width=1.0)
    plt.tick_params(which='major', length=10)
    plt.tick_params(which='minor', length=5)
    ax110.minorticks_on()

    #for s in range(len(spectra_plt)):
    plt.xlabel('X',fontsize=16,family = 'serif')
    plt.ylabel('Y',fontsize=16,family = 'serif')
    #plt.ylabel('F$_\lambda$ (Jy)',fontsize=20,family = 'serif')
    #plt.ylabel('Intensity',fontsize=20,family = 'serif')
    #lambda_sp,inten_sp,crval_sp,cdel1_sp
    if ('_ms.' in Cube2bplot_2D) or ('dta_in.' in Cube2bplot_2D) or ('dta_ot.' in Cube2bplot_2D):
        tick_color = 'white'
    elif ('msk_in.' in Cube2bplot_2D) or ('crc.' in Cube2bplot_2D) or ('msk_ot.' in Cube2bplot_2D):
        tick_color = 'black'
    else:
        tick_color = 'white'

    ax110.xaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)
    ax110.yaxis.set_tick_params(which='both',labelsize=20,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)

    plt.imshow(data_2b_plt[slc_nmb]/flx_scl, origin='lower',cmap='viridis')#, vmin=2.e3, vmax=3.e3)
    divider = make_axes_locatable(ax110)
    cax  = divider.append_axes("right", size="5%", pad=0.05)    
    #cbar = plt.colorbar(format=mpl.ticker.FuncFormatter(fmt),cax=cax)
    cbar = plt.colorbar(cax=cax,pad=0.5)
    cbar.set_label('S ['+scl_prfx+'Jy]', rotation=270,family = 'serif',size=16,labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_yticklabels(["{:.0f}".format(i) for i in cbar.get_ticks()]) # set ticks of your format

    #ax110.xaxis.set_major_locator(ScaledLocator(dx=1.0,x0=-X0_f2DG+x_ref))
    #ax110.yaxis.set_major_locator(ScaledLocator(dx=1.0,x0=-Y0_f2DG+y_ref))
    ax110.xaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-X0_f2DG+x_ref))
    ax110.yaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-Y0_f2DG+y_ref))

    plt.scatter(X0_f2DG, Y0_f2DG, s=25, c='white', marker='x')
    #plt.scatter(X0_f2DG, Y0_f2DG, s=25, c='black',marker='+')

    #plt.xlim([lambda_min,lambda_max])
    #xmin, xmax = plt.xlim()
    #plt.xlim((xmin,xmax))
    #if autoaxis == False:
        #nmin  = int(abs(lambda_min - crval_sp) / cdel1_sp)
        #nmax  = int(abs(lambda_max - crval_sp) / cdel1_sp)
        #min_y = bn.nanmin(wght_times*inten_sp[mask])#min(wght_times*inten_sp[nmin:nmax])
        #max_y = bn.nanmax(wght_times*inten_sp[mask])#max(wght_times*inten_sp[nmin:nmax])
        #plt.ylim([min_y,max_y])
        #ymin, ymax = plt.ylim()
        #plt.ylim((ymin,ymax))
    #elif autoaxis == True:
        #min_y, max_y = ax110.get_ylim()
    
    plt.savefig(PLOTFILENAME_2DS)


    Message3       = 'Generated plot file for datacubes slice ('+str(slc_nmb+1)+')'
    Message4       = PLOTFILENAME_2DS
    print
    print (colored(Message3,'cyan'))
    print (colored(Message4,'cyan'))
    print
    #plt.show()

    ##########################################2DS###################################

    ##########################################CLP###################################

    f = plt.figure(num=None, figsize=(fxsize, fysize), dpi=180, facecolor='w',
        edgecolor='k')
    plt.subplots_adjust(
        left    = (25/25.4)/fxsize,       #-26 bigger 14-def
        bottom  = (16/25.4)/fysize,       #20 bigger  16-def
        right   = 1 - (34/25.4)/fxsize,   #          20-def 
        top     = 1 - (15/25.4)/fysize)   #          15-def 
    plt.subplots_adjust(hspace=0)#,wspace=0)


    #f.suptitle('An overall title', size=20)
    gs0 = gridspec.GridSpec(1, 1)
    

    gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
        
    ax110 = plt.Subplot(f, gs11[0,0])
    f.add_subplot(ax110)

    ax110.set_rasterization_zorder(1)
    plt.autoscale(enable=True, axis='y', tight=False)
    ax110.xaxis.set_tick_params(labelsize=16)
    ax110.yaxis.set_tick_params(labelsize=16)
    ax110.set_title('2D Collapse [' +str(slice_nblw+1) + ':'+ str(slice_nbhg+1)+ ']: ' + str(clp_fnc.upper()) +
                    "\n" +  'X$_{\mathrm{c}}$,Y$_{\mathrm{c}}$: ' + str(+x_ref) + ','+str(+y_ref) + ' $\\varnothing$: '+ str(ap_size) + '"',
                    family='serif',fontsize=16)

    xticklabels = ax110.get_xticklabels()
    plt.setp(xticklabels, visible=True)
    yticklabels = ax110.get_yticklabels()
    plt.setp(yticklabels, visible=True)

    #minorLocator_x   = plt.MultipleLocator(5)
    #majorLocator_x   = plt.MultipleLocator(50)
    #minorLocator_y   = plt.MultipleLocator(0.1)
    #majorLocator_y   = plt.MultipleLocator(0.5)
    #ax110.xaxis.set_minor_locator(minorLocator_x)
    #ax110.xaxis.set_major_locator(majorLocator_x)
    ##ax110.yaxis.set_minor_locator(minorLocator_y)
    #ax110.yaxis.set_major_locator(majorLocator_y)
    plt.tick_params(which='both', width=1.0)
    plt.tick_params(which='major', length=10)
    plt.tick_params(which='minor', length=5)
    ax110.minorticks_on()

    ax110.xaxis.set_tick_params(which='both',labelsize=16,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)
    ax110.yaxis.set_tick_params(which='both',labelsize=16,direction='in',color=tick_color,bottom=True,top=True,left=True,right=True)

    #for s in range(len(spectra_plt)):
    plt.xlabel('X',fontsize=16,family = 'serif')
    plt.ylabel('Y',fontsize=16,family = 'serif')
    #plt.ylabel('F$_\lambda$ (Jy)',fontsize=20,family = 'serif')
    #plt.ylabel('Intensity',fontsize=20,family = 'serif')
    #lambda_sp,inten_sp,crval_sp,cdel1_sp
    
    plt.imshow(data_2d_clp/flx_scl, origin='lower',cmap='viridis')#, vmin=2.e3, vmax=3.e3)
    divider = make_axes_locatable(ax110)
    cax  = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(format=mpl.ticker.FuncFormatter(fmt),cax=cax)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('S ['+scl_prfx+'Jy]', rotation=270,family = 'serif',size=16,labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_yticklabels(["{:.0f}".format(i) for i in cbar.get_ticks()]) # set ticks of your format

    #ax110.xaxis.set_major_locator(ScaledLocator(dx=1.0,x0=-X0_f2DG+x_ref))
    #ax110.yaxis.set_major_locator(ScaledLocator(dx=1.0,x0=-Y0_f2DG+y_ref))
    ax110.xaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-X0_f2DG+x_ref))
    ax110.yaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-Y0_f2DG+y_ref))

    plt.scatter(X0_f2DG, Y0_f2DG, s=25, c='white', marker='x')
    #plt.scatter(X0_f2DG, Y0_f2DG, s=25, c='black',marker='+')

    #plt.xlim([lambda_min,lambda_max])
    #xmin, xmax = plt.xlim()
    #plt.xlim((xmin,xmax))
    #if autoaxis == False:
        #nmin  = int(abs(lambda_min - crval_sp) / cdel1_sp)
        #nmax  = int(abs(lambda_max - crval_sp) / cdel1_sp)
        #min_y = bn.nanmin(wght_times*inten_sp[mask])#min(wght_times*inten_sp[nmin:nmax])
        #max_y = bn.nanmax(wght_times*inten_sp[mask])#max(wght_times*inten_sp[nmin:nmax])
        #plt.ylim([min_y,max_y])
        #ymin, ymax = plt.ylim()
        #plt.ylim((ymin,ymax))
    #elif autoaxis == True:
        #min_y, max_y = ax110.get_ylim()

    #plt.show()

    plt.savefig(PLOTFILENAME_CLP)
    Message5       = 'Generated plot file for collapsed datacube ('+str(clp_fnc)+')'
    Message6       = PLOTFILENAME_CLP
    print
    print (colored(Message5,'cyan'))
    print (colored(Message6,'cyan'))
    print

    ##########################################CLP###################################
    if verbose == True:
        print
        print (colored('Generated Fits: ' + str(Cube2bclp_2D_opt) + ' Dim: ' + str(F_plt) + ' X ' + str(X_plt) + ' X ' + str(Y_plt) ,'yellow'))
        print (colored('Generated Plot: ' + str(PLOTFILENAME_2DS) + ' Dim: ' + str(F_plt) + ' X ' + str(X_plt) + ' X ' + str(Y_plt) ,'cyan'))
        print (colored('Generated Plot: ' + str(PLOTFILENAME_CLP) + ' Dim: ' + str(X_clp) + ' X ' + str(Y_clp) ,'cyan'))
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

def Plot_Cube_3D(Cube2bplot_3D,*args,**kwargs):
    dest_dir  = kwargs.get('dest_dir', None)
    autoaxis  = kwargs.get('autoaxis', False)
    verbose   = kwargs.get('verbose' , False)
    epssave   = kwargs.get('epssave' , False)
    showplot  = kwargs.get('showplot', False)
    slc_nmb   = kwargs.get('slc_nmb' , None) 
    clp_fnc   = kwargs.get('clp_fnc' ,'sum')

    redshift  = kwargs.get('redshift' ,'1')
    rst_frq   = kwargs.get('rst_frq'  ,'1')

    x_ref    = kwargs.get('x_ref',0)
    y_ref    = kwargs.get('y_ref',0)
    ap_size  = kwargs.get('ap_size',0)


    z_avg     = kwargs.get('z_avg',Header_Get(Cube2bplot_3D,'STZ_AVG'))
    z_med     = kwargs.get('z_med',Header_Get(Cube2bplot_3D,'STZ_MED'))
    frq_r     = kwargs.get('frq_r',Header_Get(Cube2bplot_3D,'RESTFRQ'))
    z_f2l     = z_med

    cube_data     = np.asarray(apgtdt(Cube2bplot_3D,memmap=False) )

    Cube_Info = Cube_Header_Get(Cube2bplot_3D,frq_r* u.Hz)
    FRQ_AXS   = Cube_Info[16].value
    VEL_AXS   = Cube_Info[17].value

    if slc_nmb != None:
        data_2b_plot = cube_data[slc_nmb]
        doublecollapsed = np.ravel(data_2b_plot)
        PlotTitle = 'Slice number: '+ str(slc_nmb+1) + '-' +str(round(VEL_AXS[slc_nmb],0)) + ' km/s'
        PLOTFILENAME = (str(Cube2bplot_3D).split('.fits')[0]).split('/')[-1] + '-3D-slc-'+str(slc_nmb+1)+'.pdf'

        if dest_dir != None:
            PLOTFILENAME = str(dest_dir)  + '/' + PLOTFILENAME
        elif dest_dir == None:
            PLOTFILENAME = plt_dir_res    + '/' + PLOTFILENAME
        Message = 'Generated Plot: ' + PLOTFILENAME + ' slice number : ' + str(slc_nmb+1)

    elif slc_nmb == None:
        PlotTitle = 'Collapse: ' + str(clp_fnc.upper())
        PLOTFILENAME = (str(Cube2bplot_3D).split('.fits')[0]).split('/')[-1] + '-3D-'+str(clp_fnc)+'.pdf'
        if dest_dir != None:
            PLOTFILENAME = str(dest_dir)  + '/' + PLOTFILENAME
        elif dest_dir == None:
            PLOTFILENAME = plt_dir_res    + '/' + PLOTFILENAME
        Message = 'Generated Plot: ' + PLOTFILENAME + ' collapse ('+str(clp_fnc)+')'

        #data_2b_plt = np.asarray(apgtdt(Cube2bFit,memmap=False) )
        if clp_fnc == 'sum':
            cube_data_clp   = np.asarray(np.nansum(np.array(cube_data)   , axis=0))  #np.nansum(np.array(img_stat)   , axis=0) #
            cube_data_clp   = cube_data_clp #*subcube_width
            data_2b_plot    = cube_data_clp
        elif clp_fnc == 'med':
            cube_data_clp   = np.asarray(np.nanmedian(np.array(cube_data), axis=0))  #np.nansum(np.array(img_stat)   , axis=0) #
            cube_data_clp   = cube_data_clp #*subcube_width
            data_2b_plot    = cube_data_clp
        elif clp_fnc == 'avg':
            cube_data_clp   = np.asarray(np.nanmean(np.array(cube_data)  , axis=0))  #np.nansum(np.array(img_stat)   , axis=0) #
            cube_data_clp   = cube_data_clp #*subcube_width
            data_2b_plot    = cube_data_clp
    freq_num,y_num,x_num = cube_data.shape
    x    = np.arange(0,x_num,1)
    y    = np.arange(0,y_num,1)
    x, y = np.meshgrid(x,y)
    z    = data_2b_plot#cube_data[10]#

    nx_f2DG, ny_f2DG = x_num,y_num #data_2b_plot.shape
    nx,ny            = nx_f2DG,ny_f2DG

    X0_f2DG     = kwargs.get('X0_f2DG',int(np.ceil(nx_f2DG/2)))
    Y0_f2DG     = kwargs.get('Y0_f2DG',int(np.ceil(ny_f2DG/2)))

    MAX_CTR     = data_2b_plot[x_num/2,y_num/2]

    fxsize=9
    fysize=8
    f = plt.figure(num=None, figsize=(fxsize, fysize), dpi=180, facecolor='w',
        edgecolor='k')
    plt.subplots_adjust(
        left    = (16/25.4)/fxsize, 
        bottom  = (12/25.4)/fysize, 
        right   = 1 - (6/25.4)/fxsize, 
        top     = 1 - (15/25.4)/fysize)
    plt.subplots_adjust(hspace=0)#,wspace=0)


    #f.suptitle('An overall title', size=20)
    #fig = plt.figure()
    ax110 = f.gca(projection='3d')

    ax110.set_rasterization_zorder(1)
    #plt.autoscale(enable=True, axis='y', tight=False)
        

    plt.title(PlotTitle + ' (' +  str(+x_ref) + ','+str(+y_ref)+') ' + ' $\\varnothing$: '+ str(ap_size) + '"')

    ax110.set_xlabel('X',family = 'serif')
    ax110.set_ylabel('Y',family = 'serif')
    ax110.set_zlabel('S [Jy]',family = 'serif')

    surf = ax110.plot_surface(x,y,z, cmap=cm.viridis,rstride=1, cstride=1,
        linewidth=0, antialiased=False,vmin=np.nanmin(z), vmax=np.nanmax(z))
    divider = make_axes_locatable(ax110)
    cax  = divider.append_axes("right", size="5%", pad=0.05)    
    #cbar = plt.colorbar(format=mpl.ticker.FuncFormatter(fmt),cax=cax)
    cbar = plt.colorbar(cax=cax)
    f.colorbar(surf, shrink=0.25, pad=0.05,aspect=5,format=mpl.ticker.FuncFormatter(fmt))

    #ax110.xaxis.set_major_locator(ScaledLocator(dx=1.0,x0=-X0_f2DG+x_ref))
    #ax110.yaxis.set_major_locator(ScaledLocator(dx=1.0,x0=-Y0_f2DG+y_ref))
    ax110.xaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-X0_f2DG+x_ref))
    ax110.yaxis.set_major_formatter(ScaledFormatter(dx=1.0,x0=-Y0_f2DG+y_ref))

    #cbar = plt.colorbar(format=mpl.ticker.FuncFormatter(fmt))
    ax110.zaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    #ax110.set_zticks([])


    
    plt.savefig(PLOTFILENAME)

    if verbose == True:
        print
        print (colored(Message,'cyan'))
    elif verbose ==False:
        pass
    if epssave == True:
        plt.savefig('Spectra.eps', rasterized=True)
        #os.system('open Spectra.eps')
    elif epssave == False:
        pass
    if showplot == True:
        plt.show()
        #os.system('open '+str(PLOTFILENAME))
        pass
    elif showplot == False:
        pass    
    plt.close('all')

def Cube_Header_Get(cube_header_ipt,freq_rfr,*args, **kwargs):
    verbose    = kwargs.get('verbose',False) 
    redshift   = kwargs.get('redshift',0)
    freq_step  = kwargs.get('freq_step',Header_Get(cube_header_ipt,'CDELT3'))
    freq_init  = kwargs.get('freq_init',Header_Get(cube_header_ipt,'CRVAL3'))
    freq_obs_f = kwargs.get('freq_obs_f',Header_Get(cube_header_ipt,'RESTFRQ'))#freq_obs
    freq_obs   = kwargs.get('freq_obs',Redshifted_freq(freq_rfr,redshift))    #freq_obs_f 

    #print
    #print freq_obs_f
    #print freq_obs
    #print

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
        print ('Dimensions                                   : ',DIM1,'X',DIM2,'X',DIM3)#len(cube.spectral_axis)
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
####Fnc_Syn_Plt####