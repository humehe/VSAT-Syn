# VSAT-Syn
Valparaíso Stacking Analysis Tool Synthetic Datacubes


VSAT-Syn is part of the Valparaíso Stacking Analysis Tool (VSAT), it provide a series of tools to generate synthetic datacubes emulating image datacubes coming from interferometric datasets. Although VSAT-Syn was designed to estimate the systematic flux measurement errors by mimicking the images generated from interferometric _uv_ datasets. VSAT-Syn can also be used to simulate _moment-0_ 2D images, by simple assuming single-channels datacubes. The generated synthetic fits files can then be used to generate composite images (2D/3D) and measure their flux to quantify systematic flux measurement errors computed with VSAT-3D and VSAT-2D .

![Alt text](./Figures-Syn/Synthetic-InOut-Stats-SNR-BIS.jpg?raw=true "3D datacube Stacked spectra Scheme.")

## Content

1. Fnc_Syn_Dir.py:
   - Location of the input catalogue and spectral data. 
   - Parameters for generating synthetic datacubes
   - Location of the resulting products of the stacking analyses _e.g. stamps, tables, plots,and stacked spectra_.

2. Fnc_Syn_Mth.py:
   - Math functions (e.g. cosmological constants, gaussian profiles for line fitting) needed throughout the stacking analysis.

3. Fnc_Syn_Spc.py 
   - Tools to modify datacubes including _e.g. masking, adding missing frequencies, eextract regions etc_

4. Fnc_Syn_Stt.py 
   - Statistical funtions for datacubes.

5. Fnc_Syn_Syn.py
   - Core of the synthetic datacubes tool.

6. Fnc_Syn_Stk.py
   - Core of the 3D stacking tool.

7. Fnc_Syn_Fts.py
   - Funtions to access and modify (add, modify, delete) fits headers

8. Fnc_Syn_Tbl.py
   - Functions to read, write and modify different tables.
 
 9. Fnc_Syn_Utl.py
   - Auxiliary functions for the stacking analysis.

## Image size
```nx```, ```ny``` and ``nchan`` define the dimensions of the datacube, while ```scale```defiines the arcsec/pixel scale.

## Noise parameters
```amp_noise``` defines the noise amplitude.

## Spectral Gaussian
If ```fixed_amp_ns=True``` the spectal gaussian amplitude will be fixed, ```fixed_width_str=True``` fixes the spectral width for the datacubes.
```random_choice_type```defines the assumed random distrubution:  ```uniform,gauss```. The limits for the generated source are defined by: ```A_min_1dg``` and ```A_max_1dg```define the amplitude limits, ```ofs_min_1dg```  and ```ofs_max_1dg```  define the offset limiits , 
```sigma_min_1dg``` and ```sigma_max_1dg``` define the channel width limits.
```n_noise_min``` and ```sigma_max_1dgn_noise_max```    defines the source amplitude in terms of the noise level _i.e. A = n X times noise_.

## Source 2D parameters
The number of generated sources is defined by ```n```, ```A``` define the amplitude limits, ```ofs_min,ofs_max``` define the spatial offset,
```sigmax_min,sigmax_max,sigmay_min,sigmay_max```  define the source size (_x,y_) limits. If ```starshape = 'circular' ```  these ```x,y```will be identical.

## PSF 2D
```fwhm_2d``` define the spatial fwhm.

## Example

The Example.py script contains an example to generate synthetic datacubes with dimensions 256X256X17 per iteration. These synthetic datacubes (_N=5_) are stacked to then measure their flux as exemplified in VSAT-3D. Then by simple running ```python Example.py``` will complete all the following steps below. The following  snippets are extracts contained in the Example.py file and will guide you through the file. To run the script a input table catalogue (```CII_Sources_HATLAS-13CO-RDS_B-0.csv```)is needed, and is available in the _Example-VSAT-Syn.zip_ zip file. 

###### "Synthetic datacubes"
The following snippet will generate 27 datacubes with the following dimensions _256 X 256 X 17_.

```python
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
						dst_img_dir       = img_dir_res)
```

###### "Stacking"
```python
Stack_Res      = Cube_Stack(cubetoread,stk_ofn_prfx,weights,
			sig_clp     = False,sufix=channel_width,freq_obs_f=restframe_frequency,
			stack_lite  = stack_light,
			cp_bs_hdrs  = False,
			stt_var     = True,
			spc_wdt_dir = channel_width,
			stt_mst_tbl = Cat_Ipt_Tbl      , stt_hdr='RDS_B',
			stt_syn     = True         , stt_syn_tbl = cat_tbl_stk)
```


VSAT-3D can then be used to measure the line flux emission of these synthetic datacubes. By fixing _nz=1_ it possible to generate synthetic _moment-0_ 2D images, and then measure their fluxes with VSAT-2D.

## Dependencies
## Dependencies
Currently VSAT works only with astropy 2.0 and python 2.X. However a new version compatible wiith python 3.X and more recent astropy versions will be soon released dropping this dependency.

 - [astropy](https://www.astropy.org)
 - [bottleneck](https://pypi.org/project/Bottleneck/)
 - [pandas](https://pandas.pydata.org)
 - [scipy](https://www.scipy.org)
 - [numpy](https://numpy.org)
 - [lmfit](https://lmfit.github.io/lmfit-py/)
 - [matplotlib](https://matplotlib.org)
 - [termcolor](https://pypi.org/project/termcolor/)
 - [progressbar](https://pypi.org/project/progressbar2/)
## License

BSD 3-Clause License

Copyright (c) 2021, VSAT-1D developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
