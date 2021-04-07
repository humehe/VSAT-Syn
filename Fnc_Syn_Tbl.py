import pandas as pd
from astropy import table as aptbl

from Fnc_Syn_Dir import *

####Fnc_Stk_Tbl####
def split_variable_vars(split_variable):
	if split_variable       == 'RDS' or split_variable == 'RDS_B':
		Splt_Col     = 8
		Splt_Hdr_Cmt = 'Redshift'
		Splt_CNm     = 8
		Splt_Hdr     = 'RDS'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'z'
	elif split_variable     == 'STM' or split_variable == 'STM_B':
		Splt_Col     = 24
		Splt_Hdr_Cmt = 'Stellar Mass [log(M_*/M_sun)]'
		Splt_CNm     = 24
		Splt_Hdr     = 'STM'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'log[$M/M_{\odot}$]'
	elif split_variable     == 'SFR' or split_variable == 'SFR_B':
		Splt_Col     = 22
		Splt_Hdr_Cmt = 'SFR [M_sun/yr]'
		Splt_CNm     = 22
		Splt_Hdr     = 'SFR'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'SFR'
	elif split_variable     == 'LCO' or split_variable == 'LCO_B':
		Splt_Col     = 35
		Splt_Hdr_Cmt = 'CO Lum [K km/s/pc2]'
		Splt_CNm     = 35
		Splt_Hdr     = 'LCO'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'L$_{CO}$'
	elif split_variable     == 'sSF' or split_variable == 'sSF_B':
		Splt_Col     = 26
		Splt_Hdr_Cmt = 'Specific SFR [1/Gyr]'
		Splt_CNm     = 26
		Splt_Hdr     = 'sSF'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'sSFR'
	elif split_variable     == 'MH2' or split_variable == 'MH2_B':
		Splt_Col     = 37
		Splt_Hdr_Cmt = 'H2 mass [log(M_H2/M_sun)]'
		Splt_CNm     = 37
		Splt_Hdr     = 'MH2'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'M$_{H2}$'
	elif split_variable     == 'SFE' or split_variable == 'SFE_B':
		Splt_Col     = 41
		Splt_Hdr_Cmt = 'SFE [1/Gyr]'
		Splt_CNm     = 41
		Splt_Hdr     = 'SFE'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'SFE'
	elif split_variable     == 'LIR' or split_variable == 'LIR_B':
		Splt_Col     = 20
		Splt_Hdr_Cmt = 'LIR [log(L_IR/L_sun)]'
		Splt_CNm     = 20
		Splt_Hdr     = 'LIR'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'log[L$_{IR}$/L${_sun}$]'
	elif split_variable     == 'LFIR' or split_variable == 'LFIR_B':
		Splt_Col     = 11
		Splt_Hdr_Cmt = 'LFIR [log(Lfir/Lo)]'
		Splt_CNm     = 11
		Splt_Hdr     = 'LFIR'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'LFIR [log(Lfir/Lo)]'
	elif split_variable     == 'SDG' or split_variable == 'SDG_B':
		Splt_Col     = 43
		Splt_Hdr_Cmt = 'Surf Dens Gas [log(M_sun/pc2)]'
		Splt_CNm     = 43
		Splt_Hdr     = 'SDG'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = '$\Sigma_{Gas}$'
	elif split_variable     == 'SDS' or split_variable == 'SDS_B':
		Splt_Col     = 45
		Splt_Hdr_Cmt = 'Surf Dens SFR [log(M_sun/yr/kpc2)]'
		Splt_CNm     = 45
		Splt_Hdr     = 'SDS'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = '$\Sigma_{SFR}$'
	elif split_variable     == 'TDT' or split_variable == 'TDT_B':
		Splt_Col     = 47
		Splt_Hdr_Cmt = 'Depletion Time [Gyr]'
		Splt_CNm     = 47
		Splt_Hdr     = 'TDT'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = '$\\tau$'
	elif split_variable       == 'MRP' or split_variable == 'MRP_B':
		Splt_Col     = 50
		Splt_Hdr_Cmt = 'Morphology'	
		Splt_CNm     = 50
		Splt_Hdr     = 'MRP'
		Splt_Hdr_Plt = split_variable
		Splt_Plt_lbl = 'Morphology'
	else:
		print
		print ('split_variable_vars')
		print (colored('Variable '+ str(split_variable) + ' does not exist!','yellow'))
		print
		quit()
	return Splt_Col,Splt_Hdr,Splt_Hdr_Cmt,Splt_CNm,Splt_Hdr_Plt,Splt_Plt_lbl

def Table_Read(table_name,format_tbl,*args, **kwargs):
	verbose = kwargs.get('verbose', False)
	if verbose == True:
		print table_name
	elif verbose == False:
		pass
	ftbl = aptbl.Table.read(table_name, format=format_tbl)
	c1   = ftbl['ID']
	c2   = ftbl['fits']
	c3   = ftbl['Source']
	c4   = ftbl['Delta_nu']
	c5   = ftbl['RMS']
	c6   = ftbl['SPW']
	c7   = ftbl['State']
	c8   = ftbl['z_1']
	c9   = ftbl['RA']
	c10  = ftbl['Dec']
	c11  = ftbl['log(Lfir/Lo)']
	c12  = ftbl['D_log(Lfir/Lo)']
	c13  = ftbl['nu_obs']
	c14  = ftbl['V_obs']
	c15  = ftbl['GAMA_ID']
	c16  = ftbl['SOURCE']
	c17  = ftbl['RAJ2000']
	c18  = ftbl['DECJ2000']
	c19  = ftbl['z_2']
	c20  = ftbl['log[L_IR/L_sun]']
	c21  = ftbl['c7_err']
	c22  = ftbl['SFR']
	c23  = ftbl['c8_err']
	c24  = ftbl['log[M_S/M_sun]']
	c25  = ftbl['c9_err']
	c26  = ftbl['sSFR']
	c27  = ftbl['c10_err']
	c28  = ftbl['nu_ob_1']
	c29  = ftbl['nu_ob_2']
	c30  = ftbl['c11_err']
	c31  = ftbl['v_fwhm']#
	c32  = ftbl['c12_err']
	c33  = ftbl['S_COXDeltaV']
	c34  = ftbl['c13_err']
	c35  = ftbl['L_CO']
	c36  = ftbl['c14_err']
	c37  = ftbl['log[M_H2/M_sun]']
	c38  = ftbl['c15_err']
	c39  = ftbl['R_FWHM']
	c40  = ftbl['c16_err']
	c41  = ftbl['SFE']
	c42  = ftbl['c17_err']
	c43  = ftbl['SIGMA_gas']
	c44  = ftbl['c18_err']
	c45  = ftbl['SIGMA_SFR']
	c46  = ftbl['c19_err']
	c47  = ftbl['Tau_gas']
	c48  = ftbl['c20_err']
	c49  = ftbl['M']
	return(ftbl,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,
		c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
		c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,
		c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,
		c41,c42,c43,c44,c45,c46,c47,c48,c49)

def Table_Read_Opt(table_name,format_tbl,*args, **kwargs):
	ftbl = aptbl.Table.read(table_name, format=format_tbl)
	c1   = ftbl['AMS_AVG_IPT']
	c2   = ftbl['AMN_AVG_IPT']
	c3   = ftbl['SNR_AVG_IPT']
	c4   = ftbl['SGN_AVG_IPT']
	c5   = ftbl['SGV_AVG_IPT']
	c6   = ftbl['FWH_AVG_IPT']
	c7   = ftbl['OFS_AVG_IPT']
	c8   = ftbl['1DG_AAM_OPT']
	c9   = ftbl['1DG_ACT_OPT']
	c10  = ftbl['1DG_ASI_OPT']
	c11  = ftbl['1DG_AFW_OPT']
	c12  = ftbl['S2G_XCT_OPT']
	c13  = ftbl['S2G_YCT_OPT']
	c14  = ftbl['S2G_AMP_OPT']
	c15  = ftbl['S2G_SGX_OPT']
	c16  = ftbl['S2G_SGY_OPT']
	c17  = ftbl['S2G_THT_OPT']
	c18  = ftbl['S2G_OFS_OPT']
	c19  = ftbl['S2G_FLS_OPT']
	c20  = ftbl['S2G_FT1_OPT']
	c21  = ftbl['S2G_FT2_OPT']
	c22  = ftbl['OPT/IPT_R_S_1']
	c23  = ftbl['OPT/IPT_A_1']
	c24  = ftbl['OPT/IPT_A_2']
	c25  = ftbl['OPT/IPT_A_3']
	c26  = ftbl['IPT/OPT_R_S_1']
	c27  = ftbl['IPT/OPT_R_A_1']
	c28  = ftbl['IPT/OPT_R_A_2']
	c29  = ftbl['IPT/OPT_R_A_3']
	try:
		c30 = ftbl['OPT/IPT_A_4']
		c31 = ftbl['IPT/OPT_R_A_4']
		c32 = ftbl['IPT/OPT_R_S2G']
		c33 = ftbl['CH2_1SG']
		c34 = ftbl['CH2_2SG']
		c35 = ftbl['CH2_AMP']
		c36 = ftbl['CH2_FLX']
		return(ftbl,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,
		c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
		c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,
		c31,c32,c33,c34,c35,c36)
	except KeyError:
		pass

	try:
		c30 = ftbl['OPT/IPT_A_4']
		c31 = ftbl['IPT/OPT_R_A_4']
		c32 = ftbl['IPT/OPT_R_S2G']
		c33 = ftbl['CH2_1SG_OPT']
		c34 = ftbl['CHR_1SG_OPT']
		c35 = ftbl['CH2_2SG_OPT']
		c36 = ftbl['CHR_2SG_OPT']
		return(ftbl,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,
		c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
		c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,
		c31,c32,c33,c34,c35,c36)
	except KeyError:
		return(ftbl,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,
		c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
		c21,c22,c23,c24,c25,c26,c27,c28,c29)

def Table_Read_Opt_1(table_name,format_tbl,*args, **kwargs):
	ftbl = aptbl.Table.read(table_name, format=format_tbl)
	c1  = ftbl['AMS_AVG_IPT']
	c2  = ftbl['AMN_AVG_IPT']
	c3  = ftbl['SNR_AVG_IPT']
	c4  = ftbl['AMP_IP1_CRT']
	c5  = ftbl['AMP_IP2_CRT']
	c6  = ftbl['SGN_AVG_IPT']
	c7  = ftbl['SGV_AVG_IPT']
	c8  = ftbl['FWH_AVG_IPT']
	c9  = ftbl['OFS_AVG_IPT']
	c10 = ftbl['S2G_FLS_IPT']
	c11 = ftbl['S2G_IPT_CRT']
	c12 = ftbl['C2G_FLS_IPT']
	c13 = ftbl['C2G_IPT_CRT']
	c14 = ftbl['1DG_SAM_OPT']
	c15 = ftbl['1DG_SCT_OPT']
	c16 = ftbl['1DG_SSI_OPT']
	c17 = ftbl['1DG_SFW_OPT']
	c18 = ftbl['1DG_SAE_OPT']
	c19 = ftbl['1DG_SCE_OPT']
	c20 = ftbl['1DG_SSE_OPT']
	c21 = ftbl['1DG_AAM_OPT']
	c22 = ftbl['1DG_ACT_OPT']
	c23 = ftbl['1DG_ASI_OPT']
	c24 = ftbl['1DG_AFW_OPT']
	c25 = ftbl['1DG_AAE_OPT']
	c26 = ftbl['1DG_ACE_OPT']
	c27 = ftbl['1DG_ASE_OPT']
	c28 = ftbl['CH2_1SG_OPT']
	c29 = ftbl['CHR_1SG_OPT']
	c30 = ftbl['CBE_BIN_OPT']
	c31 = ftbl['S2G_XCT_OPT']
	c32 = ftbl['S2G_YCT_OPT']
	c33 = ftbl['S2G_AMP_OPT']
	c34 = ftbl['S2G_SGX_OPT']
	c35 = ftbl['S2G_SGY_OPT']
	c36 = ftbl['S2G_THT_OPT']
	c37 = ftbl['S2G_OFS_OPT']
	c38 = ftbl['S2G_FLS_OPT']
	c39 = ftbl['S2G_XCE_OPT']
	c40 = ftbl['S2G_YCE_OPT']
	c41 = ftbl['S2G_AME_OPT']
	c42 = ftbl['S2G_SXE_OPT']
	c43 = ftbl['S2G_SYE_OPT']
	c44 = ftbl['S2G_THE_OPT']
	c45 = ftbl['S2G_OFE_OPT']
	c46 = ftbl['S2G_FT1_OPT']
	c47 = ftbl['S2G_FT2_OPT']
	c48 = ftbl['S2G_CNN_OPT']
	c49 = ftbl['CH2_2SG_OPT']
	c50 = ftbl['CHR_2SG_OPT']
	c51 = ftbl['CHR_2SA_OPT']
	c52 = ftbl['CHR_2SS_OPT']
	c53 = ftbl['C2G_XCT_OPT']
	c54 = ftbl['C2G_YCT_OPT']
	c55 = ftbl['C2G_AMP_OPT']
	c56 = ftbl['C2G_SGX_OPT']
	c57 = ftbl['C2G_SGY_OPT']
	c58 = ftbl['C2G_THT_OPT']
	c59 = ftbl['C2G_OFS_OPT']
	c60 = ftbl['C2G_FLS_OPT']
	c61 = ftbl['C2G_XCE_OPT']
	c62 = ftbl['C2G_YCE_OPT']
	c63 = ftbl['C2G_AME_OPT']
	c64 = ftbl['C2G_SXE_OPT']
	c65 = ftbl['C2G_SYE_OPT']
	c66 = ftbl['C2G_THE_OPT']
	c67 = ftbl['C2G_OFE_OPT']
	c68 = ftbl['C2G_FT1_OPT']
	c69 = ftbl['C2G_FT2_OPT']
	c70 = ftbl['C2G_CNN_OPT']
	c71 = ftbl['CH2_CSG_OPT']
	c72 = ftbl['CHR_CSG_OPT']
	c73 = ftbl['CHR_CSA_OPT']
	c74 = ftbl['CHR_CSS_OPT']
	c75 = ftbl['OPT/IPT_R_S_1']
	c76 = ftbl['OPT/IPT_A']
	c77 = ftbl['OPT/IPT_A_CRT']
	c78 = ftbl['IPT/OPT_R_S_1']
	c79 = ftbl['IPT/OPT_R_A']
	c80 = ftbl['IPT/OPT_R_A_CRT']
	c81 = ftbl['IPT/OPT_R_S2G_0']
	c82 = ftbl['IPT/OPT_R_S2G_1']
	c83 = ftbl['IPT/OPT_R_S2G_2']
	c84 = ftbl['IPT/OPT_R_S2G_3']
	c85 = ftbl['IPT/OPT_R_S2G_0_CRT']
	c86 = ftbl['IPT/OPT_R_S2G_1_CRT']
	c87 = ftbl['IPT/OPT_R_S2G_2_CRT']
	c88 = ftbl['IPT/OPT_R_S2G_3_CRT']
	c89 = ftbl['OPT/IPT_A_CTR']
	c90 = ftbl['OPT/IPT_A_CTR_CRT']
	c91 = ftbl['IPT/OPT_R_A_CTR']
	c92 = ftbl['IPT/OPT_R_A_CTR_CRT']
	c93 = ftbl['IPT/OPT_R_C2G_0']
	c94 = ftbl['IPT/OPT_R_C2G_1']
	c95 = ftbl['IPT/OPT_R_C2G_2']
	c96 = ftbl['IPT/OPT_R_C2G_3']
	c97 = ftbl['IPT/OPT_R_C2G_0_CRT']
	c98 = ftbl['IPT/OPT_R_C2G_1_CRT']
	c99 = ftbl['IPT/OPT_R_C2G_2_CRT']
	c100= ftbl['IPT/OPT_R_C2G_3_CRT']
	return(ftbl,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,
	c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
	c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,
	c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,
	c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,
	c51,c52,c53,c54,c55,c56,c57,c58,c59,c60,
	c61,c62,c63,c64,c65,c66,c67,c68,c69,c70,
	c71,c72,c73,c74,c75,c76,c77,c78,c79,c80,
	c81,c82,c83,c84,c85,c86,c87,c88,c89,c90,
	c91,c92,c93,c94,c95,c96,c97,c98,c99,c100)

def Table_Read_Opt_2(table_name,format_tbl,*args, **kwargs):
	ftbl = aptbl.Table.read(table_name, format=format_tbl)
	c1   = ftbl['AMS_AVG_IPT']
	c2   = ftbl['AMN_AVG_IPT']
	c3   = ftbl['SNR_AVG_IPT']
	c4   = ftbl['AMP_IP1_CRT']
	c5   = ftbl['AMP_IP2_CRT']
	c6   = ftbl['SGN_AVG_IPT']
	c7   = ftbl['SGV_AVG_IPT']
	c8   = ftbl['FWH_AVG_IPT']
	c9   = ftbl['OFS_AVG_IPT']
	c10  = ftbl['S2G_FLS_IPT']
	c11  = ftbl['S2G_IPT_CRT']
	c12  = ftbl['C2G_FLS_IPT']
	c13  = ftbl['C2G_IPT_CRT']
	c14  = ftbl['1DG_SAM_OPT']
	c15  = ftbl['1DG_SCT_OPT']
	c16  = ftbl['1DG_SSI_OPT']
	c17  = ftbl['1DG_SFW_OPT']
	c18  = ftbl['1DG_SAE_OPT']
	c19  = ftbl['1DG_SCE_OPT']
	c20  = ftbl['1DG_SSE_OPT']
	c21  = ftbl['1DG_AAM_OPT']
	c22  = ftbl['1DG_ACT_OPT']
	c23  = ftbl['1DG_ASI_OPT']
	c24  = ftbl['1DG_AFW_OPT']
	c25  = ftbl['1DG_AAE_OPT']
	c26  = ftbl['1DG_ACE_OPT']
	c27  = ftbl['1DG_ASE_OPT']
	c28  = ftbl['CH2_1SG_OPT']
	c29  = ftbl['CHR_1SG_OPT']
	c30  = ftbl['CBE_BIN_OPT']
	c31  = ftbl['S2G_XCT_OPT']
	c32  = ftbl['S2G_YCT_OPT']
	c33  = ftbl['S2G_AMP_OPT']
	c34  = ftbl['S2G_SGX_OPT']
	c35  = ftbl['S2G_SGY_OPT']
	c36  = ftbl['S2G_THT_OPT']
	c37  = ftbl['S2G_OFS_OPT']
	c38  = ftbl['S2G_XCE_OPT']
	c39  = ftbl['S2G_YCE_OPT']
	c40  = ftbl['S2G_AME_OPT']
	c41  = ftbl['S2G_SXE_OPT']
	c42  = ftbl['S2G_SYE_OPT']
	c43  = ftbl['S2G_THE_OPT']
	c44  = ftbl['S2G_OFE_OPT']
	c45  = ftbl['S2G_FLS_OPT']
	c46  = ftbl['S2G_FT1_OPT']
	c47  = ftbl['S2G_FT2_OPT']
	c48  = ftbl['S2G_FT3_OPT']
	c49  = ftbl['S2G_CNN_OPT']
	c50  = ftbl['CH2_2SG_OPT']
	c51  = ftbl['CHR_2SG_OPT']
	c52  = ftbl['CHR_2SA_OPT']
	c53  = ftbl['CHR_2SS_OPT']
	c54  = ftbl['C2G_XCT_OPT']
	c55  = ftbl['C2G_YCT_OPT']
	c56  = ftbl['C2G_AMP_OPT']
	c57  = ftbl['C2G_SGX_OPT']
	c58  = ftbl['C2G_SGY_OPT']
	c59  = ftbl['C2G_THT_OPT']
	c60  = ftbl['C2G_OFS_OPT']
	c61  = ftbl['C2G_XCE_OPT']
	c62  = ftbl['C2G_YCE_OPT']
	c63  = ftbl['C2G_AME_OPT']
	c64  = ftbl['C2G_SXE_OPT']
	c65  = ftbl['C2G_SYE_OPT']
	c66  = ftbl['C2G_THE_OPT']
	c67  = ftbl['C2G_OFE_OPT']
	c68  = ftbl['C2G_FLS_OPT']
	c69  = ftbl['C2G_FT1_OPT']
	c70  = ftbl['C2G_FT2_OPT']
	c71  = ftbl['C2G_FT3_OPT']
	c72  = ftbl['C2G_CNN_OPT']
	c73  = ftbl['CH2_CSG_OPT']
	c74  = ftbl['CHR_CSG_OPT']
	c75  = ftbl['CHR_CSA_OPT']
	c76  = ftbl['CHR_CSS_OPT']
	c77  = ftbl['OPT/IPT_R_S_1']
	c78  = ftbl['OPT/IPT_A']
	c79  = ftbl['OPT/IPT_A_CRT']
	c80  = ftbl['IPT/OPT_R_S_1']
	c81  = ftbl['IPT/OPT_R_A']
	c82  = ftbl['IPT/OPT_R_A_CRT']
	c83  = ftbl['IPT/OPT_R_S2G_0']
	c84  = ftbl['IPT/OPT_R_S2G_1']
	c85  = ftbl['IPT/OPT_R_S2G_2']
	c86  = ftbl['IPT/OPT_R_S2G_3']
	c87  = ftbl['IPT/OPT_R_S2G_0_CRT']
	c88  = ftbl['IPT/OPT_R_S2G_1_CRT']
	c89  = ftbl['IPT/OPT_R_S2G_2_CRT']
	c90  = ftbl['IPT/OPT_R_S2G_3_CRT']
	c91  = ftbl['OPT/IPT_A_CTR']
	c92  = ftbl['OPT/IPT_A_CTR_CRT']
	c93  = ftbl['IPT/OPT_R_A_CTR']
	c94  = ftbl['IPT/OPT_R_A_CTR_CRT']
	c95  = ftbl['IPT/OPT_R_C2G_0']
	c96  = ftbl['IPT/OPT_R_C2G_1']
	c97  = ftbl['IPT/OPT_R_C2G_2']
	c98  = ftbl['IPT/OPT_R_C2G_3']
	c99  = ftbl['IPT/OPT_R_C2G_0_CRT']
	c100 = ftbl['IPT/OPT_R_C2G_1_CRT']
	c101 = ftbl['IPT/OPT_R_C2G_2_CRT']
	c102 = ftbl['IPT/OPT_R_C2G_3_CRT']
	c103 = ftbl['SBS_FLS_OPT']
	c104 = ftbl['SBS_FT1_OPT']
	c105 = ftbl['SBS_FT2_OPT']
	c106 = ftbl['SBS_FT3_OPT']
	c107 = ftbl['IPT/OPT_R_SBS_0_SUM']
	c108 = ftbl['IPT/OPT_R_SBS_1_SUM']
	c109 = ftbl['IPT/OPT_R_SBS_2_SUM']
	c110 = ftbl['IPT/OPT_R_SBS_3_SUM']
	c111 = ftbl['IPT/OPT_R_SBS_0_SUM_CRT']
	c112 = ftbl['IPT/OPT_R_SBS_1_SUM_CRT']
	c113 = ftbl['IPT/OPT_R_SBS_2_SUM_CRT']
	c114 = ftbl['IPT/OPT_R_SBS_3_SUM_CRT']
	
	return(ftbl,
		c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,
		c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,
		c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,
		c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,
		c41,c42,c43,c44,c45,c46,c47,c48,c49,c50,
		c51,c52,c53,c54,c55,c56,c57,c58,c59,c60,
		c61,c62,c63,c64,c65,c66,c67,c68,c69,c70,
		c71,c72,c73,c74,c75,c76,c77,c78,c79,c80,
		c81,c82,c83,c84,c85,c86,c87,c88,c89,c90,
		c91,c92,c93,c94,c95,c96,c97,c98,c99,c100,
		c101,c102,c103,c104,c105,c106,c107,c108,c109,c110,
		c111,c112,c113)

def readtable_cat(table_name,format_tbl):
	cat = Table.read(table_name, format=format_tbl)
	c1  = cat['X']
	c2  = cat['Y']
	c3  = cat['A']
	c4  = cat['A_M']
	c5  = cat['SX']
	c6  = cat['SY']
	c7  = cat['T']
	c8  = cat['OS']
	c9  = cat['xN']
	return(cat,c1,c2,c3,c4,c5,c6,c7,c8,c9)

def Table_Read_Syn_Singl_It(table_name,format_tbl,*args, **kwargs):
	ftbl = aptbl.Table.read(table_name, format=format_tbl)
	c1   = ftbl['cube_fn']  
	c2   = ftbl['src_amp']  
	c3   = ftbl['nse_amp']  
	c4   = ftbl['snr_amp']  
	c5   = ftbl['sig_nmb']  
	c6   = ftbl['sig_vel']  
	c7   = ftbl['sig_fwh']  
	c8   = ftbl['src_ofs'] 
	c9   = ftbl['x_ctr_2D']
	c10  = ftbl['y_ctr_2D']
	c11  = ftbl['amp_2D']
	c12  = ftbl['amp_max_2D']
	c13  = ftbl['sgm_x_2D']
	c14  = ftbl['sgm_y_2D']
	c15  = ftbl['theta_2D']
	c16  = ftbl['offset_2D']
	c17  = ftbl['TN_2D']
	return(ftbl,
			c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,
			c11,c12,c13,c14,c15,c16,c17)

def Table_Ipt_Cat_Stats(Cat_Ipt_Tbl_Sts,hdr_sts):
	Splt_Vars = split_variable_vars(hdr_sts)
	Tbl_Splt_Col = Splt_Vars[0]
	Tbl_Splt_Hdr = Splt_Vars[1]
	Tbl_Splt_Hdr_Cmt = Splt_Vars[2]

	z_sample_avg     = np.mean(Cat_Ipt_Tbl_Sts[8])
	z_sample_med     = np.median(Cat_Ipt_Tbl_Sts[8])
	z_sample_1sl     = np.nanpercentile(Cat_Ipt_Tbl_Sts[8], 15.9)
	z_sample_1sh     = np.nanpercentile(Cat_Ipt_Tbl_Sts[8], 84.1)
	z_sample_2sl     = np.nanpercentile(Cat_Ipt_Tbl_Sts[8], 2.30)
	z_sample_2sh     = np.nanpercentile(Cat_Ipt_Tbl_Sts[8], 97.7)
	z_sample_3sl     = np.nanpercentile(Cat_Ipt_Tbl_Sts[8], 0.20)
	z_sample_3sh     = np.nanpercentile(Cat_Ipt_Tbl_Sts[8], 99.8)
	z_sample_p25     = np.nanpercentile(Cat_Ipt_Tbl_Sts[8], 25.0)
	z_sample_p75     = np.nanpercentile(Cat_Ipt_Tbl_Sts[8], 75.0)

	Splt_sample_avg  = np.mean(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col])
	Splt_sample_med  = np.median(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col])
	Splt_sample_1sl  = np.nanpercentile(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col], 15.9)
	Splt_sample_1sh  = np.nanpercentile(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col], 84.1)
	Splt_sample_2sl  = np.nanpercentile(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col], 2.30)
	Splt_sample_2sh  = np.nanpercentile(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col], 97.7)
	Splt_sample_3sl  = np.nanpercentile(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col], 0.20)
	Splt_sample_3sh  = np.nanpercentile(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col], 99.8)
	Splt_sample_p25  = np.nanpercentile(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col], 25.0)
	Splt_sample_p75  = np.nanpercentile(Cat_Ipt_Tbl_Sts[Tbl_Splt_Col], 75.0)
	print
	print ('Redshift (avg): ',z_sample_avg)
	print ('Redshift (med): ',z_sample_med)
	print ('Redshift (avg): ',Splt_sample_avg)
	print ('Redshift (med): ',Splt_sample_med)
	print ('subcube_width : ',subcube_width)
	var_sts = [
				['STZ_AVG','STZ_MED',
				'STZ_1SL','STZ_1SH',
				'STZ_2SL','STZ_2SH',
				'STZ_3SL','STZ_3SH',
				'STZ_P25','STZ_P75',
				'STS_AVG','STS_MED',
				'STS_1SL','STS_1SH',
				'STS_2SL','STS_2SH',
				'STS_3SL','STS_3SH',
				'STS_P25','STS_P75'],
				[z_sample_avg,z_sample_med,
				z_sample_1sl,z_sample_1sh,
				z_sample_2sl,z_sample_2sh,
				z_sample_3sl,z_sample_3sh,
				z_sample_p25,z_sample_p75,
				Splt_sample_avg,Splt_sample_med,
				Splt_sample_1sl,Splt_sample_1sh,
				Splt_sample_2sl,Splt_sample_2sh,
				Splt_sample_3sl,Splt_sample_3sh,
				Splt_sample_p25,Splt_sample_p75],
				Tbl_Splt_Hdr,
				Tbl_Splt_Hdr_Cmt					
				]
	return var_sts