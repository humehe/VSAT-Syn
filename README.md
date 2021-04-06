# VSAT-Syn
Valparaíso Stacking Analysis Tool Synthetic Datacubes


VSAT-Syn is part of the Valparaíso Stacking Analysis Tool (VSAT), it provide a series of tools for generating synthetic datacubes simulatiing datacubes coming from interferometric datasets. Although VSAT-Syn was designed to estimate the systematic flux measurement errors by mimiicking images geenerated from interferometric _uv_ datasets, VSAT-Syn can also be used to simulate other types of datacubes. VSAT-Syn can also be used to simulate 2D images, by simple assuming single-channels datacubes. The generated datacubes can be then stacked and used to measure the flux the composite images to estimate the systematic flux measurement errors.

![Alt text](./Figures-Syn/Synthetic-InOut-Stats-SNR-BIS.jpg?raw=true "3D datacube Stacked spectra Scheme.")

## Content

1. Fnc_Syn_Dir.py:
   - Location of the input catalogue and spectral data. 
   - Parameters for generating synthetic datacubes
   - Location of the resulting products of the stacking analyses _e.g. stamps, tables, plots,and stacked spectra_.

2. Fnc_Syn_Mth.py:
   - Math functions (e.g. cosmological constants, gaussian profiles for line fitting) needed throughout the stacking analysis.

3. Fnc_Syn_Spc.py 
   - Tools for modyfing datacubes including _e.g. masking, adding missing frequencies, eextract regions etc_

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

## Parameters
VSAT-Syn generate synthetic datacubes mimicking the observatioinal conditions of spectroscopic datacubes. Although VSAT-Syn was design to mimic the generated imaages from interferometric datasets, VSAT-Syn can be used to simulate other types of datacubes. VSAT-Syn can also obe used to simulate 2D images assuming single-channels datacubes. The generated datacubes can be then stacked and used to measure the flux froom the composite images to estimate the systematic flux measurement errors.
