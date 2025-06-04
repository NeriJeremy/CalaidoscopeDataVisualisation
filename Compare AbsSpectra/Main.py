from module_caleidoscopeNew import Open_txt, Substract, Column_adj, Norm,Plot_dfs

"""
Requiries:
    Calaidoscope data
    For each conditions, the filename should be abs_1204051U1_***

V3:
    - A correction has been made in the Column_adj function
"""

##########################################################################################################################################
##################################################### INPUT PARAMETERS #########################################################################

#Requirements :Having some preprocessed datas from Dominique's Matlab code "Multiple_spectra_reading_Dom" 

#Choose the directory containing the .txt files
directory = 'S:/Jeremy/PinkChromoRS/Results/caleidoscope/20250228_pH_switch/Results/162A/raw_data'

#Choose the name of the column to keep
ColToKeep = '1'

#Smooth data with mean value
Smooth = True
#Choose the smooth range
SmoothRange=10

#Perform baseline correction
BaselineCorr=True
#Range for baseline correction
BaselineWlgth=[700,750]

#Adjust abs spectra
Adj_abs_spectra=True
#Choose lower and upper limit of the adjust abs spectra
Adj_abs_low=350
Adj_abs_high=650
#Normalize abs spectra (with Abs280nm)
Norm_abs=True

#Choose the saving parameters
Exp_name = '162A_AbsSpectra_pH'
SaveDir = 'S:/Jeremy/Presentations/20250423-Dorus_Update'
#Choose data to save [CSV, Plot]
Save = [False, True]

#################################################################################################################################################################

#call the functions
Rawdfs = Open_txt(directory)

dfs = Substract(Rawdfs)

dfs = Column_adj(dfs, ColToKeep)

dfs = Norm(dfs, directory, Adj_abs_spectra, Adj_abs_high, Adj_abs_low, Norm_abs,
           SaveDir, Smooth, SmoothRange, BaselineCorr, BaselineWlgth)

Plt = Plot_dfs(dfs, Norm_abs, Save, SaveDir, Exp_name, ColToKeep)























