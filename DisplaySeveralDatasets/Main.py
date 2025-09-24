from module_caleidoscopeNew import Open_txt, Substract, Column_adj, Adjust, TimeSeries, Plot_dfs

"""
Requiries:
    Calaidoscope data
    For each conditions, the filename should be either fluo_1004063U1_***.TXT or abs_1204051U1_***.TXT

V7:
    - You can normalize Abs data based on 280nm absorbance and then apply min/max normalization

"""

##########################################################################################################################################
##################################################### INPUT PARAMETERS #########################################################################
 

#Choose the directory containing the .txt files
directory = 'S:/Ludwig/Kaleidoscope/Test Folder for Jeremy code'

#Choose the used laser for fluo (the lasers wavelength will be removed)
Laser_405=True
Laser_561=True

#Choose the beginning and the end of the acquisition
first_frame = 1
last_frame = 100

#Choose data to plot [Abs,Fluo]
DataToPlot = [True,True]

#Normalize abs data (based on 280nm absorbance)
Norm_abs=False

#Smooth Abs data
SmoothAbs=True
#Choose the smooth range (Warning, the smoothing range has to be non even, otherwise it'll lead to dataframe misalignements)
SmoothAbsRange=5

#Perform baseline correction
BaselineCorr=True
#Range for baseline correction
BaselineWlgth=[700,750]

#Min/Max Normalization of abs spectra 
Norm_abs_MinMax=True

#Choose Fluo wavelength integration for time serie
AbsTS = [480, 490]

#Smooth fluo data
SmoothFluo=True
#Choose the smooth range (Warning, the smoothing range has to be non even, otherwise it'll lead to dataframe misalignements)
SmoothFluoRange=5

#Normalize fluo spectra (With Min Max values)
Norm_fluo=True

#Choose Abs wavelength integration for time serie
FluoTS = [505, 515]

#Choose the saving parameters
Exp_name = 'WithFluoNorm'
SaveDir = 'S:/Ludwig/Kaleidoscope/Test Folder for Jeremy code'
#Choose data to save [CSV, Plot]
Save = [False, True]

#################################################################################################################################################################

#call the functions
Rawdfs = Open_txt(directory)

dfs = Substract(Rawdfs)

dfs = Column_adj(dfs, first_frame, last_frame)

dfs = Adjust(dfs, Laser_405, Laser_561, directory, Exp_name, SaveDir, BaselineCorr, BaselineWlgth, Norm_abs)

AbsTimeSeries, FluoTimeSeries = TimeSeries(dfs, FluoTS, AbsTS, Save, SmoothAbs, SmoothAbsRange, SmoothFluo, SmoothFluoRange, Norm_fluo, Norm_abs_MinMax)

Plt = Plot_dfs(AbsTimeSeries, FluoTimeSeries, Norm_fluo, Norm_abs, Save, SaveDir, Exp_name, FluoTS, AbsTS, DataToPlot)























