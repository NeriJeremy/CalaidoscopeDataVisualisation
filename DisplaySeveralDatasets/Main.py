from module_caleidoscopeNew import Open_txt, Substract, Column_adj, Norm, TimeSeries, Plot_dfs

"""
Requiries:
    Calaidoscope data
    For each conditions, the filename should be either fluo_1004063U1_***.TXT or abs_1204051U1_***.TXT

V6:
    - Normalisation of Fluo data are now made on the Time series
    - You can now choose to plot Fluo and/or Abs data
    - A block of code is available in the module to sort the data by indexes (in TimeSeries function)
    - A block of code is available in the module to change the colors by colorblind friendly colors
"""

##########################################################################################################################################
##################################################### INPUT PARAMETERS #########################################################################
 

#Choose the directory containing the .txt files
directory = 'S:/Jeremy/PinkChromoRS/Results/caleidoscope/20250523-Glox/20250523/Pool'

#Choose the used laser for fluo (the lasers wavelength will be removed)
Laser_405=False
Laser_561=True

#Choose the beginning and the end of the acquisition
first_frame = 1
last_frame = 500

#Choose data to plot [Abs,Fluo]
DataToPlot = [False,True]


#Smooth Abs data
SmoothAbs=True
#Choose the smooth range (Warning, the smoothing range has to be non even, otherwise it'll lead to dataframe misalignements)
SmoothAbsRange=5

#Perform baseline correction
BaselineCorr=True
#Range for baseline correction
BaselineWlgth=[700,750]

#Normalize abs spectra (with Abs280nm)
Norm_abs=True

#Choose Fluo wavelength integration for time serie
AbsTS = [550, 600]


#Smooth fluo data
SmoothFluo=True
#Choose the smooth range (Warning, the smoothing range has to be non even, otherwise it'll lead to dataframe misalignements)
SmoothFluoRange=5

#Normalize fluo spectra (With Min Max values)
Norm_fluo=True

#Choose Abs wavelength integration for time serie
FluoTS = [600, 650]

#Choose the saving parameters
Exp_name = 'Pink_Glox'
SaveDir = 'S:/Jeremy/PinkChromoRS/Results/caleidoscope/20250523-Glox/20250523/Pool'
#Choose data to save [CSV, Plot]
Save = [False, True]

#################################################################################################################################################################

#call the functions
Rawdfs = Open_txt(directory)

dfs = Substract(Rawdfs)

dfs = Column_adj(dfs, first_frame, last_frame)

dfs = Norm(dfs, Laser_405, Laser_561, directory, Exp_name, Norm_abs, SaveDir,
         BaselineCorr, BaselineWlgth)

AbsTimeSeries, FluoTimeSeries = TimeSeries(dfs, FluoTS, AbsTS, Save, SmoothAbs, SmoothAbsRange, SmoothFluo, SmoothFluoRange, Norm_fluo)

Plt = Plot_dfs(AbsTimeSeries, FluoTimeSeries, Norm_fluo, Norm_abs, Save, SaveDir, Exp_name, FluoTS, AbsTS, DataToPlot)























