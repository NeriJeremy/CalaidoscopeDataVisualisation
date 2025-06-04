from module_caleidoscope import Open_txt, Substract, Column_adj, Norm, TimeSeries, Plot_dfs

"""

Requirements :
    Calaidoscope data
    For each conditions, the filename should be either fluo_1004063U1_*** or abs_1204051U1_***

V7 :
    - bug correction, you can now read only abs or fluo data 

"""

##########################################################################################################################################
##################################################### INPUT PARAMETERS ######################################################################### 

#Choose the directory containing the .txt files
directory = 'S:/Jeremy/PinkChromoRS/Results/caleidoscope/20250218/log_18févr.25_133836'

#Choose the used laser for fluo (the lasers wavelength will be removed)
Laser_405=True
Laser_561=True

#Choose the beginning and the end of the acquisition
first_frame = 1
last_frame = 854

#Choose data to process and plot [Spectra, Time Series]
DataToProcess = [True, True]


#Smooth Abs data [Spectra, TimeSerie]
SmoothAbs=[False, False]
#Choose the smooth range [Spectra, TimeSerie]
SmoothAbsRange=[5,10]

#Perform baseline correction
BaselineCorr=True
#Range for baseline correction
BaselineWlgth=[700,750]

#Normalize abs spectra (with Abs280nm)
Norm_abs=True

#Choose Abs wavelength integration for time serie
AbsTS = [550, 600]

#Adjust abs spectra
Adj_abs_spectra=True
#Choose lower and upper limit of the adjust abs spectra
Adj_abs_low=250
Adj_abs_high=700


#Smooth Fluo data [Spectra, TimeSerie]
SmoothFluo=[False, False]
#Choose the smooth range [Spectra, TimeSerie]
SmoothFluoRange=[5,5]

#Normalize fluo spectra (With Min Max values)
Norm_fluo=False

#Choose Fluo wavelength integration for time serie
FluoTS = [600, 650]

#Adjust fluo spectra
Adj_fluo_spectra=True
#Choose lower and upper limit of the adjust fluo spectra
Adj_fluo_low=400
Adj_fluo_high=800

#Choose the saving parameters
Exp_name = 'Pink_Offswitch'
SaveDir = 'S:/Jeremy/Presentations/20250423-Dorus_Update/'
#Choose data to save [CSV, Plot]
Save = [False, True]

#################################################################################################################################################################

#call the functions
Rawdfs = Open_txt(directory)

dfs = Substract(Rawdfs)

dfs = Column_adj(dfs, first_frame, last_frame)

dfs = Norm(dfs, Laser_405, Laser_561, directory, Exp_name, 
           Adj_abs_spectra, Adj_abs_high, Adj_abs_low, Adj_fluo_high,
           Adj_fluo_low, Adj_fluo_spectra, Norm_abs, Norm_fluo, SaveDir,
           BaselineCorr, BaselineWlgth, SmoothAbs, SmoothAbsRange, SmoothFluo, SmoothFluoRange)

dfs = TimeSeries(dfs, FluoTS, AbsTS, Save, DataToProcess, SmoothAbs, SmoothAbsRange, SmoothFluo, SmoothFluoRange)

dfs = Plot_dfs(dfs, first_frame, last_frame, directory, Exp_name, Save, Norm_abs, Norm_fluo, SaveDir, Save, AbsTS, FluoTS, DataToProcess)























