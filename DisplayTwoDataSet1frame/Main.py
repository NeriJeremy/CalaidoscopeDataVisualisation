from module_caleidoscopeNew import Open_txt, Column_adj, Norm, Plot_dfs

##########################################################################################################################################
##################################################### INPUT PARAMETERS #########################################################################

#Requirements :Having some preprocessed datas from Dominique's Matlab code "Multiple_spectra_reading_Dom" 

#Choose the directory containing the .txt files
dir1 = 'S:/Jeremy/PinkChromoRS/Results/caleidoscope/20241115/Pink_exp1/log_15nov.24_092200/'
dir2 = 'S:/Jeremy/PinkChromoRS/Results/caleidoscope/20241115/162_exp1/'


#Choose the used laser for fluo
Laser_405=True
Laser_561=True

#Choose the beginning and the frame you want to display
first_frame1 = 25
last_frame1 = 25
first_frame2 = 25
last_frame2 = 25

#Adjust abs spectra
Adj_abs_spectra=True
#Choose lower and upper limit of the adjust abs spectra
Adj_abs_low=200
Adj_abs_high=800
#Normalize abs spectra
Norm_abs=True

#Adjust fluo spectra
Adj_fluo_spectra=True
#Choose lower and upper limit of the adjust fluo spectra
Adj_fluo_low=570
Adj_fluo_high=620
#Normalize fluo spectra
Norm_fluo=True

ProtName1 = 'Pink'
ProtName2 = '162A'

#Choose the name of your experiment to save it
Exp_name = 'Pink_spectra'

#Save normalized data as csv file and/or as png
Save_dir= 'S:/Jeremy/Presentations/20250212/'
Save_csv=False
Save_png=False

#################################################################################################################################################################

#call the functions
dfs_dir1, dfs_dir2 = Open_txt(dir1, dir2)

dfs_dir1, dfs_dir2 = Column_adj(dfs_dir1, dfs_dir2, first_frame1, last_frame1, first_frame2, last_frame2)

dfs_dir1, dfs_dir2 = Norm(dfs_dir1, dfs_dir2, Laser_405, Laser_561, 
                          Save_dir, Exp_name, 
                          Save_csv, Adj_abs_spectra, Adj_abs_high,
                          Adj_abs_low, Adj_fluo_high, Adj_fluo_low,
                          Adj_fluo_spectra, Norm_abs, Norm_fluo)

plt = Plot_dfs(dfs_dir1, dfs_dir2, Save_dir, Exp_name, Save_png, Norm_abs, Norm_fluo, ProtName1, ProtName2)























