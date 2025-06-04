import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


#function to store abs and fluo datas in a dictionary
def Open_txt(directory):
    #create an empty dictionary
    Rawdfs = {}
    
    try:
        for filename in os.listdir(directory):
            
            #if re.search('fluo_.*_\d{7}U1\.TXT_processed_spectra.txt', filename) or re.search('abs_.*_\d{7}U1\.TXT_processed_spectra.txt', filename) :
            if filename.startswith(('abs', 'fluo')):
                file_path = os.path.join(directory, filename)
                
                try:
                    # Read the file into a DataFrame
                    df = pd.read_csv(file_path, delimiter= ';', engine='python', skiprows=10, header=None)
                    #keep the columns with lambda and the corrected signal (abs or fluo)
                    df = df.replace({',': '.'}, regex=True)
                    df = df.apply(pd.to_numeric, errors='coerce')
                    # Add the DataFrame to the dictionary
                    print('Nb of frames ',filename,' = ', len(df.columns))
                    Rawdfs[filename] = df
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                
        # Return the dictionary of DataFrames
        return Rawdfs
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None

#funtion to remove Dark and reference spectrum and apply the baseline substraction
def Substract(Rawdfs):
    
    dfs={}
    
    try:
        for filename, df in Rawdfs.items():
            if filename.startswith('abs'):
                #Convert transmitance data into abs data
                DarkSpectrum = df.iloc[:,1]
                RefSpectrum = df.iloc[:,2]
                TransSample = df.iloc[:,3:]
                #Apply the formula -log10((Sample-Dark)/Ref)
                df.iloc[:, 3:] = TransSample.apply(lambda x: -np.log10((x - DarkSpectrum) / RefSpectrum))
                #Remove dark and reference columns
                df = df.drop(df.columns[[1, 2]], axis=1)
                dfs[filename] = df
            elif filename.startswith('fluo'):
                df = df.drop(df.columns[[1, 2]], axis=1)
                dfs[filename] = df
        return dfs
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#function to put headers on each df and keep only selected frames
def Column_adj(dfs, first_frame, last_frame):
    try:
        for filename, df in dfs.items():
            #Give a name to the columns, 'Wavelength' and one number for each frame
            df.columns = ['Wavelength'] + [str(i) for i in range(1, len(df.columns))]    
            #Filtering columns to keep 'Wavelength' and the choosen window
            columns_to_keep = ['Wavelength'] + [col for col in df if col != 'Wavelength' and first_frame <= int(col) <= last_frame]
            # Filter the DataFrame based on the columns to keep
            df = df[columns_to_keep]
            dfs[filename] = df
        
        return dfs
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#function to delete noise from laser(for fluo) and normalize datas
def Norm(dfs, Laser_405, Laser_561, directory, Exp_name, 
           Adj_abs_spectra, Adj_abs_high, Adj_abs_low, Adj_fluo_high,
           Adj_fluo_low, Adj_fluo_spectra, Norm_abs, Norm_fluo, SaveDir,
           BaselineCorr, BaselineWlgth, SmoothAbs, SmoothAbsRange, SmoothFluo, SmoothFluoRange):
    
    try:        
        for filename, df in dfs.items():
            columns_to_treat = df.columns[df.columns != 'Wavelength']
            
            #Keep desired abs Wavelength
            if filename.startswith('abs'):
                
                #Smooth Abs data
                if SmoothAbs[0]:
                    for col in columns_to_treat:
                        df[col] = np.convolve(df[col], np.ones(SmoothAbsRange[0]) / SmoothAbsRange[0], mode='same')
                
                #Apply baseline correction
                if BaselineCorr:
                    # Find the indices where x is within the baseline range
                    BaselineRows_pos = df[(df['Wavelength'] >= BaselineWlgth[0]) & (df['Wavelength'] <= BaselineWlgth[1])]
                    #Calculate the offset to apply
                    BaselineMean = BaselineRows_pos.mean(numeric_only=True)
                    
                    #Substract the y values in columns_to_treat by the calculated baseline
                    for col in columns_to_treat:
                        df[col] = df[col] - BaselineMean[col]
                
                #Normalize abs data
                if Norm_abs:                    
                    A280rows_pos = df[(df['Wavelength'] >= 275) & (df['Wavelength'] <= 285)]                    
                    #Mean the rows within the selected Wavelength window
                    A280mean = A280rows_pos.mean(numeric_only=True)
                    
                    # Apply normalization to the selected columns according to the A280nm (275/285) mean values
                    for col in columns_to_treat:
                        df[col] = df[col] / A280mean[col]

                if Adj_abs_spectra:                    
                    df = df[(((df['Wavelength'] >= Adj_abs_low) & (df['Wavelength'] <= Adj_abs_high)))]                    
                    
            if filename.startswith('fluo'):
                #Remove lasers noise (405 and 561) with a 7nm width      
                if Laser_405:
                    df = df[~(((df['Wavelength'] >= 402) & (df['Wavelength'] <= 409)))]
                if Laser_561:
                    df = df[~(((df['Wavelength'] >= 558) & (df['Wavelength'] <= 565)))]
                    
                #Smooth Abs data
                if SmoothFluo[0]:
                    for col in columns_to_treat:
                        df[col] = np.convolve(df[col], np.ones(SmoothFluoRange[0]) / SmoothFluoRange[0], mode='same')
                
                #Keep desired fluo Wavelength    
                if Adj_fluo_spectra:
                    df = df[(((df['Wavelength'] >= Adj_fluo_low) & (df['Wavelength'] <= Adj_fluo_high)))]
                
                #Normalize fluo data
                if Norm_fluo:
                   # Apply Min-Max normalization to the selected columns
                   df[columns_to_treat] = df[columns_to_treat].apply(lambda x: (x - x.min()) / (x.max() - x.min())) 
            
            #Apply the filter on the dictionary
            dfs[filename] = df
    
        return dfs    
    except Exception as e:
        print(f"An error occurred in Norm function: {e}")
        return None

#function to create time series from the selected wavelength ranges
def TimeSeries(dfs, FluoTS, AbsTS, Save, DataToProcess, SmoothAbs, SmoothAbsRange, SmoothFluo, SmoothFluoRange):
    
    try:
        if DataToProcess[1]:
            
            AbsTimeSeries = pd.DataFrame()
            FluoTimeSeries = pd.DataFrame()
            
            for key, df in dfs.items():
                
                if key.startswith('abs_12'):
                    
                    Amin_wavelength = AbsTS[0]
                    Amax_wavelength = AbsTS[1]
                    rows_pos = df[(df['Wavelength'] >= Amin_wavelength) & (df['Wavelength'] <= Amax_wavelength)]
                    #Mean the rows within the selected Wavelength window
                    select_rows = rows_pos.mean(numeric_only=True)
                    #Concat the sum results for each columns in AbsTimeSeries df
                    AbsTimeSeries = pd.concat([AbsTimeSeries, select_rows.to_frame().T], ignore_index=True)
                    AbsTimeSeries = AbsTimeSeries.drop(columns=['Wavelength'], errors='ignore')
                    
                    if SmoothAbs[1]:
                        # Apply smoothing to the single row (convert to numpy array for smoothing)
                        AbsTimeSeries.iloc[0] = np.convolve(AbsTimeSeries.iloc[0].values, np.ones(SmoothAbsRange[1]) / SmoothAbsRange[1], mode='same')

                if key.startswith('fluo_10'):
                    
                    Fmin_wavelength = FluoTS[0]
                    Fmax_wavelength = FluoTS[1]
                    rows_pos = df[(df['Wavelength'] >= Fmin_wavelength) & (df['Wavelength'] <= Fmax_wavelength)]
                    select_rows = rows_pos.mean(numeric_only=True)
                    FluoTimeSeries = pd.concat([FluoTimeSeries, select_rows.to_frame().T], ignore_index=True)
                    FluoTimeSeries = FluoTimeSeries.drop(columns=['Wavelength'], errors='ignore')
                    
                    if SmoothFluo[1]:
                        # Apply smoothing to the single row (convert to numpy array for smoothing)
                        FluoTimeSeries.iloc[0] = np.convolve(FluoTimeSeries.iloc[0].values, np.ones(SmoothFluoRange[1]) / SmoothFluoRange[1], mode='same')
                    
            #Update the modified df in dfs    
            if not AbsTimeSeries.empty:
                dfs['AbsTimeSeries'] = AbsTimeSeries
            if not FluoTimeSeries.empty:
                dfs['FluoTimeSeries'] = FluoTimeSeries
            
        return dfs    
    except Exception as e:
        print(f"An error occurred in TimeSeries function : {e}")
        return None  

def Plot_dfs(dfs, first_frame, last_frame, directory, Exp_name, Save_png, Norm_abs, Norm_fluo, SaveDir, Save, AbsTS, FluoTS, DataToProcess):
    try:    
        # Choose a colormap
        cmap = plt.get_cmap('viridis')      
        # Normalize the data for the colormap based on the colnames (frame numbers)
        norm = Normalize(vmin=first_frame, vmax=last_frame)
        
        # Initialize axes variables
        abs_ax = None
        fluo_ax = None
        TSabs_ax = None
        TSfluo_ax = None
        
        if all(DataToProcess):
            # Create a 2x2 subplot
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            plt.subplots_adjust(hspace=0.3)    
            # Assign axes for absorbance (left) and fluorescence (right) explicitly
            abs_ax = axs[0, 0]  
            fluo_ax = axs[0, 1]  
            TSabs_ax = axs[1, 0]  
            TSfluo_ax = axs[1, 1]
        
        elif DataToProcess[0]:
            # Create a 1x2 subplot
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            # Assign axes for absorbance (left) and fluorescence (right) explicitly
            abs_ax = axs[0]
            fluo_ax = axs[1]
        
        elif DataToProcess[1]:
            # Create a 1x2 subplot
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            # Assign axes for absorbance (left) and fluorescence (right) explicitly
            TSabs_ax = axs[0]
            TSfluo_ax = axs[1]
            
        # Create a scalar mappable object for color mapping across all subplots
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array for colorbar
        
        # Iterate through the DataFrames in the dfs dictionary
        for filename, df in dfs.items():
            
            if filename.startswith('fluo_10') and fluo_ax is not None:  # Fluorescence data
                for col in df.columns:
                    if col != 'Wavelength':
                        frame = int(col)  # Extract the frame number
                        color = cmap(norm(frame))  # Normalize frame for colormap
                        fluo_ax.plot(df['Wavelength'], df[col], label=f'Frame {frame}', color=color)
                
                fluo_ax.set_xlabel('Wavelength')
                fluo_ax.set_title('Fluorescence')
                if Norm_fluo:
                    fluo_ax.set_ylim(0, 1.1)
                    fluo_ax.set_ylabel('Normalized Emitted Fluorescence Intensity')
                else:
                    fluo_ax.set_ylabel('Fluorescence Intensity')

            elif filename.startswith('abs_12') and abs_ax is not None:  # Absorbance data
                for col in df.columns:
                    if col != 'Wavelength':  # Skip 'Wavelength' column
                        frame = int(col)
                        color = cmap(norm(frame))
                        abs_ax.plot(df['Wavelength'], df[col], label=f'Frame {frame}', color=color)
                
                if Norm_abs:
                    abs_ax.set_ylabel('Normalized Absorbance')
                else:    
                    abs_ax.set_ylabel('Absorbance')
                abs_ax.set_title('Absorbance')
                abs_ax.set_xlabel('Wavelength')
            
            elif filename.startswith('AbsTime') and TSabs_ax is not None:  # Absorbance data                
                TSabs_ax.plot(df.columns, df.iloc[0])   
                if Norm_abs:
                    TSabs_ax.set_ylabel('Mean Normalized Absorbance')
                else:    
                    TSabs_ax.set_ylabel('Absorbance')
                TSabs_ax.set_title(f'Absorbance Time Series from {AbsTS[0]}nm to {AbsTS[1]}nm')
                TSabs_ax.set_xlabel('Frames')
                TSabs_ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
            
            elif filename.startswith('FluoTime') and TSfluo_ax is not None:  # Fluo data
                TSfluo_ax.plot(df.columns, df.iloc[0])   
                TSfluo_ax.set_ylabel('Mean Intensity')
                TSfluo_ax.set_title(f'Fluorescence Time Series from {FluoTS[0]}nm to {FluoTS[1]}nm')
                TSfluo_ax.set_xlabel('Frames')
                TSfluo_ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

        # Add the colorbar (legend for the colormap)
        if all(DataToProcess):
            cbar = plt.colorbar(sm, ax=axs[0, ], fraction=0.01, pad=0.1)
            cbar.set_ticks([first_frame, last_frame])
        elif DataToProcess[0]:
            cbar = plt.colorbar(sm, ax=axs, fraction=0.01, pad=0.1)
            cbar.set_ticks([first_frame, last_frame])
        plt.show()

        # Save the data
        if Save[0]:    
            for filename, df in dfs.items():
                df.to_csv(SaveDir + Exp_name + filename + '.csv')                

        if Save[1]:    
            fig.savefig(SaveDir + Exp_name + '.png', bbox_inches='tight', dpi=300)
        
        return plt
    
    except Exception as e:
        print(f"An error occurred in Plot_dfs function : {e}")
        return None
