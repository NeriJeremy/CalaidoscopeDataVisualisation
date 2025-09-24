import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import signal


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
def Adjust(dfs, Laser_405, Laser_561, directory, Exp_name, SaveDir, BaselineCorr, BaselineWlgth, Norm_abs):
    
    try:        
        for filename, df in dfs.items():
            
            columns_to_treat = df.columns[df.columns != 'Wavelength']
            
            #Keep desired abs Wavelength
            if filename.startswith('abs'):
                
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
                 
            if filename.startswith('fluo'):
                
                #Remove lasers noise (405 and 561) with a 7nm width      
                if Laser_405:
                    df = df[~(((df['Wavelength'] >= 400) & (df['Wavelength'] <= 409)))]
                if Laser_561:
                    df = df[~(((df['Wavelength'] >= 558) & (df['Wavelength'] <= 565)))] 
                
            #Apply the filter on the dictionary
            dfs[filename] = df
            
        return dfs    
    except Exception as e:
        print(f"An error occurred in Norm function: {e}")
        return None

#function to create time series from the selected wavelength ranges
def TimeSeries(dfs, FluoTS, AbsTS, Save, SmoothAbs, SmoothAbsRange, SmoothFluo, SmoothFluoRange, Norm_fluo, Norm_abs):
    
    try:
        AbsTimeSeries = pd.DataFrame()
        FluoTimeSeries = pd.DataFrame()
            
        for key, df in dfs.items():
                
            if key.startswith('abs_12'):
                Amin_wavelength = AbsTS[0]
                Amax_wavelength = AbsTS[1]                
                rows_pos = df[(df['Wavelength'] >= Amin_wavelength) & (df['Wavelength'] <= Amax_wavelength)]
                #Mean the rows within the selected Wavelength window
                select_rows = rows_pos.mean(numeric_only=True)
                # Convert the series to a DataFrame
                select_rows_df = select_rows.to_frame().T
                #Isolate the name of the condition and append it to the df
                Condition_name = re.search(r'(?<=U1_).*(?=\.TXT)', key)
                #store in 'filename' the condition name, .group(0) is to extract the string 
                select_rows_df['filename'] = Condition_name.group(0)
                select_rows_df.set_index('filename', inplace=True)                
                select_rows_df = select_rows_df.drop(columns=['Wavelength'], errors='ignore')
                
                if SmoothAbs:
                    # Get the first row as a NumPy array
                    data = select_rows_df.iloc[0, :].values
                    
                    # Create the moving average kernel (a normalized filter)
                    kernel = np.ones(SmoothAbsRange) / SmoothAbsRange

                    padded_data = np.pad(data, (SmoothAbsRange // 2, SmoothAbsRange // 2), mode='edge')
                        
                    # Apply convolution to the padded data
                    smoothed_values = signal.convolve(padded_data, kernel, mode = 'valid', method = 'auto')
                    
                    # Update the DataFrame with the smoothed values
                    select_rows_df.iloc[0, :] = smoothed_values.astype(np.float64)
                    
                #Normalize abs data
                if Norm_abs:
                    # Apply Min-Max normalization to the selected columns
                    data = select_rows_df.iloc[0, :].values
                    normalized_data = (data - data.min()) / (data.max() - data.min())
                    select_rows_df.iloc[0, :] = normalized_data.astype(np.float64)
                    
                #Concat the sum results for each columns in AbsTimeSeries df
                AbsTimeSeries = pd.concat([AbsTimeSeries, select_rows_df], ignore_index=False)
                
            elif key.startswith('fluo_10'):
                Fmin_wavelength = FluoTS[0]
                Fmax_wavelength = FluoTS[1]
                rows_pos = df[(df['Wavelength'] >= Fmin_wavelength) & (df['Wavelength'] <= Fmax_wavelength)]
                select_rows = rows_pos.mean(numeric_only=True)
                select_rows_df = select_rows.to_frame().T
                Condition_name = re.search(r'(?<=U1_).*(?=\.TXT)', key)
                select_rows_df['filename'] = Condition_name.group(0)
                select_rows_df.set_index('filename', inplace=True)
                select_rows_df = select_rows_df.drop(columns=['Wavelength'], errors='ignore')
                
                if SmoothFluo:
                    #Same process as abs data
                    data = select_rows_df.iloc[0, :].values
                    kernel = np.ones(SmoothFluoRange) / SmoothFluoRange
                    
                    # Manually pad data to compensate the loss caused by the convolution
                    padded_data = np.pad(data, (SmoothFluoRange // 2, SmoothFluoRange // 2), mode='edge')
                    
                    # Apply convolution to the padded data
                    smoothed_values = signal.convolve(padded_data, kernel, mode = 'valid', method = 'auto')
                    
                    # Update the DataFrame with the smoothed values
                    select_rows_df.iloc[0, :] = smoothed_values.astype(np.float64)
                
                #Normalize fluo data
                if Norm_fluo:
                   # Apply Min-Max normalization to the selected columns
                   data = select_rows_df.iloc[0, :].values
                   normalized_data = (data - data.min()) / (data.max() - data.min())
                   select_rows_df.iloc[0, :] = normalized_data.astype(np.float64)
                
                FluoTimeSeries = pd.concat([FluoTimeSeries, select_rows_df], ignore_index=False)
                
                """
                #Sort the indexes 
                FluoTimeSeries['numeric_index'] = [int(re.findall(r'\d+', str(idx))[0]) for idx in FluoTimeSeries.index]
                FluoTimeSeries = FluoTimeSeries.sort_values('numeric_index')
                FluoTimeSeries.index = [f'{num} min' for num in FluoTimeSeries['numeric_index']]
                FluoTimeSeries = FluoTimeSeries.drop(columns=['numeric_index'])
                """
                
            
        return AbsTimeSeries, FluoTimeSeries    
    except Exception as e:
        print(f"An error occurred in TimeSeries function : {e}")
        return None

def Plot_dfs(AbsTimeSeries, FluoTimeSeries, Norm_fluo, Norm_abs, Save, SaveDir, Exp_name, FluoTS, AbsTS, DataToPlot):
    
    try:
        
        # Initialize axes variables
        abs_ax = None
        fluo_ax = None
        fig = None
        """
        # Define the color mapping for each condition
        condition_colors = {
            'pH4.7': '#E69F00',
            'pH5.2': '#56B4E9',
            'pH6.1': '#009E73',
            'pH7.4': '#F0E442',
            'pH8.0': '#0072B2',
            'pH9.3': '#D55E00'
        }"""
        
        if all(DataToPlot):
            # Create a 2x2 subplot
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            plt.subplots_adjust(hspace=0.3)    
            # Assign axes for absorbance (left) and fluorescence (right) explicitly
            abs_ax = axs[0]  
            fluo_ax = axs[1]
            
        elif DataToPlot[0]:
            # Create a 1x2 subplot
            fig, abs_ax = plt.subplots(figsize=(8, 6))
        
        elif DataToPlot[1]:
            # Create a 1x2 subplot
            fig, fluo_ax = plt.subplots(figsize=(8, 6))
        
        if DataToPlot[0] and abs_ax is not None:    
            for filename in AbsTimeSeries.index:
                #color = condition_colors.get(filename, '#000000')
                abs_ax.plot(AbsTimeSeries.columns, AbsTimeSeries.loc[filename], label=filename) #, color = color)
                abs_ax.set_xlabel('Frames')
                abs_ax.set_title(f'Absorbance Time Series from {AbsTS[0]}nm to {AbsTS[1]}nm')
                abs_ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
                if Norm_abs:
                    abs_ax.set_ylabel('Normalized Absorbance')
                else:    
                    abs_ax.set_ylabel('Absorbance')
        
        if DataToPlot[1] and fluo_ax is not None:
            for filename in FluoTimeSeries.index:
                #color = condition_colors.get(filename, '#000000')
                fluo_ax.plot(FluoTimeSeries.columns, FluoTimeSeries.loc[filename], label=filename) #, color = color)
                fluo_ax.set_xlabel('Frames')
                fluo_ax.set_title(f'Fluorescence Time Series from {FluoTS[0]}nm to {FluoTS[1]}nm')
                fluo_ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
                if Norm_fluo:
                    fluo_ax.set_ylim(0, 1.1)
                    fluo_ax.set_ylabel('Normalized Fluorescence Intensity')
                else:
                    fluo_ax.set_ylabel('Fluorescence Intensity')

        if all(DataToPlot):
            axs[1].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        else:
            plt.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
            
        plt.tight_layout()
        # Save the data
        if Save[0]:    
            AbsTimeSeries.to_csv(SaveDir + Exp_name + '_Abs.csv')
            FluoTimeSeries.to_csv(SaveDir + Exp_name + '_Fluo.csv')                

        if Save[1]:    
            fig.savefig(SaveDir + Exp_name + '.png', bbox_inches='tight', dpi=300)
        
        return plt
    
    except Exception as e:
        print(f"An error occurred in Plot_dfs function : {e}")
        return None    