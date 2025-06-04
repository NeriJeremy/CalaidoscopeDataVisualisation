import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt


#function to store abs and fluo datas in a dictionary
def Open_txt(directory):
    #create an empty dictionary
    Rawdfs = {}
    
    try:
        for filename in os.listdir(directory):
            
            if filename.startswith('abs'):
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
            #Convert transmitance data into abs data
            DarkSpectrum = df.iloc[:,1]
            RefSpectrum = df.iloc[:,2]
            TransSample = df.iloc[:,3:]
            #Apply the formula -log10((Sample-Dark)/Ref)
            df.iloc[:, 3:] = TransSample.apply(lambda x: -np.log10((x - DarkSpectrum) / RefSpectrum))
            #Remove dark and reference columns
            df = df.drop(df.columns[[1, 2]], axis=1)
            dfs[filename] = df
        return dfs
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
#function to put headers on each df and keep only selected frames
def Column_adj(dfs, ColToKeep):
    try:
        for filename, df in dfs.items():
            #Give a name to the columns, 'Wavelength' and one number for each frame
            df.columns = ['Wavelength'] + [str(i) for i in range(1, len(df.columns))]
            if ColToKeep not in df.columns:
                raise ValueError(f"Column '{ColToKeep}' does not exist in the DataFrame.")    
            #Filtering columns to keep 'Wavelength' and the choosen column
            columns_to_keep = ['Wavelength', str(ColToKeep)]
            # Filter the DataFrame based on the columns to keep
            df = df[columns_to_keep]
            dfs[filename] = df
        
        return dfs
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#function to normalize datas
def Norm(dfs, directory, Adj_abs_spectra, Adj_abs_high, Adj_abs_low, Norm_abs,
           SaveDir, Smooth, SmoothRange, BaselineCorr, BaselineWlgth):
    
    try:        
        for filename, df in dfs.items():
            
            if Norm_abs:
                
                A280rows_pos = df[(df['Wavelength'] >= 275) & (df['Wavelength'] <= 285)]                    
                #Mean the rows within the selected Wavelength window
                A280mean = A280rows_pos.iloc[:, 1].mean() #numeric_only=True                   
                # Apply normalization to the selected columns according to the A280nm (275/285) mean values
                df.iloc[:, 1] = df.iloc[:, 1] / A280mean
                
            #Apply baseline correction
            if BaselineCorr:
                # Find the indices where x is within the baseline range
                BaselineRows_pos = df[(df['Wavelength'] >= BaselineWlgth[0]) & (df['Wavelength'] <= BaselineWlgth[1])]
                #Calculate the offset to apply
                BaselineMean = BaselineRows_pos.iloc[:, 1].mean()  # Take the mean of the second column (index 1)    
                # Subtract the calculated baseline mean from the entire column
                df.iloc[:, 1] = df.iloc[:, 1] - BaselineMean
            
            #Smooth data 
            if Smooth:
                # Apply convolution on the first row of the DataFrame (convert to numpy array for smoothing)
                smoothed_values = np.convolve(df.iloc[:, 1].values, np.ones(SmoothRange) / SmoothRange, mode='same')
                # Ensure the result is a float64 array to match the DataFrame dtype
                df.iloc[:, 1] = smoothed_values.astype(np.float64)
            
            if Adj_abs_spectra:                    
                df = df[(((df['Wavelength'] >= Adj_abs_low) & (df['Wavelength'] <= Adj_abs_high)))]
            
            #Apply the filter on the dictionary
            dfs[filename] = df
    
        return dfs    
    except Exception as e:
        print(f"An error occurred in Norm Function : {e}")
        return None


def Plot_dfs(dfs, Norm_abs, Save, SaveDir, Exp_name, ColToKeep):
    try:
        # Create a 1x2 subplot (one for fluorescence, one for absorbance)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define the color mapping for each condition
        condition_colors = {
            'pH4.7': '#E69F00',
            'pH5.2': '#56B4E9',
            'pH6.1': '#009E73',
            'pH7.4': '#F0E442',
            'pH8.0': '#0072B2',
            'pH9.3': '#D55E00'
        }

        for filename, df in dfs.items():
            LegRegex = re.search(r'(?<=U1_)(.*?)(?=\.TXT)', filename).group(0)
            color = condition_colors.get(LegRegex, '#000000')
            ax.plot(df['Wavelength'], df[ColToKeep], label=LegRegex, color=color)
            ax.set_xlabel('Wavelength')
            ax.set_title('Absorbance spectra')
            if Norm_abs:
                ax.set_ylabel('Normalized Absorbance')
            else:    
                ax.set_ylabel('Absorbance')
            if Save[0]:    
                df.to_csv(f"{SaveDir}{Exp_name}_{filename}_Abs.csv", index=False)
        
        
        ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))

        plt.tight_layout()
        # Save the data
        if Save[1]:    
            fig.savefig(SaveDir + Exp_name + '.png', bbox_inches='tight', dpi=300)
        
        return plt
    
    except Exception as e:
        print(f"An error occurred in Plot_dfs function : {e}")
        return None    