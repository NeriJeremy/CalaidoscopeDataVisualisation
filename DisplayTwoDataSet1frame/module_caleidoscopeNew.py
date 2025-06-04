import pandas as pd
import re
import os
import matplotlib.pyplot as plt


#function to store abs and fluo datas in a dictionary
def Open_txt(dir1, dir2):
    # Create empty dictionaries to store DataFrames for each directory
    dfs_dir1 = {}
    dfs_dir2 = {}

    # List of directories and their corresponding dictionaries
    directories = [(dir1, dfs_dir1), (dir2, dfs_dir2)]

    try:
        # Loop through both directories and their corresponding dictionaries
        for directory, dfs in directories:
            # Check if the directory exists
            if not os.path.isdir(directory):
                print(f"Error: The directory {directory} does not exist.")
                continue
            
            for filename in os.listdir(directory):
                # Check if the filename matches the pattern for 'abs' or 'fluo' files
                if re.search('(?:abs|fluo)(?:_exp\d?)?_\d{7}U1\.TXT_processed_spectra\.txt', filename):     
                    file_path = os.path.join(directory, filename)
                    
                    try:
                        # Read the file into a DataFrame
                        df = pd.read_csv(file_path, sep='\\s', engine='python')
                        # Add each DataFrame to the corresponding dictionary
                        dfs[filename] = df
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

        # Return the dictionaries for both directories
        return dfs_dir1, dfs_dir2

    except Exception as e:
        print(f"Error: {e}")
        return None, None 

#function to put headers on each df and keep only selected frames
def Column_adj(dfs_dir1, dfs_dir2, first_frame1, last_frame1, first_frame2, last_frame2):
    try:
        # Adjust columns for the DataFrames in the first directory (dfs_dir1)
        for filename, df in dfs_dir1.items():
            # Give a name to the columns, 'Wavelength' and one number for each frame
            df.columns = ['Wavelength'] + [str(i) for i in range(1, len(df.columns))]    
            # Filtering columns to keep 'Wavelength' and the chosen window
            columns_to_keep = ['Wavelength'] + [col for col in df if col != 'Wavelength' and first_frame1 <= int(col) <= last_frame1]
            # Filter the DataFrame based on the columns to keep
            df = df[columns_to_keep]
            dfs_dir1[filename] = df

        # Adjust columns for the DataFrames in the second directory (dfs_dir2)
        for filename, df in dfs_dir2.items():
            # Give a name to the columns, 'Wavelength' and one number for each frame
            df.columns = ['Wavelength'] + [str(i) for i in range(1, len(df.columns))]    
            # Filtering columns to keep 'Wavelength' and the chosen window
            columns_to_keep = ['Wavelength'] + [col for col in df if col != 'Wavelength' and first_frame2 <= int(col) <= last_frame2]
            # Filter the DataFrame based on the columns to keep
            df = df[columns_to_keep]
            dfs_dir2[filename] = df

        return dfs_dir1, dfs_dir2

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


#function to delete noise from laser(for fluo) and normalize datas
def Norm(dfs_dir1, dfs_dir2, Laser_405, Laser_561, Save_dir, Exp_name, Save_csv,
         Adj_abs_spectra, Adj_abs_high, Adj_abs_low, Adj_fluo_high,
         Adj_fluo_low, Adj_fluo_spectra, Norm_abs, Norm_fluo):
    
    try:        
        # Function to normalize and adjust a dictionary of DataFrames
        def Norm_dfs(dfs):
            for filename, df in dfs.items():
                
                # Normalize and adjust the abs data if the filename starts with 'abs'
                if filename.startswith('abs'):
                    # Normalize abs data
                    if Norm_abs:
                        columns_to_normalize = df.columns[df.columns != 'Wavelength']
                        # Apply normalization to the selected columns according to the A280nm values
                        abs_at_280 = df[df['Wavelength'] == 280].iloc[0]
                        for col in columns_to_normalize:
                            df[col] = df[col] / abs_at_280[col]
                    
                    # Adjust abs spectra if needed
                    if Adj_abs_spectra:
                        df = df[(((df['Wavelength'] >= Adj_abs_low) & (df['Wavelength'] <= Adj_abs_high)))]
                    
                # Normalize and adjust the fluo data if the filename starts with 'fluo'
                if filename.startswith('fluo'):
                    # Remove lasers noise (405 and 561)
                    if Laser_405:
                        df = df[~(((df['Wavelength'] >= 402) & (df['Wavelength'] <= 409)))]
                    if Laser_561:
                        df = df[~(((df['Wavelength'] >= 558) & (df['Wavelength'] <= 565)))]                
                    
                    # Adjust fluo spectra if needed
                    if Adj_fluo_spectra:
                        df = df[(((df['Wavelength'] >= Adj_fluo_low) & (df['Wavelength'] <= Adj_fluo_high)))]

                    # Normalize fluo data
                    if Norm_fluo:
                        columns_to_normalize = df.columns[df.columns != 'Wavelength']
                        # Apply Min-Max normalization to the selected columns
                        df[columns_to_normalize] = df[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
                
                # Apply the changes to the DataFrame in the dictionary
                dfs[filename] = df

                # Save the DataFrame as a CSV if needed
                if Save_csv:
                    df.to_csv(Save_dir + Exp_name + filename + '.csv')
            
            return dfs

        # Process both directories' DataFrames
        dfs_dir1 = Norm_dfs(dfs_dir1)
        dfs_dir2 = Norm_dfs(dfs_dir2)

        return dfs_dir1, dfs_dir2
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def Plot_dfs(dfs_dir1, dfs_dir2, Save_dir, Exp_name, Save_png, Norm_abs, Norm_fluo, ProtName1, ProtName2):
    try:
        # Create a 1x2 subplot (one for fluorescence, one for absorbance)
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Loop through both directories and plot their data
        for dfs, ProtName in [(dfs_dir1, ProtName1), (dfs_dir2, ProtName2)]:
            for filename, df in dfs.items():
                if filename.startswith('fluo'):  # Fluorescence data
                    for col in df.columns:
                        if col != 'Wavelength':
                            axs[0].plot(df['Wavelength'], df[col], label=f'{ProtName}')

                elif filename.startswith('abs'):  # Absorbance data
                    for col in df.columns:
                        if col != 'Wavelength':  # Skip 'Wavelength' column
                            axs[1].plot(df['Wavelength'], df[col], label=f'{ProtName}')

        # Set labels and titles for both subplots
        axs[0].set_xlabel('Wavelength')        
        axs[0].set_title('Fluorescence Emission')
        if Norm_fluo:
            axs[0].set_ylim(0, 1.1)
            axs[0].set_ylabel('Normalized Fluorescence Intensity')
        else:
            axs[0].set_ylabel('Fluorescence Intensity')
        axs[1].set_xlabel('Wavelength')
        if Norm_abs:
            axs[1].set_ylabel('Normalized Absorbance')
        else:    
            axs[1].set_ylabel('Absorbance')
        axs[1].set_title('Absorbance')

        # Add legends to both subplots
        axs[0].legend()
        axs[1].legend()

        # Show the plots
        plt.tight_layout()
        plt.show()

        # Save the figure if Save_png is True
        if Save_png:
            fig.savefig(Save_dir + Exp_name + '_combined.png')

        return plt

    except Exception as e:
        print(f"An error occurred: {e}")
        return None