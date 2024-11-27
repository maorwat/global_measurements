import pytimber
import yaml
import re
import pandas as pd
import numpy as np

from global_package.utils import *

class Bunches():
    
    def __init__(self,
                start_time,
                end_time,
                beam,
                spark,
                peaks,
                all_peaks,
                bottleneck=None):
        
        # Convert time to UTC
        self.start_time, self.end_time = get_utc_time(start_time, end_time)
        
        self.ldb = pytimber.LoggingDB(spark_session=spark)
        # Find plane and beam from the given string
        self.beam, self.plane = parse_beam_plane(beam)
        self.beam = self.beam.upper()
        
        self.all_peaks = all_peaks
        self.peaks = peaks
        
        self.get_bunches()
        self.smooth_bunch_intensity()
        self.get_protons_lost()
        
    def get_bunches(self):
    
        if self.beam == 'B1': name = 'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY'
        elif self.beam == 'B2': name = 'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY'

        bunches = self.ldb.get(name, self.start_time, self.end_time)

        # Extract time and bunch intensity arrays from the dictionary
        time_array = bunches[name][0]
        bunch_intensity_array = np.stack(bunches[name][1], axis=1)

        # Initialize the DataFrame with the time array
        df = pd.DataFrame({'time': time_array})

        # Create a DataFrame for the bunch intensity data, transposed so each bunch is a column
        bunch_data_df = pd.DataFrame(bunch_intensity_array.T, columns=[f'Bunch {n}' for n in range(bunch_intensity_array.shape[0])])

        # Concatenate the time DataFrame and the bunch data DataFrame along columns
        df = pd.concat([df, bunch_data_df], axis=1)

        # Remove entries with only zeros
        df = df.drop(columns=[col for col in df if (df[col] == 0).all()])

        # Calculate the difference between the first and last values for each column, ignoring "Time"
        # Keep only the numeric columns (ignores "Time")
        drops = df.iloc[0] - df.iloc[-1]

        # Find the column with the maximum decrease
        self.bunch = drops.idxmax()
        self.bunch_df = df[['time', self.bunch]]
    
    def smooth_bunch_intensity(self):   
    
        bunch_intensity = self.bunch_df[self.bunch].values
        time = self.bunch_df['time'].values

        # Create segments based on filtered transition indices
        segments = []
        start_idx = 0

        for idx in self.all_peaks.peaks.values:
            if idx > start_idx:  # Ensure segment has at least one point
                segments.append((start_idx, idx))
            start_idx = idx

        # Append the final segment if it has data
        if start_idx < len(bunch_intensity):
            segments.append((start_idx, len(bunch_intensity)))

        # Smooth each step segment by averaging within the segment
        smoothed_bunch_intensity = np.copy(bunch_intensity)

        for start, end in segments:
            if end > start:  # Ensure segment is non-empty
                smoothed_bunch_intensity[start:end] = np.mean(bunch_intensity[start:end])

        # Handle any remaining NaNs if they appear
        self.smoothed_bunch_intensity = np.nan_to_num(smoothed_bunch_intensity, nan=np.nanmean(bunch_intensity))
        
    def get_protons_lost(self):
    
        unique_values = np.unique(self.smoothed_bunch_intensity)
        unique_values = unique_values[::-1]  # Sort unique values in decreasing order

        # Step 2: Use indices to retrieve corresponding times from blm_data.time
        corresponding_times = self.all_peaks.time.values

        protons_lost = []

        for i in range(len(unique_values)-1):
            protons_lost.append(unique_values[i]-unique_values[i+1])

        # Step 3: Create a DataFrame with unique values and corresponding times
        df = pd.DataFrame({
            'protons_lost': protons_lost,
            'time': corresponding_times
        })

        self.protons_lost = pd.merge(self.peaks, df, on='time', how='left')