import pytimber
import yaml
import re
import tfs
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from global_package.utils import *

class Collimators():
    
    def __init__(self,
                start_time,
                end_time,
                beam,
                tfs_path,
                spark,
                gap_step=0.15,
                reference_collimator=None,
                emittance=3.5e-6,
                yaml_path='/eos/project-c/collimation-team/machine_configurations/LHC_run3/2023/colldbs/injection.yaml'):
        """
        Initialize the Collimators class for gap and loss data processing.

        Parameters:
        - start_time: Start time for data.
        - end_time: End time for data.
        - beam: Beam identifier 'B1H', 'B2H', 'B1V', or 'B2V'.
        - tfs_path:
        - spark: Spark session for data processing.
        - gap_step:
        - reference_collimator:
        - emittance:
        - yaml_path:
        """
        # Convert time to UTC
        self.start_time, self.end_time = get_utc_time(start_time, end_time)
        
        # Initialize attributes
        self.ldb = pytimber.LoggingDB(spark_session=spark)
        self.yaml_path = yaml_path
        self.emittance = emittance
        self.gap_step = gap_step
        self.tfs_path = tfs_path
        
        # Parse beam and plane
        self.beam, self.plane = parse_beam_plane(beam)
        
        # Load necessary data
        self.load_data(reference_collimator)
        self.load_blm()
        self.find_peaks()
        self.load_optics_and_rescale()
    
    # Load collimators 
    def load_data(self, reference_collimator):
        """
        Load and process collimator data.

        Parameters:
        - reference_collimator: 
        """
        # If reference collimator not given, find
        if reference_collimator is not None:
            self.load_data_given_ref_col(reference_collimator)
        else:
            # Load the file  
            with open(self.yaml_path, 'r') as file:
                f = yaml.safe_load(file)

            # Create a data frame
            cols = pd.DataFrame(f['collimators'][self.beam]).loc[['angle']].T
            cols = cols.reset_index().rename(columns={'index': 'name'})

            if self.plane == 'horizontal': angle = 0
            elif self.plane == 'vertical': angle = 90
            cols = cols[cols['angle'] == angle].dropna()

            # Get a list of collimator names to load from timber
            names = cols['name'].str.upper().to_list()
            for i, name in enumerate(names): names[i]=name+':MEAS_LVDT_GD'
                
            # Load from timber
            cols = self.ldb.get(names, self.start_time, self.end_time)
            
            # Find collimators that moved
            moved_collimators = self.find_moved_collimators(cols)
            
            # Find the reference collimator
            self.process_dataframe(moved_collimators)

    def load_data_given_ref_col(self, reference_collimator):
        """
        Load data for the given reference collimator.

        Parameters:
        - reference_collimator: 
        """    
        col = self.ldb.get(reference_collimator+':MEAS_LVDT_GD', self.start_time, self.end_time)

        self.ref_col_df = pd.DataFrame({
                        'time': col[reference_collimator+':MEAS_LVDT_GD'][0],
                        'gap': col[reference_collimator+':MEAS_LVDT_GD'][1]
                    })

        self.ref_col_df['time'] = self.ref_col_df['time'].astype(int)
        self.reference_collimator = reference_collimator+':MEAS_LVDT_GD'
        
    def find_moved_collimators(self, cols):
        """
        Identify collimators that moved during the time interval.

        Parameters:
        - cols: Dictionary containing collimator gap data
        """
        # Create a dataframe
        data = {'time': cols[next(iter(cols))][0]}
        moved_collimators = pd.DataFrame(data)
        moved_collimators['time'] = moved_collimators['time'].astype(int) # Change time to int to enable merging

        # Find all the collimators that moved 
        # TODO: a bit inefficient, maybe change??
        for key, (timestamps, values) in cols.items():
            # Compute the difference between consecutive values
            differences = np.diff(values)

            # Check if there are any non-zero differences
            if np.any(differences > 0.1):
                # Create a new df
                df = pd.DataFrame({
                    'time': cols[key][0],
                    key: cols[key][1]
                })
                df['time'] = df['time'].astype(int)
                # Add the new collimator data by merging
                merged = pd.merge(moved_collimators, df, on='time')
                moved_collimators = merged.copy()
                    
        return moved_collimators
        
    def process_dataframe(self, df):
        """
        Process the DataFrame to determine the reference collimator.

        Parameters:
        - df:
        """
        # Round all columns except 'time' to one decimal place
        columns_to_round = [col for col in df.columns if col != 'time']
        df[columns_to_round] = df[columns_to_round].round(1)

        # Determine the column with the most unique values
        # Exclude 'time' column from consideration
        unique_counts = {col: df[col].nunique() for col in columns_to_round}
        self.reference_collimator = max(unique_counts, key=unique_counts.get)
        
        # Store reference collimator data
        self.ref_col_df = df[['time', self.reference_collimator]].rename(columns={self.reference_collimator: 'gap'})
    
    def load_blm(self):
        """
        Load the BLM data associated with the reference collimator.
        """
        # Split the string by '.' and ':'
        split_string = self.reference_collimator.replace(':', '.').split('.')
        
        # Try with 'E10'
        try:
            blm_string = 'BLMTI.0'+re.search(r'([0-9]+[RL][0-9]+)', split_string[1]).group()+'.'+split_string[2]+'E10_'+self.reference_collimator.split(':')[0]+':LOSS_RS09'
            blm = self.ldb.get(blm_string, self.start_time, self.end_time)
            df = pd.DataFrame({'time': blm[blm_string][0],
                            'loss': blm[blm_string][1]})
        # If not check 'I10'
        except:
            blm_string = 'BLMTI.0'+re.search(r'([0-9]+[RL][0-9]+)', split_string[1]).group()+ '.'+split_string[2]+'I10_'+self.reference_collimator.split(':')[0]+':LOSS_RS09'

            blm = self.ldb.get(blm_string, self.start_time, self.end_time)
            df = pd.DataFrame({'time': blm[blm_string][0],
                            'loss': blm[blm_string][1]})

        df['time'] = df['time'].astype(int)

        # Make attributes
        self.reference_collimator_blm = blm_string
        self.ref_col_blm_df = df
        
    def find_peaks(self):
        """
        Identify peaks in the BLM data corresponding to collimator movements.
        """
        # TODO: improve
        combined_data = pd.concat([self.ref_col_blm_df, self.ref_col_df.drop(self.ref_col_df.columns[0], axis=1)], axis=1)
        # Normalize the 'loss' column
        combined_data['loss'] /= combined_data['loss'].max()
        
        # Calculate the difference in 'gap'
        gap_diff = combined_data['gap'].diff().abs()  # Get the absolute difference

        # Create a mask for where the difference is greater than 0.1
        split_mask = gap_diff > self.gap_step
        dfs = []

        # Track start index for each split segment
        segment_start_indices = []
        start_index = 0
        for i in range(len(split_mask)):
            if split_mask[i]:
                # If there's a split point, slice the DataFrame
                segment = combined_data.iloc[start_index:i]
                if segment.shape[0] > 5:  # Ensure segment has more than 5 rows
                    dfs.append(segment)
                    segment_start_indices.append(start_index)  # Record the start index of the segment
                start_index = i  # Update the start index for the next segment

        # Append the last segment after the loop
        dfs.append(combined_data.iloc[start_index:])
        segment_start_indices.append(start_index)

        # List to store peak indices relative to df1
        peak_indices = []

        # Identify valid start and end range by skipping empty (no peaks) segments
        valid_start = 0
        valid_end = len(dfs)

        # Skip empty segments at the start
        for i, segment in enumerate(dfs):
            peaks, _ = find_peaks(segment.loss, prominence=0.01)
            if peaks.size > 0:
                valid_start = i
                break

        # Skip empty segments at the end
        for i, segment in enumerate(reversed(dfs)):
            peaks, _ = find_peaks(segment.loss, prominence=0.01)
            if peaks.size > 0:
                valid_end = len(dfs) - i
                break

        # Process only the valid range of segments
        for i, (segment, segment_start) in enumerate(zip(dfs[valid_start:valid_end], segment_start_indices[valid_start:valid_end])):
            # Use find_peaks to locate peaks
            peaks, _ = find_peaks(segment.loss)
            
            if peaks.size > 0:  # If peaks are found in the segment
                # Get the index of the highest peak within the segment
                highest_peak_idx = peaks[np.argmax(segment.loss.iloc[peaks])]

                # Convert to index in df1 and store
                peak_indices.append(segment_start + highest_peak_idx)
            else:  # No peaks found in the segment
                # Take the highest value in the segment
                highest_peak_index = np.argmax(segment.loss)

                # Convert to index in df1 and store
                peak_indices.append(segment_start + highest_peak_index)

        peak_indices = [i for i in peak_indices if i != 0 and i != self.ref_col_df.shape[0]]

        corresponding_times = combined_data['time'].iloc[peak_indices].values

        self.peaks = pd.DataFrame({'time': corresponding_times,
                                   'peaks': peak_indices})

    def load_optics_and_rescale(self):
        """
        Load optics data and rescale collimator gaps to beam sigma.
        """
        # Reset indices
        self.ref_col_df.reset_index(drop=True, inplace=True)
        # Find collimator gaps corresponding to blm peaks
        gaps = self.ref_col_df.iloc[self.peaks.peaks.values].gap
        
        # Load optics data
        optics_data = tfs.read(self.tfs_path)
        if self.plane == 'horizontal': column = 'BETX'
        elif self.plane == 'vertical': column = 'BETY'   
        beta = optics_data.loc[optics_data.NAME == self.reference_collimator.split(':')[0], column].values[0]
        gamma = tfs.reader.read_headers(self.tfs_path)['GAMMA']
        
        # Calculate sigma and rescale gaps
        self.sigma = np.sqrt(beta * self.emittance / gamma)
        self.gaps = gaps * 1e-3 / self.sigma

    def change_reference_collimator(self, reference_collimator):
        """
        Change the reference collimator and reload associated data.

        Parameteres:
        - reference collimator: 
        """
        self.load_data_given_ref_col(reference_collimator)
        self.load_blm()
        self.load_optics_and_rescale()
        self.find_peaks()