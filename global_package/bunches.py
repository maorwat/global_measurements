import pytimber
import yaml
import re
import pandas as pd
import numpy as np

from global_package.utils import *

class Bunches:
    def __init__(self, start_time, end_time, beam, spark, peaks, all_peaks, bunch=None):
        """
        Initialize the Bunches class for processing bunch intensity data.

        Parameters:
        - start_time: Start time for data.
        - end_time: End time for data.
        - beam: Beam identifier 'B1H', 'B2H', 'B1V', or 'B2V'.
        - spark: Spark session for data processing.
        - peaks: Identified peaks from collimators.
        - all_peaks: Combined peaks from collimators and bottlenecks.
        """
        # Convert time to UTC
        self.start_time, self.end_time = get_utc_time(start_time, end_time)
        self.ldb = pytimber.LoggingDB(spark_session=spark)
        self.beam, self.plane = parse_beam_plane(beam.upper())

        # Set attributes
        self.all_peaks = all_peaks
        self.peaks = peaks

        # Load and process bunch intensity data
        self.get_bunches()
        self.find_used_bunch(bunch)
        self.smooth_bunch_intensity()
        self.get_protons_lost()

    def get_bunches(self):
        """
        Retrieve and process bunch intensity data.

        This method fetches bunch intensity data from the database, 
        converts it into a DataFrame, removes empty columns, and 
        identifies the bunch with the highest intensity drop over time.
        """
        # Construct string for retrieving bunch intensity data
        name = f'LHC.BCTFR.A6R4.{self.beam.upper()}:BUNCH_INTENSITY'
        bunches = self.ldb.get(name, self.start_time, self.end_time)

        # Extract time and intensity data
        time_array = bunches[name][0]
        bunch_intensity_array = np.stack(bunches[name][1], axis=1)
        # Create a DataFrame with time values
        df = pd.DataFrame({'time': time_array})

        # Assign column names for each bunch
        bunch_columns = [f'Bunch {n}' for n in range(bunch_intensity_array.shape[0])]
        bunch_data_df = pd.DataFrame(bunch_intensity_array.T, columns=bunch_columns)

        # Remove columns where all values are zero
        columns = [col for col in df if (df[col] == 0).all()]
        self.all_bunches = pd.concat([df, bunch_data_df], axis=1).drop(columns=columns)

    def find_used_bunch(self, bunch):
        # Compute the drop in intensity for each bunch and find the maximum drop
        try: 
            self.bunch = f'Bunch {bunch}'
            self.bunch_df = self.all_bunches[['time', self.bunch]]
        except:
            drops = self.all_bunches.iloc[0] - self.all_bunches.iloc[-1]
            self.bunch = drops.idxmax()
            self.bunch_df = self.all_bunches[['time', self.bunch]]

    def smooth_bunch_intensity(self):
        """
        Smooth the bunch intensity using detected peaks as segmentation points.

        This method averages the bunch intensity within segments defined by peaks 
        to reduce fluctuations. The beginning and end of each segment are 
        smoothed over a short window to facilitate calculating intensity difference at blow up.
        """
        bunch_intensity = self.bunch_df[self.bunch].values
        time = self.bunch_df['time'].values

        segments = []
        start_idx = 0

        # Split into segments based on detected peaks
        for idx in self.all_peaks.peaks.values:
            if idx > start_idx:
                segments.append((start_idx, idx))
            start_idx = idx
        if start_idx < len(bunch_intensity):
            segments.append((start_idx, len(bunch_intensity)))

        # Initialize smoothed arrays
        smoothed_bunch_intensity = np.copy(bunch_intensity)
        steps = np.copy(bunch_intensity)
        
        # Number of points (seconds) to average over
        time_to_average = 15

        for start, end in segments:
            if end > start:
                # Apply stepwise averaging within segments
                steps[start:end] = np.mean(bunch_intensity[start:end])
                # Average out the intensities around the peaks
                if (end-start) > time_to_average:
                    smoothed_bunch_intensity[start:start+time_to_average] = np.mean(bunch_intensity[start:start+time_to_average])
                    smoothed_bunch_intensity[end-time_to_average:end] = np.mean(bunch_intensity[end-time_to_average:end])
                # If short, average out the whole segment
                else:
                    smoothed_bunch_intensity[start:end] = np.mean(bunch_intensity[start:end])

        # Replace NaN values with the mean intensity to avoid computation issues
        self.smoothed_bunch_intensity = np.nan_to_num(
            smoothed_bunch_intensity, nan=np.nanmean(bunch_intensity))
        self.steps = np.nan_to_num(
            steps, nan=np.nanmean(bunch_intensity))

    def get_protons_lost(self):
        """
        Calculate protons lost based on smoothed bunch intensity and peak locations.

        This method identifies intensity drops over time and associates 
        them with known peak times to estimate proton losses.
        """
        # Identify indices where the intensity decreases (blow up events)
        drops = np.where(np.diff(self.steps) < 0)[0]
        corresponding_times = self.all_peaks.time.values

        # Compute the proton losses at these points
        protons_lost = [
            self.smoothed_bunch_intensity[drops[i]] - self.smoothed_bunch_intensity[drops[i]+1] 
            for i in range(len(drops))
            ]

        # Create a DataFrame with proton loss data
        df = pd.DataFrame(
            {'protons_lost': protons_lost, 
             'time': corresponding_times}
             )
        
        # Merge with detected peaks to align losses with event times
        self.protons_lost = pd.merge(self.peaks, df, on='time', how='left')