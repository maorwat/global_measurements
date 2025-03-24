import pytimber
import yaml
import re
import pandas as pd
import numpy as np

from global_package.utils import *

class Bunches:
    def __init__(self, start_time, end_time, beam, spark, peaks, all_peaks):
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
        self.smooth_bunch_intensity()
        self.get_protons_lost()

    def get_bunches(self):
        """
        Retrieve and process bunch intensity data.
        """
        name = f'LHC.BCTFR.A6R4.{self.beam.upper()}:BUNCH_INTENSITY'
        bunches = self.ldb.get(name, self.start_time, self.end_time)

        # Convert data to DataFrame
        time_array = bunches[name][0]
        bunch_intensity_array = np.stack(bunches[name][1], axis=1)
        df = pd.DataFrame({'time': time_array})

        bunch_columns = [f'Bunch {n}' for n in range(bunch_intensity_array.shape[0])]
        bunch_data_df = pd.DataFrame(bunch_intensity_array.T, columns=bunch_columns)

        columns = [col for col in df if (df[col] == 0).all()]
        df = pd.concat([df, bunch_data_df], axis=1).drop(columns=columns)

        # Identify the bunch with the maximum intensity drop
        drops = df.iloc[0] - df.iloc[-1]
        self.bunch = drops.idxmax()
        self.bunch_df = df[['time', self.bunch]]

    def smooth_bunch_intensity(self):
        """
        Smooth the bunch intensity using the peaks as segmentation points.
        """
        bunch_intensity = self.bunch_df[self.bunch].values
        time = self.bunch_df['time'].values

        segments = []
        start_idx = 0

        for idx in self.all_peaks.peaks.values:
            if idx > start_idx:
                segments.append((start_idx, idx))
            start_idx = idx
        if start_idx < len(bunch_intensity):
            segments.append((start_idx, len(bunch_intensity)))

        smoothed_bunch_intensity = np.copy(bunch_intensity)
        steps = np.copy(bunch_intensity)
        
        time_to_average = 15

        for start, end in segments:
            if end > start:
                steps[start:end] = np.mean(bunch_intensity[start:end])
                if (end-start) > time_to_average:
                    smoothed_bunch_intensity[start:start+time_to_average] = np.mean(bunch_intensity[start:start+time_to_average])
                    smoothed_bunch_intensity[end-time_to_average:end] = np.mean(bunch_intensity[end-time_to_average:end])
                else:
                    smoothed_bunch_intensity[start:end] = np.mean(bunch_intensity[start:end])

        self.smoothed_bunch_intensity = np.nan_to_num(
            smoothed_bunch_intensity, nan=np.nanmean(bunch_intensity))
        self.steps = np.nan_to_num(
            steps, nan=np.nanmean(bunch_intensity))

    def get_protons_lost(self):
        """
        Calculate the protons lost based on smoothed bunch intensity and peaks.
        """
        # Descending order
        drops = np.where(np.diff(self.steps) < 0)[0]
        corresponding_times = self.all_peaks.time.values

        protons_lost = [
            self.smoothed_bunch_intensity[drops[i]] - self.smoothed_bunch_intensity[drops[i]+1] 
            for i in range(len(drops))
            ]

        df = pd.DataFrame(
            {'protons_lost': protons_lost, 
             'time': corresponding_times}
             )
        self.protons_lost = pd.merge(self.peaks, df, on='time', how='left')