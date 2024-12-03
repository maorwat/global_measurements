import pytimber
import yaml
import re
import pandas as pd
import numpy as np
import lossmaps as lm

from global_package.utils import *

class BLMs:
    def __init__(self, start_time, end_time, beam, spark, option=3, 
                 filter_out_collimators=True, peaks=None, threshold=0.8, bottleneck=None):
        """
        Initialize the BLMs class for Beam Loss Monitor data processing.

        Parameters:
        - start_time: Start time for data.
        - end_time: End time for data.
        - beam: Beam identifier 'B1H', 'B2H', 'B1V', or 'B2V'.
        - spark: Spark session for data processing.
        - option: Strategy for bottleneck selection (default: 3).
        - filter_out_collimators: Whether to exclude collimator losses in bottleneck selection (default: True).
        - peaks: Identified peaks from collimators (default: None).
        - threshold: Noise threshold for filtering signals (default: 0.8).
        - bottleneck: Specific bottleneck to set (default: None).
        """
        # Convert time to UTC
        self.start_time, self.end_time = get_utc_time(start_time, end_time)
        # Parse beam and plane
        self.beam, self.plane = parse_beam_plane(beam)
        # Set attributes
        self.spark = spark
        self.option = option
        self.threshold = threshold
        self.peaks = peaks.peaks
        self.filter_out_collimators = filter_out_collimators
        self.bottleneck = bottleneck

        # Load BLM data
        self.get_blm_df()

        # Automatically find bottleneck if not provided or incorrect
        if self.bottleneck not in self.blm_mqx_df.columns or self.bottleneck == '':
            self.find_highest_losses()

    def get_blm_df(self):
        """
        Load BLM data and prepare the DataFrame for analysis.
        """
        data = lm.blm.get_BLM_data(self.start_time, self.end_time, spark=self.spark)
        names = lm.blm.get_BLM_names(self.start_time, spark=self.spark)

        vals_df = pd.DataFrame(data['vals'].tolist(), columns=names)
        data_expanded = pd.concat([data['timestamp'], vals_df], axis=1)
        data_expanded['time'] = pd.to_datetime(data_expanded['timestamp']).astype(int) / 10**9
        data_expanded = data_expanded.drop(columns=['timestamp']).sort_values(by='time').reset_index(drop=True)

        self.blm_mqx_df = data_expanded

    def find_highest_losses(self):
        """
        Identify the bottleneck with the highest losses based on the specified option.
        """
        # Clean the data
        df_cleaned = self.blm_mqx_df.dropna(axis=1, how='all').drop('time', axis=1)
        if self.filter_out_collimators:
            df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains('TC|TDI')]

        # Normalize the data for easier noise detection
        normalized_df = df_cleaned / df_cleaned.max()
        noise_flags = normalized_df.apply(self._is_noise, axis=0)
        filtered_df = normalized_df.loc[:, noise_flags]

        # Identify bottleneck based on the chosen option
        if self.option == 1:
            self.bottleneck = normalized_df[filtered_df.columns].iloc[self.peaks.iloc[-1]].idxmax()
        elif self.option == 2:
            df_cleaned[filtered_df.columns] = df_cleaned[filtered_df.columns].apply(lambda col: col - col.min())
            self.bottleneck = df_cleaned[filtered_df.columns].iloc[self.peaks.iloc[-1]].idxmax()
        elif self.option == 3:
            self.bottleneck = normalized_df[filtered_df.columns].apply(lambda col: col.max() - col.min()).idxmax()

    def _is_noise(self, signal):
        """
        Determine if a signal is noise based on the threshold.
        """
        return (signal.max() - signal.min()) > self.threshold