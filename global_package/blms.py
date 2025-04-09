import pytimber
import yaml
import re
import pandas as pd
import numpy as np
import lossmaps as lm

from global_package.utils import *

class BLMs:
    def __init__(self, start_time, end_time, beam, spark, option=2, 
                 filter_out_collimators=False, peaks=None, threshold=0.8, 
                 bottleneck=None, reference_collimator_blm=None):
        """
        Initialize the BLMs class for Beam Loss Monitor (BLM) data processing.

        This class handles data retrieval, filtering, and bottleneck identification 
        for beam loss analysis.

        Parameters:
        - start_time (datetime): Start time for data retrieval.
        - end_time (datetime): End time for data retrieval.
        - beam (str): Beam identifier ('B1H', 'B2H', 'B1V', or 'B2V').
        - spark (SparkSession): Spark session for distributed data processing.
        - option (int, optional): Strategy for bottleneck selection (default: 2).
        - filter_out_collimators (bool, optional): Whether to exclude collimator losses 
        in bottleneck selection (default: False).
        - peaks (object, optional): Object containing identified peaks from collimators (default: None).
        - threshold (float, optional): Noise threshold for filtering signals (default: 0.8).
        - bottleneck (str, optional): Predefined bottleneck identifier (default: None).
        - reference_collimator_blm (str, optional): Reference collimator to exclude (default: None).

        Behavior:
        - Converts start and end times to UTC.
        - Parses beam identifier to extract beam and plane.
        - Loads BLM data using `get_blm_df()`.
        - If no valid bottleneck is provided, it determines the highest-loss location using `find_highest_losses()`.
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
        self.reference_collimator_blm = reference_collimator_blm

        # Load BLM data
        self.get_blm_df()

        # Automatically find bottleneck if not provided or incorrect
        if self.bottleneck not in self.blm_mqx_df.columns or self.bottleneck == '':
            try: self.find_highest_losses(filter = self.bottleneck)
            except: self.find_highest_losses()

    def get_blm_df(self):
        """
        Load BLM data and prepare the DataFrame for analysis.
        """

        # Load data using lossmaps
        data = lm.blm.get_BLM_data(self.start_time, self.end_time, spark=self.spark)
        names = lm.blm.get_BLM_names(self.start_time, spark=self.spark)

        vals_df = pd.DataFrame(data['vals'].tolist(), columns=names)
        data_expanded = pd.concat([data['timestamp'], vals_df], axis=1)
        data_expanded['time'] = pd.to_datetime(data_expanded['timestamp']).astype(int) / 10**9
        data_expanded = data_expanded.drop(columns=['timestamp']).sort_values(by='time').reset_index(drop=True)

        self.blm_mqx_df = data_expanded

    def find_highest_losses(self, filter=None):
        """
        Identify the location with the highest beam losses based on the selected method.

        The function processes BLM data by cleaning, normalising, and filtering noise 
        before determining the bottleneck using Option 2:

        - Option 2: Adjusts signals by subtracting their minimum value,
            then finds the highest losses at the last peak.
        """
        # Clean the data
        df_cleaned = self.blm_mqx_df.dropna(axis=1, how='all').drop('time', axis=1)
        if filter:
            df_cleaned = df_cleaned[df_cleaned.columns[df_cleaned.columns.str.contains(filter)]]

        # Don't consider the reference collimator
        if self.reference_collimator_blm is not None:
            try: df_cleaned = df_cleaned.drop(self.reference_collimator_blm, axis=1)
            except: pass
        # If specified, filter out all the collimators
        if self.filter_out_collimators:
            df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains('TC|TDI')]

        # Normalise the data for easier noise detection
        normalized_df = df_cleaned / df_cleaned.max()
        # Remove noise
        noise_flags = normalized_df.apply(self._is_noise, axis=0)
        filtered_df = normalized_df.loc[:, noise_flags]

        # Identify bottleneck based on the chosen option
        if self.option == 1:
            self.bottleneck = normalized_df[filtered_df.columns].iloc[self.peaks.iloc[-1]].idxmax()
        # This is the option that actually works well ish
        elif self.option == 2:
            # Substract from all BLMs the minimum signal
            df_cleaned[filtered_df.columns] = df_cleaned[filtered_df.columns].apply(lambda col: col - col.min())
            # Fnd the highest losses at the last peak - last blow up
            self.bottleneck = df_cleaned[filtered_df.columns].iloc[self.peaks.iloc[-1]].idxmax()
        elif self.option == 3:
            self.bottleneck = normalized_df[filtered_df.columns].apply(lambda col: col.max() - col.min()).idxmax()

    def _is_noise(self, signal):
        """
        Check if the given signal is considered noise based on a defined threshold.

        A signal is classified as noise if the difference between its maximum 
        and minimum values exceeds the specified threshold.

        Parameters:
            signal (array-like): The input signal to be evaluated.

        Returns:
            bool: True if the signal is considered noise, False otherwise.
        """
        return (signal.max() - signal.min()) > self.threshold