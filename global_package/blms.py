import pytimber
import yaml
import re
import pandas as pd
import numpy as np
import lossmaps as lm

from global_package.utils import *

class BLMs():
    
    def __init__(self,
                start_time,
                end_time,
                beam,
                spark,
                option=1,
                filter_out_collimators=True,
                peaks=None,
                threshold=0.8,
                bottleneck=None):
        
        # Convert time to UTC
        self.start_time, self.end_time = get_utc_time(start_time, end_time)
        # Find plane and beam from the given string
        self.beam, self.plane = parse_beam_plane(beam)
        self.bottleneck = bottleneck
        self.spark = spark
        self.threshold = threshold
        self.peaks = peaks.peaks
        self.filter_out_collimators = filter_out_collimators
        self.option=option
        
        self.get_blm_df()

        if self.bottleneck is None:
            self.find_highest_losses()
        
    def get_blm_df(self):
        # TODO: improve
        # Now pass the localized and converted timestamps to the function
        data = lm.blm.get_BLM_data(self.start_time, self.end_time, spark=self.spark)
        names = lm.blm.get_BLM_names(self.start_time, spark=self.spark)

        vals_df = pd.DataFrame(data['vals'].tolist(), columns=names)

        # Concatenate 'timestamp' with the new DataFrame
        data_expanded = pd.concat([data['timestamp'], vals_df], axis=1)

        data_expanded['time'] = pd.to_datetime(data_expanded['timestamp']).astype(int) / 10**9

        # Drop the old timestamp column if needed
        data_expanded = data_expanded.drop(columns=['timestamp'])

        # Reorder the columns if desired
        data_expanded = data_expanded[['time'] + [col for col in data_expanded.columns if col != 'time']]

        # Sort the DataFrame by 'column_name' in increasing order
        data_expanded = data_expanded.sort_values(by='time', ascending=True)

        # Reset index if needed to maintain a clean, sequential index
        data_expanded = data_expanded.reset_index(drop=True)
        self.blm_mqx_df = data_expanded
        
    def find_highest_losses(self):
    
        # Drop Nans
        df_cleaned = self.blm_mqx_df.dropna(axis=1, how='all').drop('time', axis=1)
        if self.filter_out_collimators:
            df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains('TC|TDI')]
        # Normalise to identify noise more easily
        normalized_df = df_cleaned.copy()
        columns_to_normalize = [col for col in df_cleaned.columns if col != 'time']
        normalized_df[columns_to_normalize] = df_cleaned[columns_to_normalize] / df_cleaned[columns_to_normalize].max()
        # Identify noise
        noise_flags = normalized_df.apply(self.is_noise, axis=0)
        new_df = normalized_df.loc[:, noise_flags]
        # Identify highest losses at the last step of collimator gap
        if self.option == 1:
            self.bottleneck = normalized_df[new_df.columns].iloc[self.peaks.iloc[-1]].idxmax()
        elif self.option == 2:
            self.bottleneck = df_cleaned[new_df.columns].iloc[self.peaks.iloc[-1]].idxmax()
        elif self.option == 3:
            self.bottleneck = normalized_df[new_df.columns].apply(lambda col: col.max() - col.min()).idxmax()

    def is_noise(self, signal):
        diff = signal.max()-signal.min()
        return diff > self.threshold