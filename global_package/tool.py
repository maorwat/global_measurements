from scipy.signal import find_peaks
import numpy as np
import pandas as pd

from global_package.collimators import Collimators
from global_package.bunches import Bunches
from global_package.blms import BLMs

class Tool():
    
    def __init__(self):
        
        pass

    def find_all_peaks_test(self, prominence, min_separation):

        # Find peaks in the normalized columns
        all_peaks1, _ = find_peaks(self.collimators.ref_col_blm_df.loss / self.collimators.ref_col_blm_df.loss.max(), prominence=prominence, distance=min_separation)
        all_peaks2, _ = find_peaks(self.blms.blm_mqx_df[self.blms.bottleneck] / self.blms.blm_mqx_df[self.blms.bottleneck].max(), prominence=prominence, distance=min_separation)

        # Combine `all_peaks1` and `all_peaks2`
        candidate_peaks = np.concatenate([all_peaks1, all_peaks2])
        
        # Start with all `peaks`, ensuring they're always included
        final_peaks = list(self.collimators.peaks.peaks.values)
        
        # Add candidate peaks if they are far enough from existing peaks
        for candidate in np.sort(candidate_peaks):
            if all(abs(candidate - existing_peak) >= min_separation for existing_peak in final_peaks):
                final_peaks.append(candidate)
        
        # Sort final peaks
        final_peaks = np.sort(final_peaks)
        
        # Get corresponding times for final peaks
        corresponding_times = self.blms.blm_mqx_df['time'].iloc[final_peaks].values

        # Create a DataFrame with the results
        self.all_peaks = pd.DataFrame({
            'time': corresponding_times,
            'peaks': final_peaks
        })
    
    def create_bunches(self):

        self.bunches = Bunches()