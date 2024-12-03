from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ipywidgets import widgets, Tab, VBox, HBox, Button, Layout, FloatText, DatePicker, Text, Dropdown, Label, GridBox
from IPython.display import display, Latex
from ipyfilechooser import FileChooser
from datetime import datetime, date

from global_package.collimators import Collimators
from global_package.bunches import Bunches
from global_package.blms import BLMs

class Tool():
    
    def __init__(self, spark, initial_path='/eos/project-c/collimation-team/machine_configurations/'):
        """
        Initialize the Tool class with a Spark session.
        
        Parameters:
        - spark: Spark session to be used for data loading and processing.
        """
        self.spark = spark
        self.initial_path = initial_path

        self.create_widgets()

    def select_time_and_beam(self, start_time, end_time, beam):
        """
        Set the time range and beam parameters for the analysis.

        Parameters:
        - start_time: Start time for the data.
        - end_time: End time for the data.
        - beam: Beam identifier 'B1H', 'B2H', 'B1V', or 'B2V'.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.beam = beam

    def load_collimators(self, 
                         tfs_path, 
                         gap_step=0.1, 
                         reference_collimator=None, 
                         emittance=3.5e-6, 
                         yaml_path='/eos/project-c/collimation-team/machine_configurations/LHC_run3/2023/colldbs/injection.yaml'):
        """
        Load collimator data.

        Parameters:
        - tfs_path: Path to TFS data files to identify beta at reference collimator.
        - gap_step: Minimum step size for gap values (default: 0.1).
        - reference_collimator: String for the reference collimator (default: None).
        - emittance: Normalised emittance value (default: 3.5e-6).
        - yaml_path: Path to the YAML file to read collimator names.
        """
        self.collimators = Collimators(
            start_time=self.start_time,
            end_time=self.end_time,
            beam=self.beam,
            tfs_path=tfs_path,
            spark=self.spark,
            gap_step=gap_step,
            reference_collimator=reference_collimator,
            emittance=emittance,
            yaml_path=yaml_path
        )

    def load_blms(self, option=2, filter_out_collimators=True, threshold=0.8, bottleneck=None):
        """
        Load Beam Loss Monitors (BLMs) data.

        Parameters:
        - option: Option identifier for finding the bottleneck, to remove that (default: 3).
        - filter_out_collimators: Whether to exclude collimator losses in bottleneck selection (default: True).
        - threshold: Threshold value to identify noise (default: 0.8).
        - bottleneck: String for the bottleneck (default: None).
        """
        self.blms = BLMs(
            start_time=self.start_time,
            end_time=self.end_time,
            beam=self.beam,
            spark=self.spark,
            option=option,
            filter_out_collimators=filter_out_collimators,
            peaks=self.collimators.peaks,  # Pass peaks identified by collimators
            threshold=threshold,
            bottleneck=bottleneck
        )

    def load_bunches(self, prominence=0.05, min_separation=5):
        """
        Load bunch intensity data and identify peaks.

        Parameters:
        - prominence: Prominence of peaks to be identified (default: 0.05).
        - min_separation: Minimum separation between peaks (default: 5 s).
        """
        # Identify peaks before loading bunches
        self.find_all_peaks(prominence, min_separation)

        self.bunches = Bunches(
            start_time=self.start_time,
            end_time=self.end_time,
            beam=self.beam,
            spark=self.spark,
            peaks=self.collimators.peaks,  # Use collimator peaks
            all_peaks=self.all_peaks         # Use combined peaks
        )

    def find_all_peaks(self, prominence, min_separation):
        """
        Identify peaks in the normalized data and combine peaks from two sources while ensuring minimum separation.

        Parameters:
        - prominence: The prominence of the peaks to find.
        - min_separation: Minimum distance between peaks.
        """
        # Find peaks in the normalized reference collimator data
        normalized_ref_loss = self._normalise_again(self.collimators.ref_col_blm_df.loss)
        all_peaks1, _ = find_peaks(normalized_ref_loss, prominence=prominence, distance=min_separation)

        # Find peaks in the normalized bottleneck data
        normalized_bottleneck_loss = self._normalise_again(self.blms.blm_mqx_df[self.blms.bottleneck])
        all_peaks2, _ = find_peaks(normalized_bottleneck_loss, prominence=prominence, distance=min_separation)

        # Combine the peaks from both sources
        candidate_peaks = np.concatenate([all_peaks1, all_peaks2])

        # Start with existing peaks to ensure inclusion
        final_peaks = list(self.collimators.peaks.peaks.values)

        # Add candidate peaks if they meet the minimum separation criterion
        for candidate in np.sort(candidate_peaks):
            if all(abs(candidate - existing_peak) >= min_separation for existing_peak in final_peaks):
                final_peaks.append(candidate)

        # Sort the final peaks
        final_peaks = np.sort(final_peaks)

        # Get corresponding times for the final peaks
        corresponding_times = self.blms.blm_mqx_df['time'].iloc[final_peaks].values

        # Store results in a DataFrame
        self.all_peaks = pd.DataFrame({
            'time': corresponding_times,
            'peaks': final_peaks
        })

    def everything_figure(self, selected_peaks=None):
        """
        Generate a comprehensive figure showing gap, loss, and bunch intensity over time.
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        reference_collimator = self.collimators.reference_collimator.split(':')[0]
        # Add Gap trace (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=self.collimators.ref_col_df.time,
                y=self.collimators.ref_col_df.gap,
                mode='lines',
                name=f'{reference_collimator} gap',
                line=dict(color='gray')
            ),
            secondary_y=False
        )

        # Add Bunch intensity traces (tertiary y-axis)
        #fig.add_trace(
        #    go.Scatter(
        #        x=self.bunches.bunch_df.time,
        #        y=self.bunches.smoothed_bunch_intensity/(10**8),
        #        mode='lines',
        #        name='Smoothed Bunch Intensity',
        #        yaxis="y3"
        #    )
        #)
        fig.add_trace(
            go.Scatter(
                x=self.bunches.bunch_df.time,
                y=self.bunches.bunch_df[self.bunches.bunch]/(10**8),
                mode='lines',
                name='Bunch Intensity',
                yaxis="y3",
                line=dict(color='#CD5C5C')
            )
        )
        reference_collimator_blm = self.collimators.reference_collimator_blm.split(':')[0]
        # Add Loss traces (secondary y-axis)
        normalized_loss = self.collimators.ref_col_blm_df.loss / self.collimators.ref_col_blm_df.loss.max()
        fig.add_trace(
            go.Scatter(
                x=self.collimators.ref_col_blm_df.time,
                y=normalized_loss,
                mode='lines',
                name=reference_collimator_blm,
                line=dict(color='#1f77b4')
            ),
            secondary_y=True
        )
        # Add Loss at bottleneck trace (secondary y-axis)
        normalized_bottleneck_loss = self.blms.blm_mqx_df[self.blms.bottleneck] / self.blms.blm_mqx_df[self.blms.bottleneck].max()
        fig.add_trace(
            go.Scatter(
                x=self.blms.blm_mqx_df.time,
                y=normalized_bottleneck_loss,
                mode='lines',
                name=self.blms.bottleneck,
                line=dict(color='orange')
            ),
            secondary_y=True
        )
        if selected_peaks is None: selected_peaks = self.collimators.peaks.peaks.values
        else: selected_peaks = self.collimators.peaks.peaks.values[selected_peaks]
        fig.add_trace(
            go.Scatter(
                x=self.collimators.ref_col_blm_df.time.iloc[selected_peaks],
                y=normalized_loss.iloc[selected_peaks],
                mode='markers',
                name='Peaks',
                marker=dict(symbol='star-open-dot', size=20, color='#2E8B57')
            ),
            secondary_y=True
        )

        #fig.add_trace(
        #    go.Scatter(
        #        x=self.blms.blm_mqx_df.time.iloc[self.all_peaks.peaks.values],
        #        y=normalized_bottleneck_loss.iloc[self.all_peaks.peaks.values],
        #        mode='markers',
        #        name='All peaks'
        #    ),
        #    secondary_y=True
        #)

        # Update layout with a tertiary y-axis
        fig.update_layout(
            width=1400,
            height=600,
            xaxis=dict(title='Time'),
            yaxis=dict(title=f'{reference_collimator} gap [\u03C3]'),  # Primary y-axis
            yaxis2=dict(
                title='BLM signal / Max. BLM signal', 
                overlaying='y', 
                side='right'),  # Secondary y-axis
            yaxis3=dict(  
                title="Bunch Intensity [10⁸ p]",
                anchor="free",
                overlaying="y",
                side="right",
                position=1), # Tertiary y-axis
            margin=dict(r=100),
            legend=dict(
            x=1.05,  # Position the legend at 80% of the width (from the left)
            xanchor='left',  # Align the legend to the left
            orientation='v')  # Arrange the legend items vertically
        )

        return fig
    
    def _normalise_again(self, array):
        """
        Normalize an array by dividing by its maximum value.
        """
        return array / array.max()

    def _normalise_blm_data(self, df_blm, column, selected_peaks=None):
        """
        Normalize BLM data by dividing loss values by protons lost.

        Parameters:
        - df_blm: DataFrame containing BLM data.
        - column: The column name in the DataFrame to normalize.
        """
        loss_values = df_blm[column].iloc[self.collimators.peaks.peaks.values].values
        protons_lost_values = self.bunches.protons_lost.protons_lost.values

        if selected_peaks is None:
            # Perform the normalized division
            return loss_values / protons_lost_values
        else: return loss_values[selected_peaks] / protons_lost_values[selected_peaks]

    def normalised_losses_figure(self, selected_peaks=None):
        """
        Create a figure for normalized losses at the collimator and bottleneck.
        """
        # Normalize losses
        normalised_col_blm = self._normalise_blm_data(self.collimators.ref_col_blm_df, 'loss', selected_peaks)
        normalised_bottleneck_blm = self._normalise_blm_data(self.blms.blm_mqx_df, self.blms.bottleneck, selected_peaks)
        if selected_peaks is None: selected_gaps = self.collimators.gaps
        else: selected_gaps = self.collimators.gaps.values[selected_peaks]

        # Create figure
        fig = go.Figure()

        # Add normalized collimator trace
        fig.add_trace(
            go.Scatter(
                x=selected_gaps,
                y=self._normalise_again(normalised_col_blm),
                mode='lines+markers',
                name='Collimator',
                line=dict(color='#2E8B57')
            )
        )

        # Add normalized bottleneck trace
        fig.add_trace(
            go.Scatter(
                x=selected_gaps,
                y=self._normalise_again(normalised_bottleneck_blm),
                mode='lines+markers',
                name='Bottleneck',
                line=dict(color='orange')
            )
        )

        # Update layout
        fig.update_layout(
            width=600,
            height=600,
            xaxis=dict(title=f'Collimator setting [\u03C3]'),
            yaxis=dict(title='Normalised BLM signal'),
        )
        
        intersection = self.find_trace_intersections(
            np.array(selected_gaps), 
            self._normalise_again(normalised_col_blm),
            self._normalise_again(normalised_bottleneck_blm))

        return fig, intersection
    
    def find_trace_intersections(self, x, y1, y2, method='linear'):
        """
        Finds the intersection points of two traces.

        Parameters:
        - x: Array-like, the x-values common to both traces.
        - y1: Array-like, the y-values of the first trace.
        - y2: Array-like, the y-values of the second trace.
        - method: Interpolation method, default is 'linear'.

        Returns:
        - x_int: x where the two traces intersect.
        """
        # Create interpolating functions
        f1 = interp1d(x, y1, kind=method, fill_value="extrapolate")
        f2 = interp1d(x, y2, kind=method, fill_value="extrapolate")
        
        # Define the difference function
        def diff_func(x_val):
            return f1(x_val) - f2(x_val)
        
        # Find roots (intersection points)
        for i in range(len(x) - 1):
            # Check if a sign change occurs in this interval
            if (y1[i] - y2[i]) * (y1[i+1] - y2[i+1]) <= 0:  # Possible root in [x[i], x[i+1]]
                root = root_scalar(diff_func, bracket=[x[i], x[i+1]], method='brentq')
                if root.converged:
                    # Calculate the corresponding y-value
                    x_int = root.root
        
        return x_int

    def create_widgets(self):

        self.beam_dropdown = Dropdown(
            options=['B1H', 'B2H', 'B1V', 'B2V'],
            description='Beam:'
        )

        self.start_date_picker = DatePicker(
            description='Start date:',
            value=date.today(),
            style={'description_width': 'initial'},
            layout=Layout(width='300px')
        )

        self.start_time_input = Text(
            description='Start time (HH:MM:SS):',
            value=datetime.now().strftime('%H:%M:%S'),
            placeholder='10:53:15',
            style={'description_width': 'initial'},
            layout=Layout(width='300px')
        )

        self.end_date_picker = DatePicker(
            description='End date:',
            value=date.today(),
            style={'description_width': 'initial'},
            layout=Layout(width='300px')
        )

        self.end_time_input = Text(
            description='End time (HH:MM:SS):',
            value=datetime.now().strftime('%H:%M:%S'),
            placeholder='10:53:15',
            style={'description_width': 'initial'},
            layout=Layout(width='300px')
        )

        self.progress_label = Label(
            value='',
            style={'description_width': 'initial'},
            layout=Layout(width='200px')
        )

        self.min_protons_lost_input = FloatText(
            value=1e6,
            description='Min number of protons lost:',
            style={'description_width': 'initial'},
            layout=Layout(width='300px')
        )

        self.reference_collimator_input = Text(
            value='',
            description='Reference collimator:',
            style={'description_width': 'initial'},
            layout=Layout(width='400px')
        )

        self.bottleneck_input = Text(
            value='',
            description='Bottleneck:',
            style={'description_width': 'initial'},
            layout=Layout(width='300px')
        )

        self.reference_collimator_label = Label(
            value='',
            description='Reference collimator:',
            style={'description_width': 'initial'},
            layout=Layout(width='300px')
        )

        self.bottleneck_label = Label(
            value='',
            description='Bottleneck BLM:',
            style={'description_width': 'initial'},
            layout=Layout(width='400px')
        )

        self.intersection_label = Label(
            value='',
            style={'description_width': 'initial'},
            layout=Layout(width='200px')
        )

        self.tfs_file_chooser = FileChooser(self.initial_path)

        self.analyse_button = Button(
            description="Analyse", 
            icon="star",
            style=widgets.ButtonStyle(button_color='pink')
        )
        self.analyse_button.on_click(self.on_analyse_button_clicked)

        # Arrange widgets in HBoxes with custom styles
        self.row1 = HBox(
            [self.start_date_picker, self.start_time_input, self.end_date_picker, self.end_time_input],
            layout=Layout(
                justify_content="space-around",
                align_items="center",
                gap="20px",
                border="2px solid lightgray",
                padding="10px",
                width="100%"
            )
        )

        self.row2 = HBox(
            [self.beam_dropdown, self.tfs_file_chooser, self.min_protons_lost_input, self.analyse_button],
            layout=Layout(
                justify_content="space-around",
                align_items="center",
                gap="20px",
                border="2px solid lightgray",
                padding="10px",
                width="100%"
            )
        )

        self.row3 = HBox(
            [self.progress_label, self.reference_collimator_input, self.bottleneck_input, self.reference_collimator_label, self.bottleneck_label, self.intersection_label],
            layout=Layout(
                justify_content="space-around",
                align_items="center",
                gap="20px",
                border="2px solid lightgray",
                padding="10px",
                width="100%"
            )
        )

        # Arrange HBoxes in a VBox with custom styles
        self.widgets_vbox = VBox(
            [self.row1, self.row2, self.row3],
            layout=Layout(
                align_items="center",
                border="2px lightgray",
                padding="0px",
                width="90%"
            )
        )   

        # Figure container
        self.row4 = HBox(
            [],
            layout=Layout(
                justify_content="space-around",
                align_items="center",
                gap="20px",
                border="2px solid lightgray",
                padding="10px",
                width="100%"
            )
        )

        self.multi_select = widgets.SelectMultiple(
            options=[],
            value=[],  # Default selected values
            disabled=False,
            layout=widgets.Layout(width='150px')
        )

        self.rescale_button = Button(
            description="Rescale", 
            icon="coffee",
            style=widgets.ButtonStyle(button_color='pink')
        )
        self.rescale_button.on_click(self.on_rescale_button_clicked)

        self.multi_select_box = VBox(
            [widgets.HTML("<h4>Points to analyse</h4>"), self.multi_select, self.rescale_button],
            layout=Layout(
                align_items="center",
                width="10%"
            )
        )

        self.widgets_hbox = HBox(
            [self.widgets_vbox, self.multi_select_box],
            layout=Layout(
                justify_content="space-around",
                align_items="center",
                gap="20px",
                border="2px solid lightgray",
                padding="10px",
                width="100%"
            )
        )

        # Arrange HBoxes in a VBox with custom styles
        self.layout_box = VBox(
            [self.widgets_hbox, self.row4],
            layout=Layout(
                align_items="center",
                border="2px lightgray",
                width="100%"
            )
        )

    def convert_to_datetime(self, date_picker, time_input):

        # Extract the selected date and time
        selected_date = date_picker.value  # Date from the DatePicker widget
        selected_time = time_input.value  # Time from the Text widget

        # Combine the date and time into a single string
        combined_datetime_str = f"{selected_date} {selected_time}"

        # Convert the combined string to a pandas datetime object
        time = pd.to_datetime(combined_datetime_str)

        return time
    
    def find_enough_protons_peaks(self):
        
        all_values = self.bunches.protons_lost.protons_lost.values
        self.valid_peaks = np.where(all_values > self.min_protons_lost_input.value)[0]

    def on_analyse_button_clicked(self, b):

        start_time = self.convert_to_datetime(self.start_date_picker, self.start_time_input)
        end_time = self.convert_to_datetime(self.end_date_picker, self.end_time_input)
        beam = self.beam_dropdown.value
        self.progress_label.value = 'Setting time...'
        self.select_time_and_beam(start_time, end_time, beam)

        self.progress_label.value = 'Loading collimators...'
        tfs_path = self.tfs_file_chooser.selected
        reference_collimator = self.reference_collimator_input.value.upper()
        self.load_collimators(tfs_path=tfs_path, reference_collimator=reference_collimator)
        reference_collimator = self.collimators.reference_collimator.split(':')[0]
        self.reference_collimator_label.value = f'Reference collimator: {reference_collimator}'
        self.multi_select.options = np.round(self.collimators.gaps, 2)

        self.progress_label.value = 'Loading blms...'
        bottleneck = self.bottleneck_input.value.upper()
        self.load_blms(bottleneck=bottleneck)
        self.bottleneck_label.value = f'Bottleneck BLM: {self.blms.bottleneck}'
        self.progress_label.value = 'Loading bunch intensities...'
        self.load_bunches()

        self.find_enough_protons_peaks()
        #self.multi_select.options = np.round(self.collimators.gaps.values, 2)[self.valid_peaks]
        self.multi_select.value = tuple(np.round(self.collimators.gaps.values, 2)[self.valid_peaks])
        self.progress_label.value = 'Creating figures...'
        fig1 = self.everything_figure(self.valid_peaks)
        fig2, intersection = self.normalised_losses_figure(self.valid_peaks)
        self.row4.children = [go.FigureWidget(fig1), go.FigureWidget(fig2)]
        self.intersection_label.value = f'Intersection: {intersection:.2f} [\u03C3]'

    def on_rescale_button_clicked(self, b):

        all_values = np.round(self.collimators.gaps.values, 2)
        selected_values = self.multi_select.value
        selected_peaks = [np.where(all_values == value)[0][0] for value in np.array(selected_values)]

        self.progress_label.value = 'Creating figures...'
        fig1 = self.everything_figure(selected_peaks)
        fig2, intersection = self.normalised_losses_figure(selected_peaks)
        self.row4.children = [go.FigureWidget(fig1), go.FigureWidget(fig2)]
        self.intersection_label.value = f'Intersection: {intersection:.2f} \u03C3'

    def show(self):

        display(self.layout_box)
        
