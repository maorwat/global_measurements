from scipy.signal import find_peaks
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

    def everything_figure(self):
        """
        Generate a comprehensive figure showing gap, loss, and bunch intensity over time.
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add Gap trace (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=self.collimators.ref_col_df.time,
                y=self.collimators.ref_col_df.gap,
                mode='lines',
                name='Gap'
            ),
            secondary_y=False
        )

        # Add Loss traces (secondary y-axis)
        normalized_loss = self.collimators.ref_col_blm_df.loss / self.collimators.ref_col_blm_df.loss.max()
        fig.add_trace(
            go.Scatter(
                x=self.collimators.ref_col_blm_df.time,
                y=normalized_loss,
                mode='lines',
                name='Loss at reference collimator'
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                x=self.collimators.ref_col_blm_df.time.iloc[self.collimators.peaks.peaks.values],
                y=normalized_loss.iloc[self.collimators.peaks.peaks.values],
                mode='markers',
                name='Peaks'
            ),
            secondary_y=True
        )

        # Add Bunch intensity traces (tertiary y-axis)
        fig.add_trace(
            go.Scatter(
                x=self.bunches.bunch_df.time,
                y=self.bunches.smoothed_bunch_intensity,
                mode='lines',
                name='Smoothed Bunch Intensity',
                yaxis="y3"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.bunches.bunch_df.time,
                y=self.bunches.bunch_df[self.bunches.bunch],
                mode='lines',
                name='Bunch Intensity',
                yaxis="y3"
            )
        )

        # Add Loss at bottleneck trace (secondary y-axis)
        normalized_bottleneck_loss = self.blms.blm_mqx_df[self.blms.bottleneck] / self.blms.blm_mqx_df[self.blms.bottleneck].max()
        fig.add_trace(
            go.Scatter(
                x=self.blms.blm_mqx_df.time,
                y=normalized_bottleneck_loss,
                mode='lines',
                name='Loss at bottleneck'
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                x=self.blms.blm_mqx_df.time.iloc[self.all_peaks.peaks.values],
                y=normalized_bottleneck_loss.iloc[self.all_peaks.peaks.values],
                mode='markers',
                name='All peaks'
            ),
            secondary_y=True
        )

        # Update layout with a tertiary y-axis
        fig.update_layout(
            width=1000,
            height=600
            xaxis=dict(title='Time'),
            yaxis=dict(title='Gap'),  # Primary y-axis
            yaxis2=dict(title='Loss', overlaying='y', side='right', tickfont=dict(size=10)),  # Secondary y-axis
            yaxis3=dict(  # Tertiary y-axis
                title="Bunch Intensity",
                titlefont=dict(color="blue", size=10),
                tickfont=dict(color="blue", size=9),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.85
            )
        )

        return fig
    
    def _normalise_again(self, array):
        """
        Normalize an array by dividing by its maximum value.
        """
        return array / array.max()

    def _normalise_blm_data(self, df_blm, column):
        """
        Normalize BLM data by dividing loss values by protons lost.

        Parameters:
        - df_blm: DataFrame containing BLM data.
        - column: The column name in the DataFrame to normalize.
        """
        loss_values = df_blm[column].iloc[self.collimators.peaks.peaks.values].values
        protons_lost_values = self.bunches.protons_lost.protons_lost.values

        # Perform the normalized division
        return loss_values / protons_lost_values

    def normalised_losses_figure(self):
        """
        Create a figure for normalized losses at the collimator and bottleneck.
        """
        # Normalize losses
        normalised_col_blm = self._normalise_blm_data(self.collimators.ref_col_blm_df, 'loss')
        normalised_bottleneck_blm = self._normalise_blm_data(self.blms.blm_mqx_df, self.blms.bottleneck)

        # Create figure
        fig = go.Figure()

        # Add normalized collimator trace
        fig.add_trace(
            go.Scatter(
                x=self.collimators.gaps,
                y=self._normalise_again(normalised_col_blm),
                mode='lines+markers',
                name='Collimator'
            )
        )

        # Add normalized bottleneck trace
        fig.add_trace(
            go.Scatter(
                x=self.collimators.gaps,
                y=self._normalise_again(normalised_bottleneck_blm),
                mode='lines+markers',
                name='Bottleneck'
            )
        )

        # Update layout
        fig.update_layout(
            width=600,
            height=600
        )

        return fig

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
            layout=Layout(width='400px')
        )

        self.reference_collimator_label = Label(
            value='',
            description='Reference collimator:',
            style={'description_width': 'initial'},
            layout=Layout(width='400px')
        )

        self.bottleneck_label = Label(
            value='',
            description='Bottleneck:',
            style={'description_width': 'initial'},
            layout=Layout(width='400px')
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
            [self.beam_dropdown, self.tfs_file_chooser, self.analyse_button],
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
            [self.reference_collimator_input, self.bottleneck_input, self.reference_collimator_label, self.bottleneck_label],
            layout=Layout(
                justify_content="space-around",
                align_items="center",
                gap="20px",
                border="2px solid lightgray",
                padding="10px",
                width="100%"
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

        # Arrange HBoxes in a VBox with custom styles
        self.layout_box = VBox(
            [self.row1, self.row2, self.row3, self.row4],
            layout=Layout(
                align_items="center",
                gap="15px",
                border="2px lightgray",
                padding="10px",
                width="90%"
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

    def on_analyse_button_clicked(self, b):

        start_time = self.convert_to_datetime(self.start_date_picker, self.start_time_input)
        end_time = self.convert_to_datetime(self.end_date_picker, self.end_time_input)
        beam = self.beam_dropdown.value
        print('Setting time...')
        self.select_time_and_beam(start_time, end_time, beam)

        print('Loading collimators...')
        tfs_path = self.tfs_file_chooser.selected
        self.load_collimators(tfs_path)
        self.reference_collimator_label.value = f'Reference collimator: {self.collimators.reference_collimator}'

        print('Loading blms...')
        self.load_blms()
        self.bottleneck_label.value = f'Bottleneck: {self.blms.bottleneck}'
        print('Loading bunch intensities...')
        self.load_bunches()

        print('Creating figures...')
        fig1 = self.everything_figure()
        fig2 = self.normalised_losses_figure()
        self.row4.children = [go.FigureWidget(fig1), go.FigureWidget(fig2)]

    def show(self):

        display(self.layout_box)
        
