from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import ReceptiveFieldMapping

class ReceptiveFieldMapping_VBN(ReceptiveFieldMapping):

    def __init__(self, ecephys_session, col_pos_x='x_position', col_pos_y='y_position', trial_duration=0.25,
                 minimum_spike_count=10.0, mask_threshold=0.5, **kwargs):
        
        self._ecephys_session = ecephys_session
        self._unit_ids = None
        self._unit_filter = kwargs.get('filter', None)
        self._params = kwargs.get('params', None)
        self._unit_count = None
        self._stim_table = None
        self._conditionwise_statistics = None
        self._presentationwise_statistics = None
        self._presentationwise_spikes = None
        self._conditionwise_psth = None
        self._stimulus_conditions = None

        self._spikes = None
        self._stim_table_spontaneous = None
        self._stimulus_key = kwargs.get('stimulus_key', None)
        self._running_speed = None
        # self._sweep_events = None
        # self._mean_sweep_events = None
        #  self._sweep_p_values = None
        self._metrics = None

        # start and stop times of blocks for the relevant stimulus. Used by the overall_firing_rate functions that only
        # need to be calculated once, but not accessable to the user
        self._block_starts = None
        self._block_stops = None

        # self._module_name = None  # TODO: Remove, .name() should be hardcoded

        self._psth_resolution = kwargs.get('psth_resolution', 0.001)

        # Duration a sponteous stimulus should last for before it gets included in the analysis.
        self._spontaneous_threshold = kwargs.get('spontaneous_threshold', 100.0)

        # Roughly the length of each stimulus duration, used for calculating spike statististics
        self._trial_duration = trial_duration

        # Keeps track of preferred stimulus_condition_id for each unit
        self._preferred_condition = {}
        self._pos_x = None
        self._pos_y = None

        self._rf_matrix = None

        self._col_pos_x = col_pos_x
        self._col_pos_y = col_pos_y

        self._minimum_spike_count = minimum_spike_count
        self._mask_threshold = mask_threshold