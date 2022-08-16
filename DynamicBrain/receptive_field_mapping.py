from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import ReceptiveFieldMapping
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import *
from allensdk.brain_observatory.chisquare_categorical import chisq_from_stim_table
import xarray as xr
import warnings
import numpy as np
from collections.abc import Collection
import types
from scipy.optimize import curve_fit, leastsq, least_squares

class ReceptiveFieldMapping_VBN(ReceptiveFieldMapping):

    def __init__(self, ecephys_session, col_pos_x='position_x', col_pos_y='position_y', trial_duration=0.25,
                 minimum_spike_count=10.0, mask_threshold=0.5, **kwargs):
        
        units = ecephys_session.get_units()
        channels = ecephys_session.get_channels()
        ecephys_session.units = units.merge(channels, left_on='peak_channel_id', right_index=True)
        ecephys_session.presentationwise_spike_counts = types.MethodType(presentationwise_spike_counts, ecephys_session)
        ecephys_session._filter_owned_df = types.MethodType(_filter_owned_df, ecephys_session)
        self._ecephys_session = ecephys_session
        self._unit_ids = None
        self._unit_filter = kwargs.get('filter', None)
        self._params = kwargs.get('params', None)
        self._unit_count = None
        stimulus_presentations = self._ecephys_session.stimulus_presentations
        stimulus_presentations = stimulus_presentations[stimulus_presentations['stimulus_block']==2]
        self._stim_table = stimulus_presentations
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
    
    @property
    def presentationwise_statistics(self):
        """Returns a table of the spike-counts for every stimulus_presentation_id
        , unit_id pair.
        Returns
        -------
        presentationwise_statistics: pd.DataFrame
            MultiIndex : unit_id, stimulus_presentation_id
            Columns : spike_count, stimulus_condition_id, running_speed 
        """
        if self._presentationwise_statistics is None:
            # for each presentation_id and unit_id get the spike_counts across the entire duration. Since there is only
            # a single bin we can drop time_relative_to_stimulus_onset.
            df = self.ecephys_session.presentationwise_spike_counts(
                bin_edges=np.array([0.0, self.trial_duration]),
                stimulus_presentation_ids=self.stim_table.index.values,
                unit_ids=self.unit_ids
            ).to_dataframe().reset_index(level='time_relative_to_stimulus_onset', drop=True)
        
            self._presentationwise_statistics = df

        return self._presentationwise_statistics

    @property
    def metrics(self):
        if self._metrics is None:
            logger.info('Calculating metrics for ' + self.name)
            unit_ids = self.unit_ids
            metrics_df = self.empty_metrics_table()

            if len(self.stim_table) > 0:
                metrics_df.loc[:, ['azimuth_rf',
                                   'elevation_rf',
                                   'width_rf',
                                   'height_rf',
                                   'area_rf',
                                   'p_value_rf',
                                   'on_screen_rf',
                                   'is_inverted'
                                   ]] = [self._get_rf_stats(unit) for unit in unit_ids]
                # metrics_df['firing_rate_rf'] = [self._get_overall_firing_rate(unit) for unit in unit_ids]
                # metrics_df['fano_rf'] = [self._get_fano_factor(unit, self._get_preferred_condition(unit))
                #                          for unit in unit_ids]
                # metrics_df['time_to_peak_rf'] = [self._get_time_to_peak(unit, self._get_preferred_condition(unit))
                #                                  for unit in unit_ids]
                # metrics_df['lifetime_sparseness_rf'] = [self._get_lifetime_sparseness(unit) for unit in unit_ids]
                # metrics_df.loc[:, ['run_pval_rf', 'run_mod_rf']] = \
                #         [self._get_running_modulation(unit, self._get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df.dropna(axis=1, how='all')

        return self._metrics

    
    def _get_stim_table_stats(self):
        """ Extract azimuths and elevations from stimulus table."""

        self._pos_y = np.sort(self._stim_table.loc[self._stim_table[self._col_pos_y]
                                                           != 'null'][self._col_pos_y].unique())
        self._pos_x = np.sort(self._stim_table.loc[self._stim_table[self._col_pos_x]
                                                           != 'null'][self._col_pos_x].unique())


    def _response_by_stimulus_position(self, dataset, presentations, row_key=None, column_key=None, unit_key='unit_id',
                                       time_key='time_relative_to_stimulus_onset', spike_count_key='spike_count'):
        """ Calculate the unit's response to different locations
        of the Gabor patch
        Returns
        -------
        dataset : xarray
            dataset of receptive fields
        """

        if row_key is None:
            row_key = self._col_pos_y
        if column_key is None:
            column_key = self._col_pos_x

        dataset = dataset.copy()
        dataset[spike_count_key] = dataset.sum(dim=time_key)
        dataset = dataset.drop(time_key)

        dataset = dataset.assign_coords(position_x=('stimulus_presentation_id',
                                self.stim_table.loc[:, 'position_x']))
        dataset = dataset.assign_coords(position_y=('stimulus_presentation_id',
                                self.stim_table.loc[:, 'position_y']))
        #dataset[row_key] = presentations.loc[:, row_key]
        #dataset[column_key] = presentations.loc[:, column_key]
        dataset = dataset.to_dataframe()

        dataset = dataset.reset_index(unit_key).groupby([row_key, column_key, unit_key]).sum()

        return dataset.to_xarray()


    def _get_rf_stats(self, unit_id):
        """ Calculate a variety of metrics for one unit's receptive field
        Parameters
        ----------
        unit_id : int
            unique ID for the unit of interest
        Returns
        -------
        azimuth :
            preferred azimuth in degrees, based on center of mass of thresholded RF
        elevation :
            preferred elevation in degrees, based on center of mass of thresholded RF
        width :
            receptive field width in degrees, based on Gaussian fit
        height :
            receptive field height in degrees, based on Gaussian fit
        area :
            receptive field area in degrees^2, based on thresholded RF area
        p_value :
            probability that a significant receptive field is present, based on categorical chi-square test
        on_screen :
            True if the receptive field is away from the screen edge, based on Gaussian fit
        """
        rf = self._get_rf(unit_id)
        
        spikes_per_trial = self.presentationwise_statistics.xs(unit_id, level=1)['spike_counts'].values

        if np.sum(spikes_per_trial) < self._minimum_spike_count:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False

        p_value = chisq_from_stim_table(self.stim_table, [self._col_pos_x, self._col_pos_y],
                                        np.expand_dims(spikes_per_trial,1))

        #print(self._params)
        #exit()
        # is_inverted = is_rf_inverted(rf)
        # if is_inverted:
        #     rf = invert_rf(rf)
        
        (offset, peak_height, center_y, center_x, width_y, width_x), success = fit_2d_gaussian_with_offset(rf)

        rf_thresh, azimuth, elevation, area = threshold_rf(rf, self._mask_threshold)
        is_inverted = False
        if np.sign(peak_height)<0 or is_rf_inverted(rf_thresh):
            is_inverted = True
            rf = invert_rf(rf)

        #Now try this again now that we've inverted the rf if necessary
        rf_thresh, azimuth, elevation, area = threshold_rf(rf, self._mask_threshold)

        on_screen = rf_on_screen(rf, center_y, center_x)

        height_deg = convert_pixels_to_degrees(width_y)
        width_deg = convert_pixels_to_degrees(width_x)
        azimuth_deg = convert_azimuth_to_degrees(azimuth)
        elevation_deg = convert_elevation_to_degrees(elevation)
        area_deg = convert_pixel_area_to_degrees(area)

        return azimuth_deg, elevation_deg, width_deg, height_deg, area_deg, p_value[0], on_screen, is_inverted

    
def presentationwise_spike_counts(
        self,
        bin_edges,
        stimulus_presentation_ids,
        unit_ids,
        binarize=False,
        dtype=None,
        large_bin_size_threshold=0.001,
        time_domain_callback=None
    ):
        ''' Build an array of spike counts surrounding stimulus onset per
        unit and stimulus frame.
        Parameters
        ---------
        bin_edges : numpy.ndarray
            Spikes will be counted into the bins defined by these edges.
            Values are in seconds, relative to stimulus onset.
        stimulus_presentation_ids : array-like
            Filter to these stimulus presentations
        unit_ids : array-like
            Filter to these units
        binarize : bool, optional
            If true, all counts greater than 0 will be treated as 1. This
            results in lower storage overhead, but is only reasonable if bin
            sizes are fine (<= 1 millisecond).
        large_bin_size_threshold : float, optional
            If binarize is True and the largest bin width is greater than
            this value, a warning will be emitted.
        time_domain_callback : callable, optional
            The time domain is a numpy array whose values are trial-aligned bin
            edges (each row is aligned to a different trial). This optional
            function will be applied to the time domain before counting spikes.
        Returns
        -------
        xarray.DataArray :
            Data array whose dimensions are stimulus presentation, unit,
            and time bin and whose values are spike counts.
        '''

        stimulus_presentations = self._filter_owned_df(
            'stimulus_presentations',
            ids=stimulus_presentation_ids)
        units = self._filter_owned_df('units', ids=unit_ids)

        largest_bin_size = np.amax(np.diff(bin_edges))
        if binarize and largest_bin_size > large_bin_size_threshold:
            warnings.warn(
                'You\'ve elected to binarize spike counts, but your maximum '
                f'bin width is {largest_bin_size:2.5f} seconds. '
                'Binarizing spike counts with such a large bin width can '
                'cause significant loss of accuracy! '
                'Please consider only binarizing spike counts '
                f'when your bins are <= {large_bin_size_threshold} '
                'seconds wide.'
            )

        bin_edges = np.array(bin_edges)
        domain = build_time_window_domain(
            bin_edges,
            stimulus_presentations['start_time'].values,
            callback=time_domain_callback)

        out_of_order = np.where(np.diff(domain, axis=1) < 0)
        if len(out_of_order[0]) > 0:
            out_of_order_time_bins = \
                [(row, col) for row, col in zip(out_of_order)]
            raise ValueError("The time domain specified contains out-of-order "
                             f"bin edges at indices: {out_of_order_time_bins}")

        ends = domain[:, -1]
        starts = domain[:, 0]
        time_diffs = starts[1:] - ends[:-1]
        overlapping = np.where(time_diffs < 0)[0]

        if len(overlapping) > 0:
            # Ignoring intervals that overlaps multiple time bins because
            # trying to figure that out would take O(n)
            overlapping = [(s, s + 1) for s in overlapping]
            warnings.warn("You've specified some overlapping time intervals "
                          f"between neighboring rows: {overlapping}, "
                          "with a maximum overlap of"
                          f" {np.abs(np.min(time_diffs))} seconds.")

        tiled_data = build_spike_histogram(
            domain,
            self.spike_times,
            units.index.values,
            dtype=dtype,
            binarize=binarize
        )

        stim_presentation_id = stimulus_presentations.index.values

        tiled_data = xr.DataArray(
            name='spike_counts',
            data=tiled_data,
            coords={
                'stimulus_presentation_id': stim_presentation_id,
                'time_relative_to_stimulus_onset': (bin_edges[:-1] +
                                                    np.diff(bin_edges) / 2),
                'unit_id': units.index.values
            },
            dims=['stimulus_presentation_id',
                  'time_relative_to_stimulus_onset',
                  'unit_id']
        )

        return tiled_data


def _filter_owned_df(self, key, ids=None, copy=True):
        df = getattr(self, key)

        if copy:
            df = df.copy()

        if ids is None:
            return df

        ids = coerce_scalar(
            ids, f'a scalar ({ids}) was '
                 f'provided as ids, filtering to a single row of {key}.')

        df = df.loc[ids]

        if df.shape[0] == 0:
            warnings.warn(f'filtering to an empty set of {key}!')

        return df


def build_time_window_domain(bin_edges, offsets, callback=None):
    callback = (lambda x: x) if callback is None else callback
    domain = np.tile(bin_edges[None, :], (len(offsets), 1))
    domain += offsets[:, None]
    return callback(domain)

def build_spike_histogram(time_domain,
                          spike_times,
                          unit_ids,
                          dtype=None,
                          binarize=False):

    time_domain = np.array(time_domain)
    unit_ids = np.array(unit_ids)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1, unit_ids.size),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, 1:]

    for ii, unit_id in enumerate(unit_ids):
        data = np.array(spike_times[unit_id])

        start_positions = np.searchsorted(data, starts.flat)
        end_positions = np.searchsorted(data, ends.flat, side="right")
        counts = (end_positions - start_positions)

        tiled_data[:, :, ii].flat = counts > 0 if binarize else counts

    return tiled_data


def coerce_scalar(value, message, warn=False):
    if not isinstance(value, Collection) or isinstance(value, str):
        if warn:
            warnings.warn(message)
        return [value]
    return value


def rf_on_screen(rf, center_y, center_x):
    """Checks whether the receptive field is on the screen, given the center location."""
    return 0 < center_y < rf.shape[0]-1 and 0 < center_x < rf.shape[1]-1

def _gaussian_function_2d(peak_height, center_y, center_x, width_y, width_x):
    """Returns a 2D Gaussian function
    
    Parameters
    ----------
    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis
    
    Returns
    -------
    f(x,y) : function
        Returns the value of the distribution at a particular x,y coordinate
    
    """
    
    return lambda x,y: peak_height \
                       * np.exp( \
                       -( \
                         ((center_y - y) / width_y)**2 \
                       + ((center_x - x) / width_x)**2 \
                        ) \
                        / 2 \
                        )


def gaussian_moments_2d(data):
    """Finds the moments of a 2D Gaussian distribution, given an input matrix
    
    Parameters
    ----------
    data : numpy.ndarray
        2D matrix
        
    Returns
    -------
    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis
    """
    
    total = data.sum()
    height = data.max()
    
    Y, X = np.indices(data.shape)
    center_y = (Y*data).sum()/total
    center_x = (X*data).sum()/total

    if np.isnan(center_y) or np.isinf(center_y) or np.isnan(center_x) or np.isinf(center_x):
        return None

    col = data[:, int(center_x)]    
    row = data[int(center_y), :]

    width_y = np.sqrt(np.abs((np.arange(row.size)-center_y)**2*row).sum()/row.sum())
    width_x = np.sqrt(np.abs((np.arange(col.size)-center_x)**2*col).sum()/col.sum())

    return height, center_y, center_x, width_y, width_x


def fit_2d_gaussian(matrix):
    """Fits a receptive field with a 2-dimensional Gaussian distribution

    Parameters
    ----------
    matrix : numpy.ndarray
        2D matrix of spike counts

    Returns
    -------
    parameters - tuple
        peak_height : peak of distribution
        center_y : y-coordinate of distribution center
        center_x : x-coordinate of distribution center
        width_y : width of distribution along x-axis
        width_x : width of distribution along y-axis
    success - bool
        True if a fit was found, False otherwise
    """

    params = gaussian_moments_2d(matrix)
    if params is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan), False

    errorfunction = lambda p: np.ravel(_gaussian_function_2d(*p)(*np.indices(matrix.shape)) - matrix)
    fit_params, ier = leastsq(errorfunction, params)
    success = True if ier < 5 else False

    return fit_params, success


def _gaussian_function_2d_with_offset(offset, peak_height, center_y, center_x, width_y, width_x):
    """Returns a 2D Gaussian function
    
    Parameters
    ----------
    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis
    
    Returns
    -------
    f(x,y) : function
        Returns the value of the distribution at a particular x,y coordinate
    
    """
    
    return lambda x,y: offset + peak_height \
                       * np.exp( \
                       -( \
                         ((center_y - y) / width_y)**2 \
                       + ((center_x - x) / width_x)**2 \
                        ) \
                        / 2 \
                        )


def gaussian_moments_2d_with_offset(data):
    """Finds the moments of a 2D Gaussian distribution, given an input matrix
    
    Parameters
    ----------
    data : numpy.ndarray
        2D matrix
        
    Returns
    -------
    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis
    """
    
    total = data.sum()
    offset = np.percentile(data, 25)
    height = data.max()-offset
    
    Y, X = np.indices(data.shape)
    center_y = (Y*data).sum()/total
    center_x = (X*data).sum()/total

    if np.isnan(center_y) or np.isinf(center_y) or np.isnan(center_x) or np.isinf(center_x):
        return None

    col = data[:, int(center_x)]    
    row = data[int(center_y), :]

    width_y = np.sqrt(np.abs((np.arange(row.size)-center_y)**2*row).sum()/row.sum())
    width_x = np.sqrt(np.abs((np.arange(col.size)-center_x)**2*col).sum()/col.sum())

    return offset, height, center_y, center_x, width_y, width_x

def fit_2d_gaussian_with_offset(matrix):
    """Fits a receptive field with a 2-dimensional Gaussian distribution

    Parameters
    ----------
    matrix : numpy.ndarray
        2D matrix of spike counts

    Returns
    -------
    parameters - tuple
        peak_height : peak of distribution
        center_y : y-coordinate of distribution center
        center_x : x-coordinate of distribution center
        width_y : width of distribution along x-axis
        width_x : width of distribution along y-axis
    success - bool
        True if a fit was found, False otherwise
    """

    params = gaussian_moments_2d_with_offset(matrix)
    if params is None or any([np.isnan(p) for p in params]):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), False

    errorfunction = lambda p: np.ravel(_gaussian_function_2d_with_offset(*p)(*np.indices(matrix.shape)) - matrix)
    # fit_params, ier = leastsq(errorfunction, params),
    #                         bounds=([0, -matrix.max(), -np.inf, -np.inf, -np.inf, -np.inf],
    #                                 [matrix.max(), matrix.max(), np.inf, np.inf, np.inf, np.inf]))

    mat_mean = matrix.mean() if not np.isnan(matrix.mean()) else 0
    mat_max = matrix.max() if not np.isnan(matrix.max()) else np.inf
    mat_median = np.median(matrix) if not np.isnan(np.median(matrix)) else 0

    result = least_squares(errorfunction, params,
                            bounds=([0, -2*mat_max, -np.inf, -np.inf, 0, 0],
                                    [ mat_median+0.1, np.inf, np.inf, np.inf, 4.5, 4.5]))
    #success = True if ier < 5 else False
    success = result.success
    fit_params = result.x
    return fit_params, success

def fit_double_2d_gaussian(matrix):
    """Fits a receptive field with a 2-dimensional Gaussian distribution

    Parameters
    ----------
    matrix : numpy.ndarray
        2D matrix of spike counts

    Returns
    -------
    parameters - tuple
        peak_height : peak of distribution
        center_y : y-coordinate of distribution center
        center_x : x-coordinate of distribution center
        width_y : width of distribution along x-axis
        width_x : width of distribution along y-axis
    success - bool
        True if a fit was found, False otherwise
    """

    params = double_gaussian_moments_2d(matrix)
    if params is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan), False

    errorfunction = lambda p: np.ravel(_double_gaussian_function_2d(*p)(*np.indices(matrix.shape)) - matrix)
    fit_params, ier = leastsq(errorfunction, params)
    success = True if ier < 5 else False

    return fit_params, success


def double_gaussian_moments_2d(data):
    """Finds the moments of a 2D Gaussian distribution, given an input matrix
    
    Parameters
    ----------
    data : numpy.ndarray
        2D matrix
        
    Returns
    -------
    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis
    """
    
    total = data.sum()
    height = data.max()
    offset = data.mean()
    
    Y, X = np.indices(data.shape)
    center_y = (Y*data).sum()/total
    center_x = (X*data).sum()/total

    if np.isnan(center_y) or np.isinf(center_y) or np.isnan(center_x) or np.isinf(center_x):
        return None

    col = data[:, int(center_x)]    
    row = data[int(center_y), :]

    width_y = np.sqrt(np.abs((np.arange(row.size)-center_y)**2*row).sum()/row.sum())
    width_x = np.sqrt(np.abs((np.arange(col.size)-center_x)**2*col).sum()/col.sum())

    return offset, height, center_y, center_x, width_y, width_x,\
                   0, 0, 0, 0, 0

def _double_gaussian_function_2d(offset, peak_height1, center_y1, center_x1, width_y1, width_x1,
                                         peak_height2, center_y2, center_x2, width_y2, width_x2):
    """Returns a 2D Gaussian function
    
    Parameters
    ----------
    peak_height :
        peak of distribution
    center_y :
        y-coordinate of distribution center
    center_x :
        x-coordinate of distribution center
    width_y :
        width of distribution along x-axis
    width_x :
        width of distribution along y-axis
    
    Returns
    -------
    f(x,y) : function
        Returns the value of the distribution at a particular x,y coordinate
    
    """
    
    return lambda x,y: offset + peak_height1 \
                       * np.exp( \
                       -( \
                         ((center_y1 - y) / width_y1)**2 \
                       + ((center_x1 - x) / width_x1)**2 \
                        ) \
                        / 2 \
                        ) + \
                        peak_height2 \
                       * np.exp( \
                       -( \
                         ((center_y2 - y) / width_y2)**2 \
                       + ((center_x2 - x) / width_x2)**2 \
                        ) \
                        / 2 \
                        )