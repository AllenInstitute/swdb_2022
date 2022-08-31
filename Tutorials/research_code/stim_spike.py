import allensdk
from allensdk.brain_observatory.\
    behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
import os
import numpy as np
import platform
platstring = platform.platform()

INIT_BLOCK = 0

def getCache(data_dirname):
    '''
    gives the cache object for visual behaviour neuropixels dataset
    Parameters:
        data_dirname: string 
            directory of the stored data 
    returns: 
        allensdk cache object 
    '''
    #data_dirname = 'visual-behavior-neuropixels'
    use_static = False
    if 'Darwin' in platstring or 'macOS' in platstring:
        # macOS 
        data_root = "/Volumes/Brain2022/"
    elif 'Windows'  in platstring:
        # Windows (replace with the drive letter of USB drive)
        data_root = "E:/"
    elif ('amzn' in platstring):
        # then on AWS
        data_root = "/data/"
        data_dirname = 'visual-behavior-neuropixels-data'
        use_static = True
    else:
        # then your own linux platform
        # EDIT location where you mounted hard drive
        data_root = "/home/koosha/Brain2022/"

    # get the cache location
    cache_dir = os.path.join(data_root, data_dirname)

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
    #cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(
    #            cache_dir=cache_dir, use_static_cache=use_static)

    return cache

def getSessionWithMostUnits(cache, area = "VISp", experience_level = "Familiar" , isi_violations = .5, amplitude_cutoff = .1, presence_ratio = .95):
    '''
    find the session with most number of units in a given brain area
    Parameters:
        cache: allensdk Cache
            cache object
        area: string (Default = "VISp") 
            brain area name 
        experience_level: string ({"Familiar, "Novel"}, Default = "Familiar")
            experience level of the mouse regarding shown images
        isi_violations: float (Default = .5)
            isi viloation metric for QC 
        amplitude_cutoff: float (Default = .1)
            amplitude_cutoff metric for QC 
        presence_ratio: float (Default = .1)
            presence ratio metric for QC 
    returns: 
        session object 
    '''
    units_table = cache.get_unit_table()
    ecephys_sessions_table = cache.get_ecephys_session_table()

    unit_by_session = units_table.join(ecephys_sessions_table,on = 'ecephys_session_id')
    unit_in = unit_by_session[(unit_by_session['structure_acronym'] == area) &\
                              (unit_by_session['experience_level'] == experience_level) &\
                              (unit_by_session['isi_violations'] < isi_violations)&\
                              (unit_by_session['amplitude_cutoff'] < amplitude_cutoff)&\
                              (unit_by_session['presence_ratio'] > presence_ratio)]
    unit_count = unit_in.groupby(["ecephys_session_id"]).count()
    familiar_session_with_most_in_units = unit_count.index[np.argmax(unit_count['ecephys_probe_id'])]
    
    session = cache.get_ecephys_session(ecephys_session_id=familiar_session_with_most_in_units)
    return session

def getStimSpikePair(session, area = "VISp", window = .25, isi_violations = .5, amplitude_cutoff = .1, presence_ratio = .95):
    '''
    find the session with most number of units in a given brain area
    Parameters:
        session: allensdk Session
            session object
        area: string (Default = "VISp") 
            brain area name 
        window: float (Default = .25)
            time window for counting spikes from stimulus onset in ms
        isi_violations: float (Default = .5)
            isi viloation metric for QC 
        amplitude_cutoff: float (Default = .1)
            amplitude_cutoff metric for QC 
        presence_ratio: float (Default = .1)
            presence ratio metric for QC 
    Returns: 
        Numpy 2D array (Num units, Num images), dtype = float64
            spike count for each unit and image 
        Numpy 1D array (Num images), dtype = int64
            assigned category number to each image 
    '''
    session_units = session.get_units()
    session_channels = session.get_channels()
    # And accosiate each unit with the channel on which it was found with the largest amplitude
    units_by_channels= session_units.join(session_channels,on = 'peak_channel_id')

    this_units = units_by_channels[(units_by_channels.structure_acronym == area)\
                                   &(units_by_channels['isi_violations'] < isi_violations)\
                                   &(units_by_channels['amplitude_cutoff'] < amplitude_cutoff)\
                                   &(units_by_channels['presence_ratio'] > presence_ratio)]

    this_spiketimes = dict(zip(this_units.index, [session.spike_times[ii] for ii in this_units.index]))
    active_stims = session.stimulus_presentations[session.stimulus_presentations.stimulus_block == INIT_BLOCK ]
    
    [unq,cat]= np.unique(active_stims.image_name, return_inverse = True)

    X = np.zeros((len(active_stims),len(this_spiketimes)))
    # This Loop is a little slow...be patient
    # Loop Through both trials and units, counting the number of spikes
    for jj,key in enumerate(this_spiketimes):
        # Loop through the trials
        for ii, trial in active_stims.iterrows():
            # Count the number of spikes per trial. 
            startInd = np.searchsorted(this_spiketimes[key], trial.start_time)
            endInd = np.searchsorted(this_spiketimes[key], trial.start_time + window)
            X[ii,jj]  = len(this_spiketimes[key][startInd:endInd])
    
    return X, cat

    


        
        