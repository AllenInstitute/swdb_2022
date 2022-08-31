from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import  matplotlib.pyplot as plt

import allensdk
from allensdk.brain_observatory.\
    behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
import os
import numpy as np 

import platform
platstring = platform.platform()

data_dirname = 'visual-behavior-neuropixels'
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
    data_root = "/media/$USERNAME/Brain2022/"

# get the cache location
cache_dir = os.path.join(data_root, data_dirname)

#cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(
            cache_dir=cache_dir, use_static_cache=use_static)


# We are going to find the "familiar" session that contains the most V1 units. 
area = 'VISp'
# You have actually seen this code before, so we won't spend time on it...
units_table = cache.get_unit_table()
ecephys_sessions_table = cache.get_ecephys_session_table()

# For now, we are going to grab the one with the most V! units.
unit_by_session = units_table.join(ecephys_sessions_table,on = 'ecephys_session_id')
unit_in = unit_by_session[(unit_by_session['structure_acronym']==area) &\
                          (unit_by_session['experience_level']=='Familiar') &\
                          (unit_by_session['isi_violations']<.5)&\
                          (unit_by_session['amplitude_cutoff']<0.1)&\
                          (unit_by_session['presence_ratio']>0.95)]
unit_count = unit_in.groupby(["ecephys_session_id"]).count()
familiar_session_with_most_in_units = unit_count.index[np.argmax(unit_count['ecephys_probe_id'])]
# Actually imort the data
session = cache.get_ecephys_session(ecephys_session_id=familiar_session_with_most_in_units)


# Get unit information
session_units = session.get_units()
# Channel information
session_channels = session.get_channels()
# And accosiate each unit with the channel on which it was found with the largest amplitude
units_by_channels= session_units.join(session_channels,on = 'peak_channel_id')

# Filter for units in primary visual cortex
this_units = units_by_channels[(units_by_channels.structure_acronym == area)\
                               &(units_by_channels['isi_violations']<.5)\
                               &(units_by_channels['amplitude_cutoff']<0.1)\
                               &(units_by_channels['presence_ratio']>0.95)]
# Get the spiketimes from these units as a dictionary
this_spiketimes = dict(zip(this_units.index, [session.spike_times[ii] for ii in this_units.index]))

active_stims = session.stimulus_presentations[session.stimulus_presentations.stimulus_block==0 ]


def count_spikes_after_event(spikeTimes,eventTime,window):
    startInd = np.searchsorted(spikeTimes, eventTime)
    endInd = np.searchsorted(spikeTimes, eventTime+window)
    count = len(spikeTimes[startInd:endInd])
    return count

# Declare and empty variable X
X = np.zeros((len(active_stims),len(this_spiketimes)))
# This Loop is a little slow...be patient
# Loop Through both trials and units, counting the number of spikes
for jj,key in enumerate(this_spiketimes):
    # Loop through the trials
    for ii, trial in active_stims.iterrows():
        # Count the number of spikes per trial. 
        X[ii,jj] = count_spikes_after_event(this_spiketimes[key],trial.start_time,.25)

[unq,cat]= np.unique(active_stims.image_name,return_inverse=True)


[X_train,X_test,cat_train,cat_test] = model_selection.train_test_split(X,cat,test_size = .2,stratify=cat)
classifier = LDA()
classifier.fit(X_train,cat_train)
cat_hat = classifier.predict(X_test) 

C = confusion_matrix(cat_test,cat_hat,normalize='true')           
plt.figure(figsize=(8,8))
ax = plt.subplot(111)
cax = ax.imshow(C,interpolation='none',origin='lower',vmin=0,vmax=C.max())
ax.set_xlabel('Actual Class',fontsize=16)
ax.set_ylabel('Predicted Class',fontsize=16)
plt.colorbar(cax)