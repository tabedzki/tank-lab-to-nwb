"""Authors: Cody Baker."""
from scipy.io import loadmat
from dateutil.parser import parse as dateparse
from datetime import timedelta


def date_array_to_dt(array):
    temp = [str(round(x)) for x in array[0:-1]]
    date_text = temp[0] + "-" + temp[1] + "-" + temp[2] + "T" + temp[3] + ":" + temp[4] + ":" + str(array[-1])
    return dateparse(date_text)


# Epochs
session_path = "D:/Neuropixels/TowersTask/PoissonBlocksReboot_cohort1_VRTrain6_E75_T_20181105"
mat_file = session_path + ".mat"
matin = loadmat(mat_file)
session_start_time = date_array_to_dt(matin['log']['session'][0][0]['start'][0][0][0])
blocks = matin['log']['block']
epoch_start_fields = blocks[0][0]['start'][0]

n_epochs = len(epoch_start_fields)
epoch_start_dts = [date_array_to_dt(x[0]) for x in epoch_start_fields]
epoch_durationss = [timedelta(seconds=x[0][0]) for x in blocks[0][0]['duration'][0]]

epoch_windows = []
for j in range(n_epochs):
    start = epoch_start_dts[j] - session_start_time
    end = start + epoch_durationss[j]
    epoch_windows.append([start.total_seconds(), end.total_seconds()])

# Trials
trial_start_fields = []
trial_duration_fields = []
n_trials = []
for j in range(n_epochs):
    trial_start_fields.append(blocks[0][0]['trial'][0][j]['start'][0])
    trial_duration_fields.append(blocks[0][0]['trial'][0][j]['duration'][0])
    n_trials.append(len(trial_start_fields[j]))
trial_starts = [y[0][0]+epoch_windows[j][0] for j, x in enumerate(trial_start_fields) for y in x]
trial_ends = [y1[0][0]+epoch_windows[j][0]+y2[0][0]
              for j, x in enumerate(zip(trial_start_fields, trial_duration_fields))
              for y1, y2 in zip(x[0], x[1])]

trial_windows = []
for k in range(len(trial_starts)):
    trial_windows.append([trial_starts[k], trial_ends[k]])
