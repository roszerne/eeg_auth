import mne
import numpy as np
import os 

T = 160  # sliding window length
delta = 4  # sliding step
Delta = 8 # sampling step
eta = 20 # number of overlapped segments for one sample
Gamma = (eta - 1) * delta + T # sampling window
orthogonalization = True #whether the singal should be orthogonalized
data_path = "./data_3/" # path where the ndarrays will be stored

def sliding_window(raw, T, delta, eta, channels, Delta, file_name):
    """
    Create sliding windows through the data, creating one training sample for one user

    Parameters:
    raw (RawEDF): RawEDF containing the signal data
    T (int): Length of one window (in samples).
    delta (int): Step size (in samples).
    eta (int): Maximum number of samples to consider.
    channels (int): Number of EEG channels.
    Delta (int): Sampling step.
    file_name (string): name of the original file with eeg recording.

    Returns:
    None
    """
    raw_selected = raw.pick_channels(channels)
    data = raw_selected.get_data()
    num_of_samples = (data.shape[1] - Gamma) // Delta 

    # Initialize an empty list to store the windows
    windows_array = np.zeros((num_of_samples, eta, T, len(channels)), dtype=np.float16)
    index = 0
    index2 = 0 
    # Create sampling windows
    for j in range(0, num_of_samples):   
        # Create sliding windows
        for i in range(0, eta):           
            windows_array[j, i, :, :] = data[:, index2 + index:index + index2 + T].T  # Extract one window
            index += delta
        index = 0
        index2 += Delta

    indices = np.arange(windows_array.shape[0])
    np.random.shuffle(indices)
    windows_array = windows_array[indices, :, :, :]

    np.save(data_path + file_name + '_fp16.npy', windows_array)


path = './files/' # path to folder containing the recordings
subjects = os.listdir(path)
folders = [item for item in subjects if os.path.isdir(os.path.join(path, item))] # get all folder names 

# go thorugh all patients
for subject in folders:
    path2 = path + subject + '/'
    trials = os.listdir(path2)
    files = [item for item in trials if os.path.isfile(os.path.join(path2, item)) and not item.endswith('.event')]
    files = files[:1] # get the first recording (REO session)
    for file in files:
        final_path = path2 + file
        raw = mne.io.read_raw_edf(final_path , preload=True)
        data = raw.get_data()
        # Select the channels
        #channels = ['O1..', 'Oz..', 'O2..', 'Po3.', 'Po4.',  'P3..', 'Pz..',  'P4..', 'P8..', 'Po7.', 'T7..', 'T8..', 'F7..', 'F3..',  'Fz..', 'F4..', 'F8..', 'Af3.', 'Af4.', 'Fp1.', 'Fp2.', 'Cp5.', 'Cp1.', 'Cp2.', 'Cp6.','C3..', 'Cz..', 'C4..', 'Fc5.', 'Fc1.', 'Fc2.', 'Fc6.']
        #channels = raw.ch_names # All channels
        channels = ['Oz..', 'T7..', 'Cz..']
        # perform signal normalization for each channel
        for channel in channels:
            i = raw.ch_names.index(channel)
            channel_data = data[i]  # Get the data for the current channel
            v_min = np.min(channel_data)  # Calculate the minimum value
            v_max = np.max(channel_data)  # Calculate the maximum value
            data[i] = (channel_data - v_min) / (v_max - v_min)  # Apply normalization formula
        raw._data = data
        # perform signal orthogonalization
        if orthogonalization:
            data = raw.get_data()      
            for index in range(1, len(channels)):
                channel = channels[index]
                i = raw.ch_names.index(channel)           
                new_channel_data = data[i] # Get the data for the current channel
                orth = [0] * len(new_channel_data) 
                for index2 in range(0, index):       
                    prev_channel = channels[index2]
                    j = raw.ch_names.index(prev_channel)
                    prev_channel_data = data[j] # Get the data for the previous channel
                    v_nom = np.dot(prev_channel_data, new_channel_data)  # Calculate the minimum value
                    v_denom = np.dot(prev_channel_data, prev_channel_data)  # Calculate the maximum value
                    orth += ((v_nom / v_denom) * prev_channel_data)
                data[i] = new_channel_data - orth
            raw._data = data

        sliding_window(raw, T, delta, eta, channels, Delta, file.split('.')[0]) # create .npy file containing all samples for a single user
