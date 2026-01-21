# preprocess.py


import torch
import numpy as np
import scipy.io as scio
import os

def get_data_path(file_path):
    data_path = []
    for f in os.listdir(file_path):
        if f.startswith("."):
            continue
        # Only keep .mat files except label.mat
        elif f.endswith('.mat') and f != 'label.mat':
            data_path.append(os.path.join(file_path, f))
    return data_path

# Time-domain sliding window of the DE feature in an experiment
def window_slice(data, time_steps):
    #print(f"  [DEBUG] Input data shape: {data.shape}, time_steps: {time_steps}")
    original_shape = data.shape
    data = np.transpose(data, (1, 0, 2)).reshape(-1, 310)
    #print(f"  [DEBUG] After reshape: {data.shape}")
    
    xs = []
    available_windows = data.shape[0] - time_steps + 1
    #print(f"  [DEBUG] Available windows: {available_windows}")

    if available_windows <= 0:
        #print(f"  [WARNING] Not enough data for windowing. Need at least {time_steps} timesteps, but got {data.shape[0]}")
        return np.array([]).reshape(0, time_steps, 310)
    
    for i in range(available_windows):
        xs.append(data[i: i + time_steps])
    
    xs = np.concatenate(xs).reshape((len(xs), -1, 310))
    #print(f"  [DEBUG] Final output shape: {xs.shape}")
    return xs


def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type}
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variables
    label_seed3 = [[2,1,0,0,1,2,0,1,2,2,1,0,1,2,0],
                    [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0],
                    [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0]]
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    else:
        raise ValueError('Unsupported dataset name; seed3 only')

#The return is a list of x and y, where the list contains data for each subject
def load_trained_data(samples_path_list, args):
    print(f"[DEBUG] Processing {len(samples_path_list)} files")
    # load the label data
    _, _, labels = get_number_of_label_n_trial(args.dataset_name)
    label = labels[int(args.session)-1]
    if args.dataset_name=="seed3":
        label = np.resize(label, (15,))
        label = np.reshape(label, (1, 15))
    
    X_train_all = []
    Y_tain_all = []
    #Iterate through each subject (there are 14 source subjects in both datasets)
    for path in samples_path_list:
        #print(f"\n[DEBUG] Processing file: {path}")
        try:
            # load the sample data
            sample = scio.loadmat(path, verify_compressed_data_integrity=False)
            #print(f"[DEBUG] Keys in file: {[k for k in sample.keys() if not k.startswith('__')]}")
            
            flag = 0
            X_train = []
            y_train = []
            
            de_keys = [k for k in sample.keys() if k.startswith("de_LDS")]
            #print(f"[DEBUG] Found {len(de_keys)} de_LDS keys: {de_keys}")
            
            for key, val in sample.items():
                if key.startswith("de_LDS"):
                    #print(f"[DEBUG] Processing key: {key}, shape: {val.shape}")
                    windowed_data = window_slice(val, args.time_steps)
                    
                    if windowed_data.shape[0] > 0:  # Add only non-empty windows
                        X_train.append(windowed_data)
                        train_label = np.full((windowed_data.shape[0], 1), label[0, flag])
                        y_train.append(train_label)
                        #print(f"[DEBUG] Added data with shape {windowed_data.shape}")
                    else:
                        print(f"[WARNING] Skipping empty windowed data for key {key}")
                    
                    flag += 1
            
            #print(f"[DEBUG] X_train length: {len(X_train)}")
            
            if len(X_train) == 0:
                #print(f"[ERROR] No valid data found in {path}")
                continue
                
            X_train_one_subject = np.concatenate(X_train)
            y_train_one_subject = np.concatenate(y_train)
            #print(f"[DEBUG] Final subject data shape: {X_train_one_subject.shape}")
            
            X_train_all.append(X_train_one_subject)
            Y_tain_all.append(y_train_one_subject)
            
        except Exception as e:
            #print(f"[ERROR] Failed to process {path}: {str(e)}")
            continue
    
    #print(f"[DEBUG] Successfully processed {len(X_train_all)} subjects")
    return X_train_all, Y_tain_all

def normalize(features, select_dim=0):
    features_min, _ = torch.min(features, dim=select_dim)
    features_max, _ = torch.max(features, dim=select_dim)
    features_min = features_min.unsqueeze(select_dim)
    features_max = features_max.unsqueeze(select_dim)
    return (features - features_min)/(features_max - features_min)

# Load the data, return a list of samples and labels, including tensors of data for each subject
def load4train(samples_path_list, args):
    """
    load the SEED data set
    """
    train_sample, train_label = load_trained_data(samples_path_list, args)
    
    if len(train_sample) == 0:
        print("[ERROR] No data was loaded from any file!")
        return [], []
    
    sample_res = []
    label_res = []
    for subject_index in range(len(train_sample)):
        # transfer from ndarray to tensor
        one_subject_samples = torch.from_numpy(train_sample[subject_index]).type(torch.FloatTensor)
        one_subject_labels = torch.from_numpy(train_label[subject_index]).type(torch.LongTensor)
        # normalize tensor
        one_subject_samples = normalize(one_subject_samples)
        sample_res.append(one_subject_samples)
        label_res.append(one_subject_labels)
    return sample_res, label_res

#The input is the full set of data from the first session.
def getDataLoaders(one_subject, args):
    pre_path = args.path
    config_path = {"file_path": pre_path + args.session + "/",
                   "label_path": pre_path + "label.mat"}
    path_list = get_data_path(config_path["file_path"])
    #print(f"[DEBUG] All data files found: {path_list}")
    
    try:
        target_path_list = [i for i in path_list if(i.startswith(config_path["file_path"] + str(int(one_subject+1))+"_"))]
        target_path = target_path_list[0]
        #print(f"[DEBUG] Target file: {target_path}")
    except:
        #print("[ERROR] target data not exist")
        return None, None
        
    path_list.remove(target_path)
    source_path_list = path_list
    #print(f"[DEBUG] Source files: {len(source_path_list)} files")

    # read from DE feature
    sources_sample, sources_label = load4train(source_path_list, args)
    targets_sample, targets_label = load4train(target_path_list, args)

    if len(targets_label) == 1:
        target_sample = targets_sample[0]
        target_label = targets_label[0]

    # Generate Data loaders
    source_dsets = []
    for i in range(len(sources_sample)):
        source_dsets.append(torch.utils.data.TensorDataset(sources_sample[i], sources_label[i]))
    target_dset = torch.utils.data.TensorDataset(target_sample, target_label)

    source_loaders = []
    for j in range(len(source_dsets)):
        source_loaders.append(torch.utils.data.DataLoader(source_dsets[j], args.batch_size, shuffle=True, num_workers=args.num_workers_train, drop_last=True))
    test_loader = torch.utils.data.DataLoader(target_dset, args.batch_size, shuffle=False, num_workers=args.num_workers_test, drop_last=True)
    return source_loaders, test_loader