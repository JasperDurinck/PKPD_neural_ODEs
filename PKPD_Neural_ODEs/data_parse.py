import utils
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

def calculate_cmax(data, label_col, time_column, dsfq_val=1, time_intervals=(7, 21)):
    
    """
    Calculates the maximum value (Cmax) for a specified label within a time threshold, based on a conditional value in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing the time, label, and DSFQ columns.
    label_col : str
        The column name for which to calculate the maximum value (Cmax).
    time_column : str
        The name of the column representing time values.
    dsfq_val : int, optional
        The specific value to check in the 'DSFQ' column. If all values in 'DSFQ' match this,
        the function uses the first time interval in `time_intervals`; otherwise, it uses the second.
        Default is 1.
    time_intervals : tuple of int, optional
        A tuple containing two time thresholds (e.g., (7, 21)). The first threshold is applied if all 
        values in 'DSFQ' match `dsfq_val`; otherwise, the second threshold is used. Default is (7, 21).
    
    Returns:
    --------
    cmax : float
        The maximum value found in the `label_col` within the specified time frame.
    cmax_time_full : np.ndarray
        A zero-padded or truncated 1D array of length 20, containing time and label values
        flattened together up to a maximum of 20 entries.
    
    Example:
    --------
    >>> cmax, cmax_time_full = calculate_cmax(data, label_col="Concentration", time_column="TIME")
    
    This function can be useful in pharmacokinetic or time-series analyses where specific intervals 
    need to be dynamically adjusted based on dataset criteria.
    """
    
    # Determine the time limit based on DSFQ values
    time_limit = time_intervals[0] if (data.DSFQ == dsfq_val).all() else time_intervals[1]
    
    # Select values within the time limit and calculate cmax
    cmax_time = data.loc[data[time_column] < time_limit, [time_column, label_col]].values.flatten()
    cmax = data.loc[data[time_column] < time_limit, label_col].max()
    
    # Pad or truncate time array to a fixed length of 20
    cmax_time_full = np.zeros((20,))
    cmax_time_full[:min(20, len(cmax_time))] = cmax_time[:20]
    
    return cmax, cmax_time_full

class PKPD_Dataset(Dataset):

    def __init__(self, data_to_load, label_cols, feature_cols, device, phase="train", time_column='TIME', devide_time=24):
        self.data = pd.read_csv(data_to_load)

        self.label_cols = label_cols
        self.features = feature_cols
        self.device = device
        self.phase = phase
        self.data[time_column] = self.data[time_column] / devide_time
        self.time_column = time_column

    def __len__(self):
        return self.data.PTNM.unique().shape[0] 

    def __getitem__(self, index):
        ptnm = self.data.PTNM.unique()[index]
        cur_data = self.data[self.data["PTNM"] == ptnm]
        times, features, labels, cmax_time = self.process(cur_data)
        return ptnm, times, features, labels, cmax_time

    def process(self, data):
        # Ensure that the dosing amount is always in column -2 of the features.
        if "AMT" in self.features:
            assert self.features[-1-len(self.label_cols)] == 'AMT', "Expected dosing amount (AMT) in column -2."
        else:
            raise ValueError("Dosing amount column (AMT) not found in features.")

        data = data.reset_index(drop=True)

        # Loop over all columns in the DataFrame
        for i, _ in enumerate(self.label_cols, start=1):
            # Calculate cmax and cmax_time_full for the specific column
            cmax, cmax_time_full = calculate_cmax(data=data, label_col=f"PK_timeCourse_{i}", time_column=self.time_column)
            
            # Normalize the column by dividing by cmax
            data[f"PK{i}_round1"] = data[f"PK{i}_round1"] / cmax

        features = data[self.features].values
        labels = data[self.label_cols].values
        times = data[self.time_column].values

        times = torch.from_numpy(times)
        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)#.unsqueeze_(-1)
        if self.label_cols.__len__() < 2: 
            labels = labels.unsqueeze_(-1)
        cmax_time_full = torch.from_numpy(cmax_time_full)
        return times, features, labels, cmax_time_full

def pkpd_collate_fn(batch, device):
    D = batch[0][2].shape[1]  # Number of features
    L = batch[0][3].shape[1]  # Number of labels

    # Gather unique time points across all examples in the batch
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]),
                                                sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    # Initialize tensors with dimensions matching multi-label requirements
    combined_features = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_label = torch.full([len(batch), len(combined_tt), L], float('nan')).to(device)
    combined_cmax_time = torch.zeros([len(batch), 20]).to(device)

    ptnms = []
    for b, (ptnm, tt, features, label, cmax_time) in enumerate(batch):
        ptnms.append(ptnm)
        tt = tt.to(device)
        features = features.to(device)
        label = label.to(device)
        cmax_time = cmax_time.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_features[b, indices] = features.float()
        if label.shape.__len__() == 3:
            label = label.squeeze(-1)
        combined_label[b, indices] = label.float()  # Assign labels with correct shape
        combined_cmax_time[b, :] = cmax_time.float()
    
    combined_tt = combined_tt.float()

    return ptnms, combined_tt, combined_features, combined_label, combined_cmax_time

def parse_data(device, phase="train", label_cols=["PK_timeCourse_1", "PK_timeCourse_2"], feature_cols = ['TFDS','TIME','CYCL','AMT',"PK1_round1", "PK2_round1"], covariates = [], dir='./'):

    train_data_path = f"{dir}train.csv"
    val_data_path = f"{dir}validate.csv"
    test_data_path = f"{dir}test.csv"

    """
    | Column name   | Description                                                   |
    | ------------- | -------------                                                 |
    | PTNM          | patient number                                                |
    | STUD          | study number                                                  |
    | DSFQ          | dosing frequency                                              |
    | CYCL          | dosing cycles                                                 |
    | AMT           | dosing amounts                                                |
    | TIME          | time in hours since the experiment begin for one individual   |
    | TFDS          | time in hours since the last dosing                           |
    | DV            | the observations of PK                                        |
    """

    """
    covariates = ['SEX','AGE','WT','RACR','RACE','BSA',
                  'BMI','ALBU','TPRO','WBC','CRCL','CRET',
                  'SGOT','SGPT','TBIL','TMBD','ALKP', 'HER', 
                  'ECOG','KEOALL','ASIAN']
    """

    feature_cols = covariates + feature_cols

    train = PKPD_Dataset(train_data_path, label_cols, feature_cols, device, phase="train")
    validate = PKPD_Dataset(val_data_path, label_cols, feature_cols, device, phase="validate")
    test = PKPD_Dataset(test_data_path, label_cols, feature_cols, device, phase="test")

    ptnm, times, features, labels, cmax_time = train[0]
    input_dim = features.size(-1)
    #n_labels = labels.size(-1) #TODO only one at this moment
 
    if phase == "train":
        train_dataloader = DataLoader(train, batch_size=1, shuffle=True, 
            collate_fn=lambda batch: pkpd_collate_fn(batch, device))
        val_dataloader = DataLoader(validate, batch_size=1, shuffle=False,
            collate_fn=lambda batch: pkpd_collate_fn(batch, device))

        dataset_objs = {
            "train_dataloader": utils.inf_generator(train_dataloader),
            "val_dataloader": utils.inf_generator(val_dataloader),
            "n_train_batches": len(train_dataloader),
            "n_val_batches": len(val_dataloader),
            "input_dim": input_dim
        }

    else:
        test_dataloader = DataLoader(test, batch_size=1, shuffle=False,
            collate_fn=lambda batch: pkpd_collate_fn(batch, device))

        dataset_objs = {
            "test_dataloader": utils.inf_generator(test_dataloader),
            "n_test_batches": len(test_dataloader),
            "input_dim": input_dim
        }

    return dataset_objs


if __name__ == "__main__":
    print("run")
    data = parse_data("cuda", dir='./data/')
    for ptnms, times, features, labels, cmax_time in data["train_dataloader"]:
        print(ptnms)
        print(times)
        print(features)
        print(labels)
        print(cmax_time)
        break

