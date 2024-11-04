
import pandas as pd
from sklearn.model_selection import train_test_split

def augment_time_range(
    train,
    args, 
    time_column="TIME", 
    ptnm_column="PTNM", 
    ):
    
    """
    Augments data by duplicating entries within specified time ranges and adding offsets to a unique identifier column.
    
    Parameters:
    -----------
    train : pd.DataFrame
        The original dataset containing time and identifier columns.
    time_column : str, optional
        The column name representing time values. Default is "TIME".
    ptnm_column : str, optional
        The column name representing unique identifiers. Default is "PTNM".
    ranges : list of int, optional
        A list of integers representing the time range multipliers to apply for augmentation. Default is [2, 3, 4].
    interval : int, optional
        The base time interval (e.g., one week) in the units of the `time_column`. Default is 21 * 24 (for 21 days in hours).
    offset_values : list of float, optional
        A list of offsets to add to the unique identifier column for each range. Should match the length of `ranges`.
        Default is [0.1, 0.2, 0.3].
    
    Returns:
    --------
    pd.DataFrame
        The augmented dataset with duplicated rows within the specified time ranges.
    
    Example:
    --------
    >>> augmented_data = augment_time_range(train, time_column="TIME", ptnm_column="PTNM",
                                            ranges=[2, 3, 4], interval=21 * 24, offset_values=[0.1, 0.2, 0.3])
    
    This function is useful for augmenting time-based data across different datasets with variable time ranges.
    """
    
    ranges = args.AUGMENTATION_SETTINGS['ranges']
    interval = args.AUGMENTATION_SETTINGS['interval']
    offset_values = args.AUGMENTATION_SETTINGS['offset_values']

    augment_data = pd.DataFrame(columns=train.columns)

    # Ensure that `ranges` and `offset_values` are the same length
    if len(ranges) != len(offset_values):
        raise ValueError("`ranges` and `offset_values` must be the same length.")
    
    for ptnm in train[ptnm_column].unique():
        for multiplier, offset in zip(ranges, offset_values):
            # Select data within the specified time range and modify identifier
            time_limit = multiplier * interval
            df = train[(train[ptnm_column] == ptnm) & (train[time_column] <= time_limit) & (train[time_column] >= 0)]
            df[ptnm_column] = df[ptnm_column] + offset
            augment_data = pd.concat([augment_data, df], ignore_index=True)
    
    return augment_data

def data_split(df, on_col, save_cols=None, seed=2020, test_size=0.2):
    if not save_cols:
        save_cols = df.columns.values
    
    target = df[on_col].unique()
    train, test = train_test_split(target, random_state=seed, test_size=test_size, shuffle=True)
    train_df = df[df[on_col].isin(train)]
    test_df = df[df[on_col].isin(test)]

    return train_df[save_cols], test_df[save_cols]

def split_dataset(file_path, args, test_size=0.2, seed=1329, output_dir="./"):
    """
    Processes a dataset by splitting it into training, validation, and test sets, 
    augmenting the training data, and saving the processed data to CSV files.

    Parameters:
    -----------
    file_path : str
        The file path to the CSV data to be processed.
    test_size : float, optional, default=0.2
        The proportion of the dataset to include in the test (and validation) split.
    seed : int, optional, default=1329
        The random seed for reproducibility when splitting the dataset.
    output_dir : str, optional, default="./"
        The directory where the output CSV files (train, validate, test) will be saved.

    Returns:
    --------
    train : pd.DataFrame
        The processed training data, including augmented records.
    validate : pd.DataFrame
        The processed validation data.
    test : pd.DataFrame
        The processed test data.

    Workflow:
    ---------
    1. Load the dataset from the provided file path.
    2. Split the dataset into training, validation, and test sets using the `data_split` function.
    3. Move specific records from the test set back into the training and validation sets based on custom conditions.
    4. Augment the training data by duplicating certain records and adjusting the 'PTNM' column.
    5. Save the processed data into CSV files for further use.
    6. Return the training, validation, and test sets as Pandas DataFrames.

    Notes:
    ------
    - The `data_split` function is expected to split the dataset based on the 'PTNM' column. 
      It should be defined elsewhere in your codebase.
    - Augmentation steps mimic the original logic by creating slight modifications in 'PTNM' values 
      and expanding the training dataset.
    """

    # Load the data
    data = pd.read_csv(file_path)

    # Split the data into train and test sets
    train, test = data_split(data, "PTNM", seed=seed, test_size=test_size)
    train, validate = data_split(train, "PTNM", seed=seed, test_size=test_size)

    # Create a dataframe for adding specific test data to train
    test_add_to_train = pd.DataFrame()

    # Add specific test data to train and validate based on conditions
    for dsfq, time_threshold in args.DSFQ_SETTINGS.items():
        test_add_to_train = pd.concat([test_add_to_train, test[(test.DSFQ == dsfq) & (test.TIME < time_threshold)]], ignore_index=True)
        test_add_to_train = pd.concat([test_add_to_train, test[(test.DSFQ == dsfq) & (test.TIME < time_threshold)]], ignore_index=True)
    train = pd.concat([train, test_add_to_train], ignore_index=True)
    validate = pd.concat([validate, test_add_to_train], ignore_index=True)

    augment_data = augment_time_range(train=train, args=args)

    # Combine augmented data with train data
    train = pd.concat([train, augment_data], ignore_index=True).reset_index(drop=True)

    # Save to CSV files
    train.to_csv(f"{output_dir}/train.csv", index=False)
    validate.to_csv(f"{output_dir}/validate.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)

    return train, validate, test
