

import pandas as pd

def process_data(args):

    data_complete = pd.read_csv(args.GET_DATA, na_values='.')

    #set max cycles
    data_complete = data_complete[data_complete.CYCL < args.MAX_CYCLES]

    #cols in the df to keep
    data_complete = data_complete[args.COLUMNS_TO_SELECT]

    data_complete = data_complete.rename(columns=args.COLUMNS_TO_RENAME)

    data_complete["PTNM"] = data_complete["PTNM"].astype("int").map(lambda x: f"{x:0{args.PTNM_MAX_CHAR}d}")
    data_complete["ID"] = data_complete["STUD"].astype("int").astype("str") + data_complete["PTNM"]

    time_summary = data_complete[["ID", "TIME"]].groupby("ID").max().reset_index()
    selected_ptnms = time_summary[time_summary.TIME > args.MINIMAL_START_TIME].ID
    data_complete = data_complete[data_complete.ID.isin(selected_ptnms)]

    data_complete["AMT"] = data_complete["AMT"].fillna(0) #NAN values replaced with 0

    # Loop over each column that starts with 'PK_timeCourse'
    for col in data_complete.columns:
        if col.startswith("PK_timeCourse"):
            # Extract the unique identifier for this PK column
            i = col.split('_')[-1]  

            # Create a new column name for 'PK{i}_round1'
            round_col = f"PK{i}_round1"

            # Initialize the new column with the values of the current 'PK_timeCourse' column
            data_complete[round_col] = data_complete[col]

            # Apply the threshold condition and set values to 0 where necessary
            for dsfq, time_threshold in args.DSFQ_SETTINGS.items():
                data_complete.loc[(data_complete.DSFQ == dsfq) & (data_complete.TIME >= time_threshold), round_col] = 0

            # Fill any NaNs in the new 'PK{i}_round1' column
            data_complete[round_col] = data_complete[round_col].fillna(0)

            # Fill any NaNs in the original 'PK_timeCourse' column
            data_complete[col] = data_complete[col].fillna(-1)

    #this would remove non meaning full rows, like if there is first row with no dose yet and no inital amount present
    data_complete = data_complete[~((data_complete.AMT == 0) & (data_complete.TIME == 0))] 

    #this handle so duplicate times per person
    data_complete.loc[data_complete[["PTNM", "TIME"]].duplicated(keep="last"), "AMT"] = \
        data_complete.loc[data_complete[["PTNM", "TIME"]].duplicated(keep="first"), "AMT"].values
    data_complete = data_complete[~data_complete[["PTNM", "TIME"]].duplicated(keep="first")]

    data_complete.to_csv(args.STORE_DATA, index=False)
