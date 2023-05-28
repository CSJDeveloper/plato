""" Extract the personalized accuracy from results """


import os
import glob
import pandas as pd


def collect_personalized_accuracy(
    root_folder,
    target_round,
    target_epoch,
    file_format_suffix="personalized_accuracy",
    accuracy_name="personalized_accuracy",
):
    """Collecting the personalized accuracy of clients in the root folder."""
    # Find all the CSV files with the name *_personalized_accuracy.csv in the client_* folders
    csv_files = glob.glob(f"{root_folder}/client_*/*_{file_format_suffix}.csv")

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame(
        columns=["client_id", "round", "epoch", f"{accuracy_name}"]
    )

    # Iterate through the CSV files, read their content, and append the relevant data to the combined DataFrame
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        dt_frame = pd.read_csv(csv_file)

        # Filter the DataFrame by the target round and epoch
        filtered_df = dt_frame.loc[
            (dt_frame["round"] == target_round) & (dt_frame["epoch"] == target_epoch)
        ]

        # If there is data for the target round and epoch, append it to the combined DataFrame
        if not filtered_df.empty:
            client_id = os.path.basename(os.path.dirname(csv_file))
            filtered_df.insert(0, "client_id", client_id)
            combined_data = pd.concat([filtered_df, combined_data], ignore_index=True)

    return combined_data
