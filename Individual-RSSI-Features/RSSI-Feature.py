import pandas as pd
import itertools

# File paths (update as necessary)
input_file = "C:\\Users\\jace\\Documents\\Sybil_Attack\\Test-Data\\5-Vehicles\\log\\IEEE_802_11_Radio_Measurements_Log.csv"
filtered_output_file = 'RSSI-Distance.csv'
cleaned_filtered_output_file = 'Remove-Duplicate-Time-Windows.csv'
comparison_output_file = 'Average-Threshold.csv'
final_output_file = 'RSSI-RSU-Features.xlsx'

# Step 1: Filter and Save Initial Data based on Transmitter and Receiver
def extract_id(name):
    """Extracts the ID from a name string (e.g., VEHICLE_1 -> 1, RSU_4 -> 4)."""
    return name.split('_')[-1]

def filter_and_write(input_file, output_file):
    """Filters the data based on Transmitter and Receiver names and saves it to the output CSV file."""
    df = pd.read_csv(input_file)

    # Get unique Transmitter Names that start with 'VEHICLE'
    transmitters = df[df['Transmitter Name'].str.startswith('VEHICLE')]['Transmitter Name'].unique()
    
    # Get unique Receiver Names that start with 'RSU'
    receivers = df[df['Receiver Name'].str.startswith('RSU')]['Receiver Name'].unique()
    
    # Open the output file and write filtered data
    with open(output_file, 'w', newline='') as f:
        f.write("Time(ms),Vehicle_ID,Rx_Power(dBm),RSU_ID,Distance(m)\n")  # Write header
        for transmitter in transmitters:
            # Filter rows where Transmitter Name matches
            df_transmitter = df[df['Transmitter Name'] == transmitter]
            
            for receiver in receivers:
                # Filter rows where Receiver Name matches
                df_receiver = df_transmitter[df_transmitter['Receiver Name'] == receiver]
                
                # Reorder and select the required columns
                result = df_receiver[['Time(ms)', 'Rx_Power(dBm)', 'Distance(m)']].copy()
                
                # Extract IDs
                result['Vehicle_ID'] = extract_id(transmitter)
                result['RSU_ID'] = extract_id(receiver)
                
                # Write the rows to the file without extra spaces
                result[['Time(ms)', 'Vehicle_ID', 'Rx_Power(dBm)', 'RSU_ID', 'Distance(m)']].to_csv(f, index=False, header=False)

print("Step 1: Filtering and writing process started.")
filter_and_write(input_file, filtered_output_file)
print("Step 1: Filtering and writing process completed.")

# Step 2: Clean the Filtered Data and Save to New CSV
def clean_filtered_data(file_path, output_file_path):
    """Cleans the filtered data by removing duplicates based on unique RSU, Vehicle_ID, and Time Window combinations."""
    data = pd.read_csv(file_path)
    
    # Convert 'Time(ms)' into seconds for creating time windows
    data['Time(s)'] = data['Time(ms)'] / 1000
    
    # Define the time windows (e.g., 1-2 seconds, 2-3 seconds, etc.)
    data['Time Window'] = pd.cut(data['Time(s)'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], right=False)
    
    # Remove duplicates by keeping only the first entry for each unique RSU, Vehicle_ID, and Time Window combination
    cleaned_data = data.drop_duplicates(subset=['RSU_ID', 'Vehicle_ID', 'Time Window'])
    
    # Save the cleaned data to a new CSV file
    cleaned_data.to_csv(output_file_path, index=False)
    print(f"Step 2: Cleaned data saved to {output_file_path}")

clean_filtered_data(filtered_output_file, cleaned_filtered_output_file)

# Step 3: Calculate Pairwise Absolute Differences between Vehicles within each Time Window
def calculate_rssi_differences(input_file, output_file):
    """Calculates pairwise absolute differences between vehicle RSSI values within each time window and saves to output file."""
    data = pd.read_csv(input_file)
    
    # Convert Time from ms to seconds
    data['Time(s)'] = data['Time(ms)'] / 1000
    
    # Define the time windows
    time_windows = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    
    results = []
    grouped_data = data.groupby('RSU_ID')

    # Loop through each RSU and time window
    for rsu_id, rsu_data in grouped_data:
        for start, end in time_windows:
            # Filter the data for the current time window
            window_data = rsu_data[(rsu_data['Time(s)'] >= start) & (rsu_data['Time(s)'] < end)]
            vehicle_ids = window_data['Vehicle_ID'].values
            timestamps = window_data['Time(s)'].values
            rssi_values = window_data['Rx_Power(dBm)'].values
            distances = window_data['Distance(m)'].values

            # Calculate pairwise absolute differences between vehicles within the current time window
            for idx1, idx2 in itertools.combinations(range(len(vehicle_ids)), 2):
                difference = abs(rssi_values[idx1] - rssi_values[idx2])
                results.append({
                    'RSU_ID': rsu_id,
                    'Time Window': f"{start}-{end}s",
                    'Vehicle ID 1': vehicle_ids[idx1],
                    'Timestamp 1': timestamps[idx1],
                    'RSSI Value 1': rssi_values[idx1],
                    'Distance 1 (m)': distances[idx1],
                    'Vehicle ID 2': vehicle_ids[idx2],
                    'Timestamp 2': timestamps[idx2],
                    'RSSI Value 2': rssi_values[idx2],
                    'Distance 2 (m)': distances[idx2],
                    'Absolute Difference': difference
                })
                
    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Step 3: Output saved to {output_file}")

calculate_rssi_differences(cleaned_filtered_output_file, comparison_output_file)

# Step 4: Calculate Average Differences and Similarity Scores for Each Pair
def calculate_average_similarity(input_file, output_file):
    """Calculates average absolute differences and RSSI similarity scores and saves them to an Excel file."""
    data = pd.read_csv(input_file)
    threshold = 0.1  # Adjust the threshold if necessary

    # Find all unique vehicle IDs in the dataset
    vehicle_ids = pd.unique(data[['Vehicle ID 1', 'Vehicle ID 2']].values.ravel())
    pairs = list(itertools.combinations(vehicle_ids, 2))

    average_results = []
    grouped_data = data.groupby('RSU_ID')

    for rsu_id, rsu_data in grouped_data:
        total_pairs = len(pairs)
        for vehicle1, vehicle2 in pairs:
            # Filter rows where the vehicle IDs match the current pair
            pair_data = rsu_data[((rsu_data['Vehicle ID 1'] == vehicle1) & (rsu_data['Vehicle ID 2'] == vehicle2)) |
                                 ((rsu_data['Vehicle ID 1'] == vehicle2) & (rsu_data['Vehicle ID 2'] == vehicle1))]
            
            # Calculate the average absolute difference for the current pair
            if not pair_data.empty:
                avg_diff = pair_data['Absolute Difference'].mean()
                label = 0 if avg_diff > threshold else 1  # Assign a label based on the threshold
                label_proportion = label / total_pairs

                average_results.append({
                    'RSU_ID': rsu_id,
                    'Pair': f"{vehicle1}-{vehicle2}",
                    'Average Absolute Difference': avg_diff,
                    'Threshold Condition': label,
                    'Threshold Proportion': label_proportion
                })

    average_df = pd.DataFrame(average_results)

    # Create a third sheet by calculating for each unique pair
    third_sheet_data = average_df.groupby('Pair').agg(
        Sum_Threshold_Proportion=('Threshold Proportion', 'sum'),
        Entry_Count=('Threshold Proportion', 'count')
    ).reset_index()
    third_sheet_data['RSSI Similarity'] = third_sheet_data['Sum_Threshold_Proportion'] / third_sheet_data['Entry_Count']

    # Save to the same Excel file with different sheet names
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Write the original data to a sheet called "RSSI Difference"
        data.to_excel(writer, sheet_name='RSSI Difference', index=False)
        
        # Write the calculated averages with updated column names to a sheet called "Average Threshold"
        average_df.to_excel(writer, sheet_name='Average Threshold', index=False)
        
        # Write the summarized data with RSSI similarity scores to a sheet called "Similarity Score"
        third_sheet_data[['Pair', 'RSSI Similarity']].to_excel(writer, sheet_name='Similarity Score', index=False)

    print(f"Step 4: Results saved to {output_file}")

calculate_average_similarity(comparison_output_file, final_output_file)
