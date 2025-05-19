import pandas as pd

# List of file paths for the 5 Excel files to merge
file_paths = [
    "C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\5-Vehicles-trained\\log\\RSSI-RSU-Features.xlsx",
    "C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\7-Vehicles\\log\\RSSI-RSU-Features.xlsx",
    "C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\9-Vehicles\\log\\RSSI-RSU-Features.xlsx",
"C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\11\\log\\RSSI-RSU-Features.xlsx",
"C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\12\\log\\RSSI-RSU-Features.xlsx",
"C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\13\\log\\RSSI-RSU-Features.xlsx"
]

# Initialize empty lists to hold the dataframes from each sheet
rssi_diff_list = []
avg_threshold_list = []
similarity_score_list = []

# Loop through each file path and read the corresponding sheets
for file_path in file_paths:
    excel_file = pd.ExcelFile(file_path)
    
    # Read the relevant sheets
    rssi_diff_list.append(pd.read_excel(excel_file, sheet_name='RSSI Difference'))
    avg_threshold_list.append(pd.read_excel(excel_file, sheet_name='Average Threshold'))
    similarity_score_list.append(pd.read_excel(excel_file, sheet_name='Similarity Score'))

# Concatenate the data from all files for each sheet
merged_rssi_diff = pd.concat(rssi_diff_list, ignore_index=True)
merged_avg_threshold = pd.concat(avg_threshold_list, ignore_index=True)
merged_similarity_score = pd.concat(similarity_score_list, ignore_index=True)

# Save the merged data into a single Excel file
output_path = 'C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\Merged_train_Data.xlsx'
with pd.ExcelWriter(output_path) as writer:
    merged_rssi_diff.to_excel(writer, sheet_name='RSSI Difference', index=False)
    merged_avg_threshold.to_excel(writer, sheet_name='Average Threshold', index=False)
    merged_similarity_score.to_excel(writer, sheet_name='Similarity Score', index=False)

print(f'Merged data saved to {output_path}')
