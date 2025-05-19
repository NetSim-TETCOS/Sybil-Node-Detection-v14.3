import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the training and test data
train_rssi_diff_df = pd.read_excel('C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\Merged_train_Data.xlsx', sheet_name=0)  # Absolute Difference
train_similarity_score_df = pd.read_excel('C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\Merged_train_Data.xlsx', sheet_name=2)  # Similarity Score

test_rssi_diff_df = pd.read_excel('C:\\Users\\jace\\Documents\\Sybil_Attack\\Test-Data\\Merged_test_Data.xlsx', sheet_name=0)  # Absolute Difference
test_similarity_score_df = pd.read_excel('C:\\Users\\jace\\Documents\\Sybil_Attack\\Test-Data\\Merged_test_Data.xlsx', sheet_name=2)  # RSSI Similarity

# Merge the training data using both Absolute Difference and Similarity Score
train_df = pd.concat([train_rssi_diff_df[['Absolute Difference']], train_similarity_score_df[['RSSI Similarity', 'Similarity Score']]], axis=1).dropna()

# Merge the test data using both Absolute Difference and RSSI Similarity
test_df = pd.concat([test_rssi_diff_df[['Absolute Difference']], test_similarity_score_df[['RSSI Similarity']]], axis=1).dropna()

# Train the classifier using the Similarity Score to determine Sybil nodes
# Sybil nodes have a specific label in the Similarity Score (we will assume it's binary: 0 for non-Sybil, 1 for Sybil)
# Adjust this part based on your specific data if needed.
y_train = (train_df['Similarity Score'] > 0).astype(int)  # Consider values greater than 0 as Sybil nodes (1)

# Prepare the training data (Absolute Difference and RSSI Similarity as features)
X_train = train_df[['Absolute Difference', 'RSSI Similarity']]

# Prepare the test data
X_test = test_df[['Absolute Difference', 'RSSI Similarity']]

# Train a RandomForestClassifier model (binary classification)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict Sybil nodes for the test data
test_predictions_binary = classifier.predict(X_test)

# Assign 0 for non-Sybil nodes and 1 for Sybil nodes
test_df['Predicted Sybil Node'] = test_predictions_binary  # 0 = non-Sybil, 1 = Sybil

# Keep only the necessary columns (Pair, RSSI Similarity, Predicted Sybil Node)
output_df = test_similarity_score_df[['Pair', 'RSSI Similarity']]
output_df['Predicted Sybil Node'] = test_predictions_binary

# Save the results to a new Excel file
output_path = 'Random-Forest.xlsx'
output_df.to_excel(output_path, index=False)

print(f"Predicted Sybil node detection saved to {output_path}")
