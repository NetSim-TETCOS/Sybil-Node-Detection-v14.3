import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load the training and test data
train_rssi_diff_df = pd.read_excel('C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\Merged_train_Data.xlsx', sheet_name=0)  # RSSI Difference
train_similarity_score_df = pd.read_excel('C:\\Users\\jace\\Documents\\Sybil_Attack\\Training-Data\\Merged_train_Data.xlsx', sheet_name=2)  # Similarity Score

test_rssi_diff_df = pd.read_excel('C:\\Users\\jace\\Documents\\Sybil_Attack\\Test-Data\\Merged_test_Data.xlsx', sheet_name=0)  # RSSI Difference
test_similarity_score_df = pd.read_excel('C:\\Users\\jace\\Documents\\Sybil_Attack\\Test-Data\\Merged_test_Data.xlsx', sheet_name=2)  # RSSI Similarity

# Merge the training data using both RSSI Difference and Similarity Score
train_df = pd.concat([train_rssi_diff_df[['Absolute Difference']], train_similarity_score_df[['RSSI Similarity', 'Similarity Score']]], axis=1).dropna()

# Merge the test data using both RSSI Difference and RSSI Similarity
test_df = pd.concat([test_rssi_diff_df[['Absolute Difference']], test_similarity_score_df[['RSSI Similarity']]], axis=1).dropna()

# Prepare the training data (RSSI Difference and RSSI Similarity as features)
X_train = train_df[['Absolute Difference', 'RSSI Similarity']]
y_train = (train_df['Similarity Score'] > 0).astype(int)  # Convert to binary labels for Sybil detection

# Prepare the test data
X_test = test_df[['Absolute Difference', 'RSSI Similarity']]

# Standardize the features (DecisionTreeClassifier doesn't require standardization, but it's done for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Decision Tree Classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train_scaled, y_train)

# Predict Sybil nodes for the test data using the trained Decision Tree model
test_predictions_binary = decision_tree_classifier.predict(X_test_scaled)

# Assign 0 for non-Sybil nodes and 1 for Sybil nodes
test_df['Predicted Sybil Node'] = test_predictions_binary  # 0 = non-Sybil, 1 = Sybil

# Keep only the necessary columns (Pair, RSSI Similarity, Predicted Sybil Node)
output_df = test_similarity_score_df[['Pair', 'RSSI Similarity']]
output_df['Predicted Sybil Node'] = test_df['Predicted Sybil Node']

# Save the results to a new Excel file
output_path = 'Decision_tree.xlsx'
output_df.to_excel(output_path, index=False)

print(f"Predicted Sybil node detection saved to {output_path}")
