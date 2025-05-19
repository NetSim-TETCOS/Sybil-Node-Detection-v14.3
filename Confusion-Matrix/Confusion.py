import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the Excel files
test_data_path = r"C:\Users\jace\Documents\Sybil_Attack\Confusion-Matrix\Test-Data.xlsx"
predicted_data_path = r"C:\Users\jace\Documents\Sybil_Attack\Classifiers\Decision-Tree.xlsx"

# Load the data
test_data = pd.read_excel(test_data_path)
predicted_data = pd.read_excel(predicted_data_path)

# Inspect the columns to check for the correct ones
print("Test Data Columns:", test_data.columns)
print("Predicted Data Columns:", predicted_data.columns)

# Replace these with the correct column names based on your inspection
y_true = test_data['Similarity Score']  # Replace with the actual column name for actual similarity score
y_pred = predicted_data['Predicted Sybil Node']  # Replace with the actual column name for predicted similarity score

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Save the confusion matrix as an image
plt.figure(figsize=(3, 3))  # Decreased the figure size

# Create a custom color map: green for TP/TN and blue for FP/FN
colors = ['#6BBF59', '#6BBF59', '#A3D5F4', '#A3D5F4']  # Green for TP/TN and blue for FP/FN
cmap = sns.color_palette(colors)

# Plot the confusion matrix with the custom color map
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, cbar=False, square=True, linewidths=2, linecolor='black')

# Add labels and titles for confusion matrix
plt.title('Confusion Matrix for Decision-Tree Classifier', fontsize=12, fontweight='bold')  # Reduced font size
plt.xlabel('True Class', fontsize=10, fontweight='bold')
plt.ylabel('Predicted Class', fontsize=10, fontweight='bold')
plt.xticks([0.5, 1.5], ['Positive', 'Negative'], fontsize=8, fontweight='bold')  # Reduced font size
plt.yticks([0.5, 1.5], ['Positive', 'Negative'], fontsize=8, fontweight='bold')  # Reduced font size

# Save the confusion matrix as a PNG file
confusion_matrix_image_path = r'C:\Users\jace\Documents\Sybil_Attack\Confusion-Matrix\confusion_matrix.png'
plt.tight_layout()
plt.savefig(confusion_matrix_image_path)
plt.show()

# Prepare the evaluation metrics table
table_data = [
    ['Accuracy', f'{accuracy:.2f}'],
    ['Precision', f'{precision:.2f}'],
    ['Recall', f'{recall:.2f}'],
    ['F1 Score', f'{f1:.2f}']
]

# Save the evaluation metrics table as an image
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=table_data, colWidths=[0.3, 0.3], cellLoc='center', loc='center')

# Adjust the table's font size
table.set_fontsize(14)
table.scale(1.5, 1.5)

# Save the evaluation metrics table as a PNG file
evaluation_metrics_image_path = r'C:\Users\jace\Documents\Sybil_Attack\Confusion-Matrix\evaluation_metrics.png'
plt.tight_layout()
plt.savefig(evaluation_metrics_image_path)
plt.show()

print(f"Confusion matrix saved to: {confusion_matrix_image_path}")
print(f"Evaluation metrics saved to: {evaluation_metrics_image_path}")
