import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths to your files
predictions_csv = "/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/codes/capsule_vision_challenge_2024/submission/comb_mln2_test_dataset.xlsx"  
test_csv = "/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_eva02/test.csv"                # This contains: frame_path, class
output_dir = "/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_eva02/eva02_test_internal_split"  

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load CSVs
pred_df = pd.read_excel(predictions_csv)
test_df = pd.read_csv(test_csv)

print("[INFO] Loaded predictions shape:", pred_df.shape)
print("[INFO] Loaded test set shape:", test_df.shape)

# Extract filename from frame_path in test.csv
test_df["image_path"] = test_df["frame_path"].apply(os.path.basename)

# Normalize class names (lowercase)
test_df["true_class"] = test_df["class"].str.lower()
pred_df["predicted_class"] = pred_df["predicted_class"].str.lower()

# Merge predictions and ground truth
merged_df = pd.merge(pred_df, test_df[["image_path", "true_class"]], on="image_path", how="inner")

print("[INFO] Merged dataframe shape:", merged_df.shape)
print(merged_df[["image_path", "true_class", "predicted_class"]].head())

# Save merged results
merged_csv_path = os.path.join(output_dir, "merged_predictions_with_truth.csv")
merged_df.to_csv(merged_csv_path, index=False)
print(f"[SAVED] Merged predictions with ground truth to: {merged_csv_path}")

# Get ground truth and predictions
y_true = merged_df["true_class"]
y_pred = merged_df["predicted_class"]

# Compute confusion matrix
labels = sorted(list(set(y_true) | set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Save confusion matrix as image
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
conf_matrix_img_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_img_path)
print(f"[SAVED] Confusion matrix image to: {conf_matrix_img_path}")
plt.close()

# Save classification report
report = classification_report(y_true, y_pred, labels=labels)
report_txt_path = os.path.join(output_dir, "classification_report.txt")
with open(report_txt_path, "w") as f:
    f.write(report)
print(f"[SAVED] Classification report to: {report_txt_path}")

# Also print for debug
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)