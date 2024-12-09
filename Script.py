# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate 990 False Positives (FPs)
n_fp = 990
fp_data = {
    "count_subevents": np.random.randint(11, 30, size=n_fp),
    "is_internal_ip": np.random.choice([0, 1], size=n_fp, p=[0.5, 0.5]),
    "is_root": np.random.choice([0, 1], size=n_fp, p=[0.2, 0.8]),
    "parent_process_duration": np.random.uniform(1, 200, size=n_fp),
    "parent_process_category_user": np.random.choice([0, 1], size=n_fp, p=[0.6, 0.4]),
    "is_attack_tool_detected": 0,
    "success": 0,
    "ip_prior_incident": 0,
    "is_system_account": 1,
    "trusted_ip_src": 1,
    "trusted_ip_dst": 1,
}
fp_df = pd.DataFrame(fp_data)
fp_df["attack_intensity"] = fp_df["count_subevents"] * fp_df["parent_process_duration"]
fp_df["working_hours"] = 1
fp_df["label"] = 0

# Add noise to FPs
noise_indices_fp = np.random.choice(fp_df.index, size=int(0.2 * n_fp), replace=False)  # 20% noisy rows
fp_df.loc[noise_indices_fp, "is_attack_tool_detected"] = np.random.choice([0, 1], size=len(noise_indices_fp))
fp_df.loc[noise_indices_fp, "is_root"] = np.random.choice([0, 1], size=len(noise_indices_fp))
fp_df.loc[noise_indices_fp, "parent_process_duration"] *= np.random.uniform(0.5, 1.5, size=len(noise_indices_fp))

# Step 2: Generate 10 True Positives (TPs)
n_tp = 10
tp_data = {
    "count_subevents": np.random.randint(20, 50, size=n_tp),
    "is_internal_ip": np.random.choice([0, 1], size=n_tp, p=[0.6, 0.4]),
    "is_root": np.random.choice([0, 1], size=n_tp, p=[0.5, 0.5]),
    "parent_process_duration": np.random.uniform(500, 1000, size=n_tp),
    "parent_process_category_user": np.random.choice([0, 1], size=n_tp, p=[0.4, 0.6]),
    "is_attack_tool_detected": np.random.choice([0, 1], size=n_tp, p=[0.3, 0.7]),
    "success": np.random.choice([0, 1], size=n_tp, p=[0.5, 0.5]),
    "ip_prior_incident": np.random.choice([0, 1], size=n_tp, p=[0.1, 0.9]),
    "is_system_account": np.random.choice([0, 1], size=n_tp, p=[0.5, 0.5]),
    "trusted_ip_src": np.random.choice([0, 1], size=n_tp, p=[0.2, 0.8]),
    "trusted_ip_dst": np.random.choice([0, 1], size=n_tp, p=[0.2, 0.8]),
}
tp_df = pd.DataFrame(tp_data)
tp_df["attack_intensity"] = tp_df["count_subevents"] * tp_df["parent_process_duration"]
tp_df["working_hours"] = np.random.choice([0, 1], size=n_tp, p=[0.3, 0.7])
tp_df["label"] = 1

# Add noise to TPs
noise_indices_tp = np.random.choice(tp_df.index, size=int(0.3 * n_tp), replace=False)  # 30% noisy rows
tp_df.loc[noise_indices_tp, "trusted_ip_src"] = 1  # Flip to trusted IP
tp_df.loc[noise_indices_tp, "is_attack_tool_detected"] = 0  # Remove attack tool detection
tp_df.loc[noise_indices_tp, "parent_process_duration"] *= np.random.uniform(0.7, 1.3, size=len(noise_indices_tp))

# Step 3: Combine FPs and TPs
df_noisy = pd.concat([fp_df, tp_df], ignore_index=True)

# Step 4: Split the Dataset
X = df_noisy.drop(columns=["label"])
y = df_noisy["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression with Regularization
lr_model = LogisticRegressionCV(cv=5, penalty="l2", solver="lbfgs", max_iter=500, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Step 7: Evaluate the Model with Adjustable Threshold
threshold = 0
y_probs_noisy = lr_model.predict_proba(X_test_scaled)[:, 1]
y_preds_noisy = (y_probs_noisy >= threshold).astype(int)

recall_noisy = recall_score(y_test, y_preds_noisy)
precision_noisy = precision_score(y_test, y_preds_noisy, zero_division=0)
false_positives_noisy = sum((y_preds_noisy == 1) & (y_test == 0))

coefficients_noisy = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr_model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

# Output Results
print(f"Threshold: {threshold}")
print(f"Recall: {recall_noisy:.2f}")
print(f"Precision: {precision_noisy:.2f}")
print(f"False Positives: {false_positives_noisy}")
print("\nFeature Coefficients:")
print(coefficients_noisy)
print("\nLabel Distribution:")
print(df_noisy['label'].value_counts())