# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

def benign_ports(sz):
    rnd=np.random.rand(sz)
    return np.where(rnd<0.9,1,np.where(rnd<0.98,np.random.randint(2,6,sz),np.random.randint(6,11,sz)))

# Шаг 1: Генерация 990 ложных срабатываний (FP)
n_fp = 990
fp_data = {
    "distinct_ports_5s": benign_ports(n_fp),
    "events_5s": np.random.randint(2,30,n_fp),
    "proto_diversity": np.random.choice([1,2],n_fp,p=[0.95,0.05]),
    "udp_ratio": np.random.choice([0,1],n_fp,p=[0.9,0.1]),
    "icmp_ratio": np.random.choice([0,1],n_fp,p=[0.75,0.25]),
    "trusted_ip_src": np.random.choice([0,1],n_fp,p=[0.1,0.9]),
    "dst_host_amount": np.random.randint(1,4,n_fp),
    "label":0
}
fp_df = pd.DataFrame(fp_data)
fp_df["attempt_rate"]=fp_df["events_5s"]/5
fp_df["multiple_dst_host"]=0

# Шаг 2: Генерация 10 истинных срабатываний (TP)
n_tp = 10
tp_data = {
    "distinct_ports_5s": np.random.randint(2,10,n_tp),
    "events_5s": np.random.randint(15,100,n_tp),
    "proto_diversity": np.random.choice([2,3],n_tp,p=[0.6,0.4]),
    "udp_ratio": np.random.choice([0,1],n_tp,p=[0.1,0.9]),
    "icmp_ratio": np.random.choice([0,1],n_tp,p=[0.3,0.7]),
    "trusted_ip_src": np.random.choice([0,1],n_tp,p=[0.8,0.2]),  # mix trusted/untrusted
    "dst_host_amount": np.random.randint(2,6,n_tp),
    "label":1
}
tp_df = pd.DataFrame(tp_data)
tp_df["attempt_rate"]=tp_df["events_5s"]/5
tp_df["multiple_dst_host"]=np.random.choice([0,1],n_tp,p=[0.5,0.5])

tp_idx = tp_df.sample(frac=0.3, random_state=1).index
tp_df.loc[tp_idx, "distinct_ports_5s"] = (tp_df.loc[tp_idx, "distinct_ports_5s"] *
                                       np.random.uniform(0.7, 0.9, size=len(tp_idx))).astype(int).clip(lower=1)

# Шаг 3: Внесение шума
fp_idx = fp_df.sample(frac=0.2, random_state=2).index
fp_df.loc[fp_idx, "events_5s"] = (fp_df.loc[fp_idx, "events_5s"] * 1.5).astype(int).clip(lower=1)
fp_df.loc[fp_idx, "attempt_rate"] = fp_df.loc[fp_idx, "events_5s"] / 5

inv_fp = fp_df.sample(frac=0.1, random_state=3).index
inv_tp = tp_df.sample(frac=0.1, random_state=4).index
fp_df.loc[inv_fp, "trusted_ip_src"] ^= 1
tp_df.loc[inv_tp, "trusted_ip_src"] ^= 1

proto_fp = fp_df.sample(frac=0.15, random_state=5).index
proto_tp = tp_df.sample(frac=0.2, random_state=6).index
fp_df.loc[proto_fp, "proto_diversity"] = 2
tp_df.loc[proto_tp, "proto_diversity"] = 1

df = pd.concat([fp_df, tp_df], ignore_index=True)
flip_idx = df.sample(frac=0.01, random_state=7).index
df.loc[flip_idx, "label"] ^= 1 

df["attempt_rate"] = df["events_5s"] / 5

# Шаг 4: Обучение модели
X = df.drop(columns=["label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegressionCV(cv=5, penalty="l2", solver="lbfgs", max_iter=500, random_state=42)
lr_model.fit(X_train_scaled, y_train)

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

# Вывод результатов
print(f"Threshold: {threshold}")
print(f"Recall: {recall_noisy:.2f}")
print(f"Precision: {precision_noisy:.2f}")
print(f"False Positives: {false_positives_noisy}")
print("\nFeature Coefficients:")
print(coefficients_noisy)
print("\nLabel Distribution:")
print(df['label'].value_counts())

