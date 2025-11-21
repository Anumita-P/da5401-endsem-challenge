import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import lightgbm as lgb
import datetime
import warnings

warnings.filterwarnings("ignore")


AUGMENTATION_RATIO_NEGATIVE = 0.5  # 50% Synthetic Zeros (to fix low scores)
AUGMENTATION_RATIO_PERFECT = 0.5   # 50% Duplicate 10s (to fix missing 10s)
N_SPLITS = 5
PCA_COMPONENTS_GLOBAL = 30    # Components for raw vectors
PCA_COMPONENTS_DIFF = 15      #  Components for Difference Vectors
PCA_COMPONENTS_PROD = 15      #  Components for Product Vectors
np.random.seed(42)


try:
    M_raw_train = np.load("M_raw_train.npy"); M_raw_test = np.load("M_raw_test.npy")
    S_raw_train = np.load("S_raw_train.npy"); S_raw_test = np.load("S_raw_test.npy")
    U_raw_train = np.load("U_raw_train.npy"); U_raw_test = np.load("U_raw_test.npy")
    R_raw_train = np.load("R_raw_train.npy"); R_raw_test = np.load("R_raw_test.npy")
    train = pd.read_json("train_data.json", orient="records", lines=False)
    test = pd.read_json("test_data.json", orient="records", lines=False)
except FileNotFoundError:
    print("Missing files")
    exit()

y_train_raw = train["score"].values.astype(np.float32)
# Binning for stable CV
y_train_bins = pd.cut(y_train_raw, bins=[-1, 5.5, 8.5, 11], labels=[0, 1, 2]).astype(int)


print(" 2. Generating Augmented Data ")
n_train = len(y_train_raw)
high_indices = np.where(y_train_raw >= 8.0)[0]
perfect_indices = np.where(y_train_raw == 10.0)[0]


n_neg = int(n_train * AUGMENTATION_RATIO_NEGATIVE)
M_neg, S_neg, U_neg, R_neg, y_neg = [], [], [], [], []
for _ in range(n_neg):
    idx_m = np.random.randint(n_train)
    idx_r = np.random.choice(high_indices)
    while idx_m == idx_r: idx_r = np.random.choice(high_indices)
    M_neg.append(M_raw_train[idx_m]); S_neg.append(S_raw_train[idx_r])
    U_neg.append(U_raw_train[idx_r]); R_neg.append(R_raw_train[idx_r])
    y_neg.append(np.random.uniform(0.0, 0.5))

n_perf = int(n_train * AUGMENTATION_RATIO_PERFECT)
M_perf, S_perf, U_perf, R_perf, y_perf = [], [], [], [], []
if len(perfect_indices) > 0:
    
    sample_idxs = np.random.choice(perfect_indices, n_perf, replace=True)
    M_perf = M_raw_train[sample_idxs]
    S_perf = S_raw_train[sample_idxs]
    U_perf = U_raw_train[sample_idxs]
    R_perf = R_raw_train[sample_idxs]
    y_perf = y_train_raw[sample_idxs] + np.random.normal(0, 0.01, n_perf) # Tiny noise to prevent exact dups


arrays_to_stack = [M_raw_train, np.array(M_neg)]
if len(M_perf) > 0: arrays_to_stack.append(M_perf)
M_full = np.vstack(arrays_to_stack)

arrays_to_stack = [S_raw_train, np.array(S_neg)]
if len(S_perf) > 0: arrays_to_stack.append(S_perf)
S_full = np.vstack(arrays_to_stack)

arrays_to_stack = [U_raw_train, np.array(U_neg)]
if len(U_perf) > 0: arrays_to_stack.append(U_perf)
U_full = np.vstack(arrays_to_stack)

arrays_to_stack = [R_raw_train, np.array(R_neg)]
if len(R_perf) > 0: arrays_to_stack.append(R_perf)
R_full = np.vstack(arrays_to_stack)

arrays_to_stack = [y_train_raw, np.array(y_neg)]
if len(y_perf) > 0: arrays_to_stack.append(y_perf)
y_full = np.hstack(arrays_to_stack)

print(f"Original: {n_train}, Augmented Total: {len(y_full)}")





X_global_aug = np.hstack([M_full, S_full, U_full, R_full])
X_global_test = np.hstack([M_raw_test, S_raw_test, U_raw_test, R_raw_test])

# 1. Global PCA
pca_global = PCA(n_components=PCA_COMPONENTS_GLOBAL, random_state=42)
X_pca_aug = pca_global.fit_transform(X_global_aug)
X_pca_test = pca_global.transform(X_global_test)

# 2. Difference Vector PCA (The "Direction" of Error)
# calculate M - R for every sample and run PCA on that 768-dim vector
Diff_aug = M_full - R_full
Diff_test = M_raw_test - R_raw_test
pca_diff = PCA(n_components=PCA_COMPONENTS_DIFF, random_state=42)
X_diff_aug = pca_diff.fit_transform(Diff_aug)
X_diff_test = pca_diff.transform(Diff_test)

# 3. Product Vector PCA 
Prod_aug = M_full * R_full
Prod_test = M_raw_test * R_raw_test
pca_prod = PCA(n_components=PCA_COMPONENTS_PROD, random_state=42)
X_prod_aug = pca_prod.fit_transform(Prod_aug)
X_prod_test = pca_prod.transform(Prod_test)

# 4. Scalar Features (Distances)
def get_scalar_features(M, S, U, R):
    M_n, S_n, U_n, R_n = normalize(M), normalize(S), normalize(U), normalize(R)
    feats = []
    # Cosines
    pairs = [(M_n, S_n), (M_n, U_n), (M_n, R_n), (S_n, U_n), (S_n, R_n), (U_n, R_n)]
    feats.extend([(A * B).sum(axis=1, keepdims=True) for A, B in pairs])
    # L2 Distances
    feats.append(np.linalg.norm(M - R, axis=1, keepdims=True))
    feats.append(np.linalg.norm(M - S, axis=1, keepdims=True))
    # L1 Distances (Manhattan)
    feats.append(np.sum(np.abs(M - R), axis=1, keepdims=True))
    return np.hstack(feats)

X_scalar_aug = get_scalar_features(M_full, S_full, U_full, R_full)
X_scalar_test = get_scalar_features(M_raw_test, S_raw_test, U_raw_test, R_raw_test)

X_train_final = np.hstack([X_scalar_aug, X_pca_aug, X_diff_aug, X_prod_aug])
X_test_final = np.hstack([X_scalar_test, X_pca_test, X_diff_test, X_prod_test])

print(f"Total Features: {X_train_final.shape[1]}")





skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_final))

lgb_params = {
    'objective': 'regression', 'metric': 'rmse',
    'n_estimators': 6000, 'learning_rate': 0.01,
    'num_leaves': 40, 'feature_fraction': 0.6, 'bagging_fraction': 0.7, 'bagging_freq': 1,
    'lambda_l1': 0.2, 'lambda_l2': 0.2, 
    'n_jobs': -1, 'seed': 42, 'verbose': -1
}

for fold, (train_idx_orig, val_idx_orig) in enumerate(skf.split(M_raw_train, y_train_bins)):
    
    # Validation: Original Data
    X_val = X_train_final[val_idx_orig]
    y_val = y_full[val_idx_orig]
    
    # Training: Original + All Augmented
    aug_indices = np.arange(n_train, len(y_full))
    train_indices_full = np.concatenate([train_idx_orig, aug_indices])
    X_tr = X_train_final[train_indices_full]
    y_tr = y_full[train_indices_full]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(200, verbose=False)])
    
    oof_preds[val_idx_orig] = model.predict(X_val)
    test_preds += model.predict(X_test_final) / N_SPLITS


oof_final = np.clip(oof_preds, 0, 10)
test_final = np.clip(test_preds, 0, 10)
final_preds_int = np.round(test_final).astype(int)

rmse = np.sqrt(mean_squared_error(y_train_raw, oof_final))
print(f"\n✅ CV RMSE (Real Data): {rmse:.4f}")

low_mask = y_train_raw <= 5.0
rmse_low = np.sqrt(mean_squared_error(y_train_raw[low_mask], oof_final[low_mask]))
print(f"-> RAW RMSE for Low Scores (<= 5.0): {rmse_low:.4f}")

print("\nPredicted Test Distribution:")
print(pd.Series(final_preds_int).value_counts().sort_index())

now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({"ID": test["id"] if "id" in test.columns else np.arange(1, len(test)+1), "score": final_preds_int})
sub.to_csv(f"submission_LGBM_DiffPCA_Perfection_{now}.csv", index=False)