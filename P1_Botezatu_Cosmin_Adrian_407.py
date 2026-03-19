import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft2, fftshift
from scipy.stats import wasserstein_distance

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import StratifiedKFold


DATA_DIR = "Archive"
IMG_DIR = os.path.join(DATA_DIR, "samples")

HIST_BINS = 64
FREQ_RATIO = 8
PATCH_GRID = 4  # 4x4 patches

XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 1
}

RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": 1,
    "n_jobs": -1
}


def load_image(image_id):
    path = os.path.join(IMG_DIR, f"{image_id}.npy")
    # Images are loaded as float32 arrays without per-image normalization
    img = np.load(path).astype(np.float32)
    return img


def patch_statistics(img):
    # Divide the image into a fixed grid, to summarize local behavior
    h, w = img.shape
    ph, pw = h // PATCH_GRID, w // PATCH_GRID

    patch_means = []
    patch_vars = []

    for i in range(PATCH_GRID):
        for j in range(PATCH_GRID):
            patch = img[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            patch_means.append(patch.mean())
            patch_vars.append(patch.var())

    patch_means = np.array(patch_means)
    patch_vars = np.array(patch_vars)

    # Aggregation of local patches into global descriptors
    return [
        patch_means.mean(),
        patch_means.var(),
        patch_vars.mean(),
        patch_vars.var()
    ]


def extract_features(img):
    features = []
    flat = img.flatten()

    # Global pixel features
    features += [
        flat.mean(),
        flat.std(),
        flat.var(),
        flat.min(),
        flat.max(),
        skew(flat),
        kurtosis(flat)
    ]

    # Histogram features
    hist, _ = np.histogram(flat, bins=HIST_BINS, density=True)
    features += [
        entropy(hist + 1e-8),
        hist.var()
    ]

    # FFT features
    fft_img = fftshift(np.abs(fft2(img)))
    features += [
        fft_img.mean(),
        fft_img.var()
    ]

    # Separation of low-freq and high-freq energy
    h, w = fft_img.shape
    ch, cw = h // 2, w // 2
    off = h // FREQ_RATIO

    low_freq = fft_img[ch-off:ch+off, cw-off:cw+off]
    high_freq = fft_img.copy()
    high_freq[ch-off:ch+off, cw-off:cw+off] = 0

    features += [
        low_freq.mean(),
        high_freq.mean()
    ]

    # Patch-based features
    features += patch_statistics(img)

    return np.array(features, dtype=np.float32)


def build_feature_cache(df_list):
    # Store per-image features
    ids = set()
    for df in df_list:
        ids |= set(df["id_noise_1"]) | set(df["id_noise_2"])

    cache = {}
    for img_id in ids:
        img = load_image(img_id)
        cache[img_id] = extract_features(img)

    return cache


def create_dataset(df, feature_cache):
    # Construct pairs of feature vectors for each image pair
    X, y = [], []

    for _, row in df.iterrows():
        f1 = feature_cache[row["id_noise_1"]]
        f2 = feature_cache[row["id_noise_2"]]

        # Feature-wise differences
        abs_diff = np.abs(f1 - f2)
        sq_diff = (f1 - f2) ** 2

        # Similarity and distance in feature space
        cosine_sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
        euclid = np.linalg.norm(f1 - f2)                
        
        img1 = load_image(row["id_noise_1"])
        img2 = load_image(row["id_noise_2"])

        # Pixel-level correlation
        flat1 = img1.flatten()
        flat2 = img2.flatten()
        pixel_corr = np.corrcoef(flat1, flat2)[0, 1]


        # Wasserstein distance 
        hist_dist = wasserstein_distance(f1, f2)        
        pair_feat = np.concatenate([
            abs_diff,
            sq_diff,
            [cosine_sim, euclid, hist_dist, pixel_corr]
        ])

        X.append(pair_feat)

        if "label" in df.columns:
            y.append(row["label"])

    return np.array(X), np.array(y) if len(y) else None


train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

feature_cache = build_feature_cache([train_df, val_df, test_df])

# Create pairwise datasets
X_train_raw, y_train = create_dataset(train_df, feature_cache)
X_val_raw, y_val = create_dataset(val_df, feature_cache)
X_test_raw, _ = create_dataset(test_df, feature_cache)

# Standardization of features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)


xgb = XGBClassifier(**XGB_PARAMS)
xgb.fit(X_train, y_train)

val_probs_xgb = xgb.predict_proba(X_val)[:, 1]
val_preds_xgb = (val_probs_xgb >= 0.5).astype(int)


xgb_results = {
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_val, val_preds_xgb),
    "MAE": mean_absolute_error(y_val, val_probs_xgb),
    "MSE": mean_squared_error(y_val, val_probs_xgb),
    "Spearman": spearmanr(y_val, val_probs_xgb).correlation,
    "Kendall": kendalltau(y_val, val_probs_xgb).correlation
}

pd.DataFrame([xgb_results])


rf = RandomForestClassifier(**RF_PARAMS)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_val)[:, 1]
rf_preds = (rf_probs >= 0.5).astype(int)


rf_results = {
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y_val, rf_preds),
    "MAE": mean_absolute_error(y_val, rf_probs),
    "MSE": mean_squared_error(y_val, rf_probs),
    "Spearman": spearmanr(y_val, rf_probs).correlation,
    "Kendall": kendalltau(y_val, rf_probs).correlation
}

pd.DataFrame([rf_results])


pd.DataFrame([xgb_results, rf_results])


# RETRAINING OF VALIDATION+TRAIN XGBOOST 
full_df = pd.concat([train_df, val_df], ignore_index=True) 
X_full_raw, y_full = create_dataset(full_df, feature_cache) 
X_full = scaler.fit_transform(X_full_raw) 

final_xgb = XGBClassifier(**XGB_PARAMS) 
final_xgb.fit(X_full, y_full) 

test_preds = final_xgb.predict(X_test) 

submission = pd.DataFrame({ 
    "id_pair": test_df.apply( 
        lambda r: f"({r['id_noise_1']},{r['id_noise_2']})", axis=1 
    ), 
    "label": test_preds 
})
 
submission.to_csv("submission_xgboost_final.csv", index=False)

# NO RETRAINING XGBOOST 
test_preds = xgb.predict(X_test) 
submission = pd.DataFrame({ 
    "id_pair": test_df.apply( 
        lambda r: f"({r['id_noise_1']},{r['id_noise_2']})", axis=1 
    ), 
    "label": test_preds 
})

submission.to_csv("submission_xgboost_simple.csv", index=False)

# RETRAINING OF VALIDATION+TRAIN RANDOM FOREST 
full_df = pd.concat([train_df, val_df], ignore_index=True) 
X_full_raw, y_full = create_dataset(full_df, feature_cache) 
X_full = scaler.fit_transform(X_full_raw)

final_rf = RandomForestClassifier(**RF_PARAMS) 
final_rf.fit(X_full, y_full) 

test_preds_rf = final_rf.predict(X_test) 

submission_rf = pd.DataFrame({ 
    "id_pair": test_df.apply( 
        lambda r: f"({r['id_noise_1']},{r['id_noise_2']})", axis=1 
    ), 
    "label": test_preds_rf 
    }) 

submission_rf.to_csv("submission_rf_final.csv", index=False)

# NO RETRAINING RANDOM FOREST
test_preds_rf = rf.predict(X_test) 
submission_rf = pd.DataFrame({ 
    "id_pair": test_df.apply( 
        lambda r: f"({r['id_noise_1']},{r['id_noise_2']})", axis=1 
    ), 
    "label": test_preds_rf 
    }) 

submission_rf.to_csv("submission_rf_simple.csv", index=False)

# CHECKING ACCURACY FOR RANDOM FOREST USING 5-FOLD CROSS-VALIDATION
# Prepare the full labeled dataset (train + validation)
full_df = pd.concat([train_df, val_df], ignore_index=True)

# Rebuild pairwise features for the full dataset
X_full_raw, y_full = create_dataset(full_df, feature_cache)
X_full = scaler.fit_transform(X_full_raw)

# Configure stratified 5-fold cross-validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_metrics = []

print(f"Starting 5-fold cross-validation on {len(X_full)} samples")

# Cross-validation loop
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), start=1):

    # Split data for this fold
    X_f_train, X_f_val = X_full[train_idx], X_full[val_idx]
    y_f_train, y_f_val = y_full[train_idx], y_full[val_idx]

    # Initialize a fresh Random Forest for each fold
    model = RandomForestClassifier(**RF_PARAMS)

    # Train on fold-specific training data
    model.fit(X_f_train, y_f_train)

    # Predict probabilities on fold validation data
    probs = model.predict_proba(X_f_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Store evaluation metrics
    fold_metrics.append({
        "Fold": fold_idx,
        "Accuracy": accuracy_score(y_f_val, preds),
        "MSE": mean_squared_error(y_f_val, probs),
        "Spearman": spearmanr(y_f_val, probs).correlation
    })

    print(f"  Fold {fold_idx}: Accuracy = {fold_metrics[-1]['Accuracy']:.4f}")

# Aggregate results across folds
metrics_df = pd.DataFrame(fold_metrics)

print("\n Cross-Validation Results")
print(metrics_df)

print("\n Average Cross-Validation Metrics")
print(metrics_df.mean(numeric_only=True))

