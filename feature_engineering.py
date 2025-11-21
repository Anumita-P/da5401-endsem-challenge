import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
import datetime
from tqdm import tqdm

METRIC_NAMES = "metric_names.json"
METRIC_EMB = "metric_name_embeddings.npy"
TRAIN_JSON = "train_data.json"
TEST_JSON = "test_data.json"

# Load metric names and embeddings
with open(METRIC_NAMES, "r", encoding="utf-8") as f:
    metric_names = json.load(f)
metric_embs = np.load(METRIC_EMB)
name_to_emb = {name: metric_embs[i] for i, name in enumerate(metric_names)}

# Load train/test
train = pd.read_json(TRAIN_JSON, orient="records", lines=False)
test = pd.read_json(TEST_JSON, orient="records", lines=False)

# Helper function to get metric embedding
def get_metric_emb(name):
    if name in name_to_emb:
        return name_to_emb[name]
    return np.zeros(metric_embs.shape[1], dtype=float)

train_metric_emb = np.vstack([get_metric_emb(n) for n in train["metric_name"]])
test_metric_emb = np.vstack([get_metric_emb(n) for n in test["metric_name"]])

# Initialize target variables
y_train_raw = train["score"].astype(float).values
shift = 1.0 
y_train_log = np.log1p(y_train_raw + shift)

# B: Instruction (Always English)
train["instruction_text"] = train["system_prompt"]
test["instruction_text"] = test["system_prompt"]

# C: Communication (Variable Language)
train["communication_text"] = train["user_prompt"] + " " + train["response"]
test["communication_text"] = test["user_prompt"] + " " + test["response"]

train["instruction_text"] = train["instruction_text"].fillna("")
test["instruction_text"] = test["instruction_text"].fillna("")
train["communication_text"] = train["communication_text"].fillna("")
test["communication_text"] = test["communication_text"].fillna("")

# Load the Sentence Transformer model
# A good general-purpose or multilingual model is necessary here
model = SentenceTransformer("google/embeddinggemma-300m") 

# 1. Instruction Embeddings (B) - System Prompt
instruction_emb_train = model.encode(train["instruction_text"].tolist(), show_progress_bar=True, batch_size=32, convert_to_numpy=True)
instruction_emb_test = model.encode(test["instruction_text"].tolist(), show_progress_bar=True, batch_size=32, convert_to_numpy=True)

# 2. Communication Embeddings (C) - Prompt + Response
communication_emb_train = model.encode(train["communication_text"].tolist(), show_progress_bar=True, batch_size=32, convert_to_numpy=True)
communication_emb_test = model.encode(test["communication_text"].tolist(), show_progress_bar=True, batch_size=32, convert_to_numpy=True)



# Normalize all vectors for robust similarity calculation
metric_norm_train = normalize(train_metric_emb)
instruction_norm_train = normalize(instruction_emb_train)
communication_norm_train = normalize(communication_emb_train)

metric_norm_test = normalize(test_metric_emb)
instruction_norm_test = normalize(instruction_emb_test)
communication_norm_test = normalize(communication_emb_test)

features_train = []
features_test = []

# --- 3.1 Raw Embeddings (A, B, C) ---
features_train.extend([train_metric_emb, instruction_emb_train, communication_emb_train])
features_test.extend([test_metric_emb, instruction_emb_test, communication_emb_test])

# --- 3.2 Metric (A) vs Communication (C): Goal vs Result ---
# Cosine Similarity
features_train.append((metric_norm_train * communication_norm_train).sum(axis=1, keepdims=True))
features_test.append((metric_norm_test * communication_norm_test).sum(axis=1, keepdims=True))
# Element-wise Difference
features_train.append(metric_norm_train - communication_norm_train)
features_test.append(metric_norm_test - communication_norm_test)
# Element-wise Product (Hadamard)
features_train.append(metric_norm_train * communication_norm_train)
features_test.append(metric_norm_test * communication_norm_test)

# --- 3.3 Instruction (B) vs Communication (C): Rules vs Result ---
# Cosine Similarity
features_train.append((instruction_norm_train * communication_norm_train).sum(axis=1, keepdims=True))
features_test.append((instruction_norm_test * communication_norm_test).sum(axis=1, keepdims=True))
# Element-wise Difference
features_train.append(instruction_norm_train - communication_norm_train)
features_test.append(instruction_norm_test - communication_norm_test)

# Final Feature Stacking
X_train_base = np.hstack(features_train)
X_test_base = np.hstack(features_test)


print(f"Total features created: {X_train_base.shape[1]}")
