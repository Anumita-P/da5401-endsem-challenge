import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

X_train_full = np.hstack([metric_norm_train, instruction_norm_train, communication_norm_train])
X_test_full = np.hstack([metric_norm_test, instruction_norm_test, communication_norm_test])

y_train_raw = train['score'].values.astype(np.float32)
shift = 1.0 
y_train_log = np.log1p(y_train_raw + shift).astype(np.float32) # Standard log-transformation
#gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Data arrays defined. Using device: {device}")


class VectorDataset(Dataset):
    """Dataset that provides X, y_log, AND y_raw (for weighting)."""
    def __init__(self, X, y_log, y_raw):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32).unsqueeze(1)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_log[idx], self.y_raw[idx]

def weighted_mae_loss(y_pred_log, y_true_log, y_true_raw, low_score_threshold=5.0, low_score_weight=30.0):
    
    
    # Identify low scores based on the RAW score
    low_score_mask = (y_true_raw <= low_score_threshold).float()
    
    # Create the weight tensor: low_score_weight for low scores, 1.0 otherwise
    weights = (1.0 - low_score_mask) * 1.0 + low_score_mask * low_score_weight
    
    # Calculate the absolute error
    absolute_error = torch.abs(y_true_log - y_pred_log)
    
    # Apply weights and calculate the weighted mean loss
    weighted_loss = torch.sum(absolute_error * weights) / torch.sum(weights)
    
    return weighted_loss

# cnn

class CNNRegressor(nn.Module):
    def __init__(self, input_features, base_filters=32):
        super(CNNRegressor, self).__init__()
        
        # 1. Feature Extraction (1D CNN)
        # Input: (Batch, Features) -> Reshaped to (Batch, 1, Features) for Conv1D
    
        self.cnn_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(),
        )
        
       

        final_cnn_size = int(input_features / 4) * (base_filters * 4) 
        self.pool = nn.AdaptiveMaxPool1d(1) 
        
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(base_filters * 4, 512), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) 
        )

    def forward(self, x):

        x = x.unsqueeze(1) 
        
        x = self.cnn_blocks(x) # Output (Batch, base_filters*4, Length)
        
        x = self.pool(x) # Output (Batch, base_filters*4, 1)
        
        x = x.squeeze(-1) # Output (Batch, base_filters*4)
        
        x = self.head(x)
        return x


#training

def train_one_epoch(model, dataloader, optimizer, epoch, low_weight):
    model.train()
    total_loss = 0
    for X, y_log, y_raw in dataloader:
        X, y_log, y_raw = X.to(device), y_log.to(device), y_raw.to(device)
        
        optimizer.zero_grad()
        y_pred_log = model(X)
        
        # weighted loss
        loss = weighted_mae_loss(y_pred_log, y_log, y_raw, low_score_weight=low_weight)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader):
    model.eval()
    y_true_log_list = []
    y_pred_log_list = []
    with torch.no_grad():
        for X, y_log, _ in dataloader:
            X = X.to(device)
            y_pred_log = model(X)
            y_true_log_list.append(y_log.cpu().numpy())
            y_pred_log_list.append(y_pred_log.cpu().numpy())
            
    y_true_log = np.concatenate(y_true_log_list).flatten()
    y_pred_log = np.concatenate(y_pred_log_list).flatten()
    return y_true_log, y_pred_log

def predict(model, X_data):
    model.eval()
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    test_dataset = torch.utils.data.TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            X = batch[0]
            pred = model(X).cpu().numpy().flatten()
            predictions.append(pred)
            
    return np.concatenate(predictions)


#cross validation

def run_cnn_cv(X, y_log, y_raw, X_test, n_splits=5, epochs=100, batch_size=64, lr=1e-4, low_weight=30.0):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_log = np.zeros(len(X))
    test_preds_log = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # Data split
        X_tr, X_val = X[train_idx], X[val_idx]
        y_log_tr, y_log_val = y_log[train_idx], y_log[val_idx]
        y_raw_tr, y_raw_val = y_raw[train_idx], y_raw[val_idx]
        
        # DataLoader setup
        train_dataset = VectorDataset(X_tr, y_log_tr, y_raw_tr)
        val_dataset = VectorDataset(X_val, y_log_val, y_raw_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Model setup
        model = CNNRegressor(input_features=X.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_val_rmse = float('inf')
        patience_counter = 0
        patience = 10 # Early stopping
        
        for epoch in tqdm(range(epochs), desc=f"Fold {fold+1} Epoch"):
            
            # Training
            train_loss = train_one_epoch(model, train_loader, optimizer, epoch, low_weight)
            
            # Validation
            y_true_log_val, y_pred_log_val = evaluate(model, val_loader)
            y_pred_raw_val = np.expm1(y_pred_log_val) - shift
            y_true_raw_val = np.expm1(y_true_log_val) - shift

            val_rmse = np.sqrt(mean_squared_error(y_true_raw_val, y_pred_raw_val))
            
        
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), f"best_cnn_fold{fold}.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
                
        # Load best model for OOF and Test Prediction
        model.load_state_dict(torch.load(f"best_cnn_fold{fold}.pth"))
        
        # OOF prediction
        _, oof_log_fold = evaluate(model, val_loader)
        oof_log[val_idx] = oof_log_fold
        
        # Test prediction
        test_preds_log += predict(model, X_test) / n_splits

    return oof_log, test_preds_log


if __name__ == '__main__':
    try:
        if 'X_train_full' not in globals() or X_train_full is None:
            raise NameError("Data not loaded. Please replace placeholders with actual data.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(f"Using device: {device}")
        
    except NameError as e:
        print(f"ERROR: {e}")
        exit()


    N_EPOCHS = 100 
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    LOW_SCORE_WEIGHT = 40.0 

    oof_log, test_preds_log = run_cnn_cv(
        X=X_train_full, y_log=y_train_log, y_raw=y_train_raw, X_test=X_test_full, 
        n_splits=5, epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, low_weight=LOW_SCORE_WEIGHT
    )

    oof_final_raw = np.expm1(oof_log) - shift
    preds_final_raw = np.expm1(test_preds_log) - shift
    
    rmse = np.sqrt(mean_squared_error(y_train_raw, oof_final_raw))
    print(f"\nFinal CNN CV RMSE (Weighted MAE): {rmse:.4f}")


    y_train_bins = pd.cut(y_train_raw, bins=[-0.001, 5.0, 8.0, 10.0], labels=[0, 1, 2])
    oof_analysis = pd.DataFrame({'true': y_train_raw, 'pred': oof_final_raw, 'bin': y_train_bins})
    rmse_low = np.sqrt(mean_squared_error(oof_analysis[oof_analysis['bin'] == 0]['true'], oof_analysis[oof_analysis['bin'] == 0]['pred']))
    print(f"-> RMSE for Low Scores (<= 5.0): {rmse_low:.4f}")
    

    final_preds = np.clip(preds_final_raw, 0.0, 10.0)
    
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    submission_filename = f"submission_Final_CNN_WeightedMAE_{now}.csv"
    
    

    if 'id' in test.columns:
        ids = test["id"].astype(str)
    else:
        ids = [str(i + 1) for i in range(len(test))]


    sub = pd.DataFrame({"ID": ids, "score": final_preds})
    sub.to_csv(submission_filename, index=False)

    print(f"Saved submission to {submission_filename}")

    bins = [0, 2, 4, 6, 8, 10]
    sub['score_bin'] = pd.cut(sub['score'], bins=bins, right=True, include_lowest=True)
    print("\nSubmission Score Distribution:")
    print(sub['score_bin'].value_counts().sort_index()) 