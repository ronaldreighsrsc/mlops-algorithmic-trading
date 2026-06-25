import os
import re

models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Modelos puramente ML que usaremos con Meta-Labeling
ml_models = ['random_forest.py', 'xgb_model.py']
dl_models = ['lstm_model.py', 'bilstm_model.py', 'lstm_rf.py']

print("Patching ML Models for Meta-Labeling...")

def patch_file(filepath, is_dl=False):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parche 1: prepare_sequences
    # Extraemos target_col antes de crear dataset
    old_prep = r"cols = \[target_col\] \+ \[c for c in feature_cols if c != target_col\]\s*dataset = .*?\[cols\]\.values\s*X_raw, y_raw = \[\], \[\]\s*for i in range\(self\.look_back, len\(dataset\)\):\s*.*?\s*X_raw\.append\(.*?\)\s*# Etiqueta: 1 si el precio subió.*?\s*y_raw\.append\(1 if dataset\[i, 0\] > dataset\[i-1, 0\] else 0\)"
    
    # Variante para los que tienen dataset[i, 0] > dataset[i-1, 0] sin el comentario exacto
    old_prep_alt = r"cols = \[target_col\] \+ \[c for c in feature_cols if c != target_col\]\n\s*dataset = .*?\[cols\]\.values\n\s*X_raw, y_raw = \[\], \[\]\n\s*for i in range\(self\.look_back, len\(dataset\)\):\n\s*.*?\n\s*X_raw\.append\(.*?\)\n\s*.*?y_raw\.append\(1 if dataset\[i, 0\] > dataset\[i-1, 0\] else 0\)"

    if is_dl:
        new_prep = """        y_array = df_clean[target_col].values
        features = [c for c in feature_cols if c != target_col]
        dataset = df_clean[features].values
        
        X_raw, y_raw = [], []
        for i in range(self.look_back, len(dataset)):
            X_raw.append(dataset[i-self.look_back:i, :])
            y_raw.append(y_array[i])"""
    else:
        new_prep = """        y_array = df_clean[target_col].values
        features = [c for c in feature_cols if c != target_col]
        dataset = df_clean[features].values
        
        X_raw, y_raw = [], []
        for i in range(self.look_back, len(dataset)):
            window = dataset[i-self.look_back:i, :]
            X_raw.append(window.flatten())
            y_raw.append(y_array[i])"""

    # En RF/XGB no hay df_clean por defecto, así que lo inyectamos
    if 'df_clean = df.copy()' not in content:
        content = content.replace("def prepare_sequences(self, df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:",
                                  "def prepare_sequences(self, df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:\n        df_clean = df.copy()\n        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().bfill()")

    # Usamos re.sub
    content = re.sub(old_prep_alt, new_prep, content, flags=re.DOTALL)

    # Parche 2: walk_forward_predict (para LSTM_RF)
    if 'lstm_rf' in filepath:
        content = re.sub(r"y_train_list\.append\(1 if actual_val > history_target\[-1\] else 0\)",
                         r"y_train_list.append(y_test[i])", content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Patched {os.path.basename(filepath)}")


for m in ml_models:
    patch_file(os.path.join(models_dir, m), is_dl=False)

for m in dl_models:
    patch_file(os.path.join(models_dir, m), is_dl=True)

print("Patching complete.")
