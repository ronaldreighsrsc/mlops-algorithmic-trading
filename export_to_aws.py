import os
import zipfile
import glob
import json

def create_production_zip(output_filename="bot_production.zip"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    saved_models_dir = os.path.join(results_dir, "saved_models")
    
    print("Preparando paquete de Produccion para AWS (Modo Inteligente)...")
    
    # 1. Archivos base estrictamente necesarios
    includes = [
        "src/**/*.py",             
        "requirements.txt",        
        ".env",                    
        "start_bot.bat",           
        "README.md",               
        "results/hrp_weights.json", 
    ]
    
    # 2. Descubrir campeones y empacar TODO lo que necesitan
    campeon_files = []
    for json_file in glob.glob(os.path.join(results_dir, "campeon_*.json")):
        # a) El JSON del campeón
        campeon_files.append(os.path.relpath(json_file, base_dir))
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        model_file = data.get("model_file", "")
        symbol = os.path.basename(json_file).replace("campeon_", "").replace(".json", "")
        
        if model_file:
            # b) El .pkl del modelo predictivo
            pkl_path = os.path.join(saved_models_dir, model_file)
            if os.path.exists(pkl_path):
                campeon_files.append(os.path.relpath(pkl_path, base_dir))
            
            # c) El .keras satelital (para ARIMA_LSTM, LSTM, BiLSTM, LSTM_RF)
            keras_model_path = pkl_path.replace(".pkl", ".keras")
            if os.path.exists(keras_model_path):
                campeon_files.append(os.path.relpath(keras_model_path, base_dir))
                print(f"  [+] .keras satelital detectado: {os.path.basename(keras_model_path)}")
        
        # d) MLOps Monitors del campeón (autoencoder + HMM)
        #    Estos están en results/ directamente como campeon_{SYMBOL}_autoencoder.keras, etc.
        mlops_patterns = [
            f"campeon_{symbol}_autoencoder.keras",
            f"campeon_{symbol}_autoencoder_meta.pkl",
            f"campeon_{symbol}_hmm.pkl",
        ]
        for pattern in mlops_patterns:
            mlops_path = os.path.join(results_dir, pattern)
            if os.path.exists(mlops_path):
                campeon_files.append(os.path.relpath(mlops_path, base_dir))
                print(f"  [+] MLOps monitor: {pattern}")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        added_files = set()
        total_size = 0
        
        # Primero agregar los archivos descubiertos de campeones
        for rel_path in campeon_files:
            full_path = os.path.join(base_dir, rel_path)
            if os.path.isfile(full_path) and rel_path not in added_files:
                zipf.write(full_path, rel_path)
                added_files.add(rel_path)
                total_size += os.path.getsize(full_path)
        
        # Luego agregar los includes generales
        for pattern in includes:
            full_pattern = os.path.join(base_dir, pattern)
            for file_path in glob.glob(full_pattern, recursive=True):
                if os.path.isfile(file_path):
                    if "__pycache__" in file_path or ".git" in file_path:
                        continue
                    # Prevenir archivos pesados innecesarios y estados locales de MLOps
                    if file_path.endswith((".npy", ".png", ".html", ".lock")):
                        continue
                        
                    rel_path = os.path.relpath(file_path, base_dir)
                    if rel_path not in added_files:
                        zipf.write(file_path, rel_path)
                        added_files.add(rel_path)
                        total_size += os.path.getsize(file_path)
                    
    print(f"\nExito! Archivo creado: {output_filename}")
    print(f"Archivos empaquetados: {len(added_files)}")
    print(f"Tamano Total: {total_size / (1024*1024):.2f} MB")
    
    # Resumen de lo empaquetado por campeón
    print("\n--- Resumen por Campeon ---")
    for json_file in sorted(glob.glob(os.path.join(results_dir, "campeon_*.json"))):
        with open(json_file, 'r') as f:
            data = json.load(f)
        symbol = os.path.basename(json_file).replace("campeon_", "").replace(".json", "")
        print(f"  {symbol}: {data['model_type']} ({data['banco']}) -> {data['model_file']}")

if __name__ == "__main__":
    create_production_zip()
