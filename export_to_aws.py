import os
import zipfile
import glob
import json

def create_production_zip(output_filename="bot_production.zip"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    
    print("Preparando paquete de Produccion para AWS (Modo Inteligente)...")
    
    # 1. Archivos base estrictamente necesarios
    includes = [
        "src/**/*.py",             
        "requirements.txt",        
        ".env",                    
        "start_bot.bat",           
        "README.md",               
        "results/campeon_*.json",  
        "results/hrp_weights.json", 
        "results/campeon_*.*" # Monitores MLOps (están en results/ directamente ahora, no en mlops_monitors)
    ]
    
    # 2. Descubrir cuáles son los modelos campeones y empacar SOLO esos
    campeones_pkl = []
    for json_file in glob.glob(os.path.join(results_dir, "campeon_*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if "model_file" in data:
                campeones_pkl.append(f"results/saved_models/{data['model_file']}")
                
    includes.extend(campeones_pkl)
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        added_files = 0
        total_size = 0
        
        for pattern in includes:
            full_pattern = os.path.join(base_dir, pattern)
            # Para coincidencia exacta de los PKL (sin glob) o con glob
            for file_path in glob.glob(full_pattern, recursive=True):
                if os.path.isfile(file_path):
                    if "__pycache__" in file_path or ".git" in file_path:
                        continue
                        
                    # Prevenir añadir archivos npy por accidente si el glob es muy ancho
                    if file_path.endswith(".npy") or file_path.endswith(".png") or file_path.endswith(".html"):
                        continue
                        
                    rel_path = os.path.relpath(file_path, base_dir)
                    zipf.write(file_path, rel_path)
                    
                    added_files += 1
                    total_size += os.path.getsize(file_path)
                    
    print(f"Exito! Archivo creado: {output_filename}")
    print(f"Archivos empaquetados: {added_files}")
    print(f"Tamano Total: {total_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_production_zip()
