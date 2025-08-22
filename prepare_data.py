# Contenido completo y corregido para: prepare_data.py

import pandas as pd
import os

def load_trec07p_index(index_file):
    """Carga el archivo de índice y devuelve un DataFrame con 'label' y 'full_path'."""
    data = []
    # Esta ruta base se ajusta automáticamente a la localización del archivo 'index'
    base_data_folder = os.path.abspath(os.path.join(os.path.dirname(index_file), '..', 'data'))
    
    with open(index_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = parts[0]
            # La ruta en el archivo index es relativa, como '../../data/000/000'
            # La reconstruimos a partir de nuestra base_data_folder
            path_suffix = os.path.join(*parts[1].split('/')[2:]) # Extrae '000/000'
            full_path = os.path.join(base_data_folder, path_suffix)
            data.append({'label': label, 'full_path': full_path})
    return pd.DataFrame(data)

def read_email_content(file_path):
    """Lee el contenido de un archivo de correo electrónico."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Advertencia: Archivo no encontrado en la ruta: {file_path}")
        return ""

# --- SCRIPT PRINCIPAL ---
print("Iniciando la preparación de datos...")

try:
    # ✅ ¡ESTA ES LA LÍNEA QUE HEMOS CORREGIDO!
    index_file_path = 'datasets/datasets/trec07p/full/index'
    
    print("Cargando índice de archivos...")
    df_index = load_trec07p_index(index_file_path)

    print(f"Se encontraron {len(df_index)} registros. Leyendo contenido de los correos (esto puede tardar)...")
    
    df_index['content'] = df_index['full_path'].apply(read_email_content)

    df_final = df_index[['label', 'content']]
    df_final = df_final[df_final['content'] != ""]

    output_filename = 'spam_dataset_with_content.pkl'
    df_final.to_pickle(output_filename)

    print(f"\n¡Éxito! El dataset con el contenido ha sido guardado como '{output_filename}'")

except FileNotFoundError:
    print(f"\nError: No se encontró el archivo 'index' en la ruta '{index_file_path}'.")
    print("Por favor, verifica que la ruta sea correcta desde donde ejecutas el script.")
except Exception as e:
    print(f"\nOcurrió un error inesperado: {e}")