import os, fitz, docx, hashlib
from datetime import datetime
from typing import List, Optional

def extract_text_from_pdf(pdf_path):
    """Extraer texto de un archivo PDF"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error leyendo PDF {pdf_path}: {e}")
    return text


def extract_text_from_word(doc_path):
    """Extraer texto de un archivo Word (.docx file)"""
    text = ""
    try:
        doc = docx.Document(doc_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error leyendo archivo Word: {doc_path}: {e}")
    return text


def extract_text_from_txt(txt_path):
    """Extraer texto de un archivo TXT"""
    text = ""
    try:
        with open(txt_path, encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        print(f"Error leyendo archivo TXT: {txt_path}: {e}")
    return text


def extract_text(file_path):
    """Detecta el tipo de archivo y extrae el texto acorde"""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_word(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        print(f"Formato de archivo no soportado: {file_path}")


# Nuevas funciones para verificar cambios en documentos
def get_file_hash(file_path: str) -> Optional[str]:
    """Calcula el hash MD5 de un archivo para detectar cambios"""
    if not os.path.exists(file_path): return None

    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()
    except Exception as e:
        print(f"Error calculando hash de {file_path}: {e}")
        return None
    

def get_file_metadata(file_path: str) -> dict:
    """Obtiene metadatos del archivo (tamaño, fecha modificación, etc.)"""
    if not os.path.exists(file_path): return {}

    try:
        stat = os.stat(file_path)
        return {
            "path": file_path,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "hash": get_file_hash(file_path)
        }
    except Exception as e:
        print(f"Error obteniendo metadatos de {file_path}: {e}")
        return {}
    

# Ejemplo de uso para pruebas
if __name__ == "__main__":
    # Pruebas básicas de extracción
    test_files = [
        "./documentos/Logica y razonamiento.pdf",
        # Agrega aquí más archivos para probar
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nProcesando: {test_file}")
            metadata = get_file_metadata(test_file)
            print(f"Metadatos: {metadata}")
            
            text = extract_text(test_file)
            if text:
                print(f"Texto extraído (primeros 200 chars): {text[:200]}...")
        else:
            print(f"Archivo no encontrado: {test_file}")