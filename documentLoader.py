import os
import fitz # PyMuPDF for PDFs
import docx

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


# Ejemplo de uso
if __name__ == "__main__":
    sample_pdf = "./documentos/48 leyes del Poder.pdf"
    sample_word = "./documentos/Tech Support & Problem Solving Workshop.docx"
    sample_txt = "./documentos/Direcciones USA.txt"
    print(extract_text(sample_txt))