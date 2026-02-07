import os, documentLoader
from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "documents"
NAMESPACE = f"chroma/{COLLECTION_NAME}"
RECORD_MANAGER_DB_URL = "sqlite:///record_manager_cache.sql"

# 1. Inicializar el modelo de embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Inicializar el vectorstore persistente
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

# 3. Inicializar el RecordManager
record_manager = SQLRecordManager(
    namespace=NAMESPACE,
    db_url=RECORD_MANAGER_DB_URL
)
record_manager.create_schema() # Crear las tablas si no existen

# 4. Función para cargar y dividir documentos
def load_and_split_documents(file_paths):
    """Carga documentos y los divide en fragmentos."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    all_docs = []
    for file_path in file_paths:
        text = documentLoader.extract_text(file_path)
        if text:
            chunks = text_splitter.split_text(text)
            # Crear objetos Document con metadatos que incluyan la fuente
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                all_docs.append(doc)
    return all_docs


# 5. Indexación incremental
def run_indexing(file_paths):
    """Ejecuta la indexación incremental de los documentos."""
    documents = load_and_split_documents(file_paths)

    """ Llamada clave. El parámetro `cleanup="incremental"` asegura que solo se
        procesen los cambios."""
    indexing_stats = index(
        docs_source=documents,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="incremental", # Modo incremental: solo actualiza lo cambiado
        source_id_key="source", # Metadato que identifica el documento fuente
        force_update=False # Si es True, reindexa todo; normalmente False
    )

    print(f"Estadísticas de indexación: {indexing_stats}")
    return indexing_stats


if __name__ == "__main__":
    # Lista de rutas a documentos
    document_paths = [
        "./documentos/Logica y razonamiento.pdf"
        # Se agregan mas rutas...
    ]
    stats = run_indexing(document_paths)
    print(f"Indexación completada. \n{stats['num_added']} añadidos \n{stats['num_updated']} actualizados \n{stats['num_skipped']} omitidos.")