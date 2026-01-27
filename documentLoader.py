import os, requests
import fitz # PyMuPDF for PDFs
import docx

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


# Cargar el modelo de embeddings (definido antes de usar en funciones)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def search_and_summarize(query, db_path="chroma_db"):
    """RAG moderno con LangChain v1 + Chroma + Mistral usando create_agent"""
    
    # Cargar ChromaDB
    vectorstore = Chroma(
        collection_name="documents",
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    
    # Crear herramienta de recuperación de contexto
    @tool(response_format="content_and_artifact")
    def retrieve_context(search_query: str):
        """Recupera información relevante de la base de datos para responder consultas."""
        retrieved_docs = vectorstore.similarity_search(search_query, k=3)
        serialized = "\n\n".join(
            f"Contenido: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    # Middleware para inyectar contexto dinámicamente
    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inyecta contexto recuperado en el prompt del sistema."""
        last_query = request.state["messages"][-1].text if hasattr(request.state["messages"][-1], 'text') else str(request.state["messages"][-1])
        retrieved_docs = vectorstore.similarity_search(last_query, k=3)
        
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        system_message = (
            "Eres un asistente de IA útil. Usa SOLO la información del contexto para responder. "
            "Si no sabes la respuesta, di que no está en el contexto."
            f"\n\nContexto:\n{docs_content}"
        )
        
        return system_message
    
    # Inicializar modelo Ollama Mistral con URL explícita
    mistral_model = ChatOllama(
        model="mistral",
        base_url="http://127.0.0.1:11434"  # Usar 127.0.0.1 en lugar de localhost
    )
    
    # Crear agente con middleware de RAG
    agent = create_agent(
        model=mistral_model,
        tools=[retrieve_context],
        middleware=[prompt_with_context],
        system_prompt=(
            "Eres un asistente que responde preguntas basándose en el contexto proporcionado. "
            "Usa la herramienta retrieve_context si necesitas buscar información adicional."
        )
    )
    
    # Invocar el agente
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    print("\nRespuesta con IA:")
    print(response["messages"][-1].content if hasattr(response["messages"][-1], 'content') else response["messages"][-1])
    
    # Mostrar fuentes si están disponibles
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    print("\nFuentes:")
    for doc in retrieved_docs:
        print(f"- {doc.page_content[:300]}...")



# Ollama endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def generate_ai_response(context, query):
    """Envar consulta del usuario junto con los documentos recuperados a Mistral AI para RAG"""
    prompt = f"""
    Eres un asistente de IA con acceso a la siguiente información:
    {context}
    Basandote en eso, responde la siguiente consulta:
    {query}
    """

    payload = {"model": "mistral", "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_API_URL, json=payload)

    return response.json().get("response", "No se genero una respuesta")


def search_and_generate_response(query, db_path="chroma_db"):
    """Recuperar documentos relevantes y usar Mistral AI para respuesta contextual"""
    vectorStore = Chroma(collection_name="documents", persist_directory=db_path, embedding_function=embedding_model)
    results = vectorStore.similarity_search(query, k=3)

    # Combinar documentos recuperados en contexto
    context = "\n\n".join([doc.page_content for doc in results])

    # Generar respuesta de IA usando RAG
    ai_response = generate_ai_response(context, query)

    print("\nRespuesta con IA:")
    print(ai_response)


def process_document(file_path):
    """Extraer texto, cortarlo, y convertirlo a embeddings"""
    text = extract_text(file_path)

    if not text:
        return None
    
    # Cortar el texto en trozos pequeños para mejor rendimiento en busqueda
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_text(text)

    return texts


def store_embeddings(texts, db_path = "chroma_db"):
    """Almacena texto embeddings en ChromaDB"""
    vectorStore = Chroma(collection_name="documents", persist_directory=db_path, embedding_function=embedding_model)
    vectorStore.add_texts(texts)

    print("Embeddings almacenados correctamente!")


def search_documents(query, db_path="chroma_db"):
    """Busca embeddings almacenados en ChromaDB"""
    vectorStore = Chroma(collection_name="documents", persist_directory=db_path, embedding_function=embedding_model)
    results = vectorStore.similarity_search(query, k=5) # Recupera 5 coincidencias
    
    if not results:
        print("No se encontraron resultados para tu consulta.")
        return

    for idx, result in enumerate(results):
        print(f"\n Resultado {idx + 1}:")
        print(result.page_content)

    return results


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
    sample_file = "./documentos/Logica y razonamiento.pdf"
    texts = process_document(sample_file)
    if texts:
        store_embeddings(texts)

    while True:
        user_query = input("Ingresa tu consulta de busqueda (o 'salir' para terminar): ")
        if user_query.lower() == 'salir':
            break
        # results = search_documents(user_query)
        # search_and_generate_response(user_query)
        search_and_summarize(user_query)

# # Ejemplo de uso
# if __name__ == "__main__":
#     sample_pdf = "./documentos/48 leyes del Poder.pdf"
#     sample_word = "./documentos/Tech Support & Problem Solving Workshop.docx"
#     sample_txt = "./documentos/Direcciones USA.txt"
#     print(extract_text(sample_txt))