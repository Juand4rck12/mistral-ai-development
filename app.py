from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest
import os

# Inicializar FastAPI APP
app = FastAPI(title="Sistema RAG Local", version="1.0.0")

# Verificar si chromaDB existe
CHROMA_DIR = "./chroma_db"
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    raise RuntimeError(
        "ChromaDB No encontrada. Ejecuta primero: python index_document.py\n"
        "para indexar documentos antes de inciar la aplicación."
    )

# Cargar modelo via ollama
print("--- Cargando modelo Mistral via Ollama ---")
llm = ChatOllama(
    model="mistral",
    base_url="http://127.0.0.1:11434"
)

# Cargar el modelo de embeddings
print("--- Cargando modelo de Embeddings ---")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("--- Cargando base de datos vectorial ---")
# Cargar ChromaDB
vectorstore = Chroma(
    collection_name="documents",
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


def create_rag_agent():
    """Construir un agente RAG moderno con LangChain V1"""
    @tool # Definir herramienta de recuperación
    def retrieve_context(search_query: str):
        """Recupera información relevante de la base de datos vectorial."""
        retrieved_docs = vectorstore.similarity_search(search_query, k=3)
        serialized = "\n\n".join(
            f"Contenido: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    @dynamic_prompt # Definir middleware para inyectar contexto
    def prompt_with_context(request: ModelRequest) -> str:
        """Inyecta contexto recuperado en el prompt del sistema."""
        # Obtener la ultima consulta del usuario
        last_message = request.state["messages"][-1]
        last_query = last_message.content if hasattr(last_message, 'content') else str(last_message)

        # Recuperar documentos relevantes
        retrieved_docs = vectorstore.similarity_search(last_query, k=3)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Construir prompt del sistema con contexto
        system_message = (
            "Eres un asistente de IA útil. Usa SOLO la información del contexto para responder. "
            "Si no sabes la respuesta, di que no está en el contexto.\n\n"
            f"Contexto:\n{docs_content}"
        )

        return system_message
    

    # Crear el agente
    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        middleware=[prompt_with_context],
        system_prompt=(
            "Eres un asistente que responde preguntas basándose en el contexto proporcionado. "
            "Usa la herramienta retrieve_context si necesitas buscar información adicional."
        )
    )

    return agent

print("--- Inicializando agente de IA... ---")
agent = create_rag_agent()
print("--- Agente de IA inicializado correctamente. ---")


# Definir la clase para solicitud
class QueryRequest(BaseModel):
    query: str

# Endpoint de consulta
@app.post("/query")
def search_and_generate_response(request: QueryRequest):
    """Recupera documentos y genera respuestas con IA usando agente moderno."""
    # Invocar el agente con formato de mensajes
    response = agent.invoke({
        "messages": [{"role": "user", "content": request.query}]
    })

    # Extraer la respuesta del agente
    
    ai_response = response["messages"][-1].content if hasattr(response["messages"][-1], 'content') else str(response["messages"][-1])

    return {
        "query": request.query,
        "response": ai_response
    }
    

# Endpoint principal
@app.get("/")
def home():
    return {"message": "Mistral AI-powered search API is running!"}
