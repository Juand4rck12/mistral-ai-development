# Mistral AI Development

Sistema de procesamiento de documentos con Recuperación Aumentada por Generación (RAG) utilizando Mistral AI, LangChain v1 y Ollama.

---

## Descripción del Proyecto

**Mistral AI Development** es una aplicación Python que permite procesar, indexar y consultar documentos utilizando un sistema RAG moderno. Integra modelos de lenguaje ejecutados localmente con Ollama, bases de datos vectoriales con Chroma, y el framework LangChain v1 para crear agentes inteligentes que pueden responder preguntas basándose en el contenido de los documentos procesados.

---

## Propósito y Alcance

### Propósito
Proporcionar una solución escalable y privada para:
- **Procesamiento inteligente de documentos**: Extrae y procesa texto de PDFs, Word y archivos de texto
- **Búsqueda semántica**: Recupera documentos relevantes basándose en similitud vectorial
- **Generación contextual**: Genera respuestas precisas utilizando información recuperada
- **Privacidad**: Ejecuta modelos localmente sin enviar datos a servidores externos

### Alcance
El proyecto cubre:
- Extracción de texto de múltiples formatos (PDF, DOCX, TXT)
- Segmentación inteligente de documentos
- Generación de embeddings semánticos
- Almacenamiento y recuperación vectorial
- Creación de agentes RAG con LangChain v1
- Interfaz interactiva por línea de comandos

---

## Funcionalidades

### ✓ Carga de Documentos
- Soporte para PDF, Word (.docx) y archivos de texto
- Extracción automática de contenido
- Segmentación con solapamiento configurable

### ✓ Indexación Vectorial
- Embeddings con `sentence-transformers/all-mpnet-base-v2`
- Almacenamiento en ChromaDB
- Persistencia en disco

### ✓ RAG Moderno
- Agentes inteligentes con `create_agent()` de LangChain v1
- Herramientas de recuperación de contexto
- Inyección dinámica de prompts mediante middleware
- Soporte para herramientas personalizadas

### ✓ Modelos Locales
- Integración con Ollama para modelos locales
- Mistral como modelo de generación principal
- Sin dependencia de APIs externas

---

## Requisitos

### Sistema
- Python 3.11+
- 8GB RAM (recomendado)
- Ollama instalado y ejecutándose localmente

### Dependencias
Se instalan automáticamente mediante `requirements.txt`:
```
langchain>=1.0.0
langchain-core>=0.3.0
langgraph
langchain-ollama
langchain-chroma
langchain-huggingface
langchain-text-splitters
sentence-transformers
chromadb
pymupdf
python-docx
```

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd mistral-ai-development
```

### 2. Crear entorno virtual
```bash
python -m venv venv
```

### 3. Activar entorno virtual
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Configurar Ollama
```bash
# Instalar Ollama desde https://ollama.ai
# Ejecutar Ollama
ollama serve

# En otra terminal, descargar modelo Mistral
ollama pull mistral
```

---

## Uso

### Estructura de archivos
```
mistral-ai-development/
├── documentLoader.py          # Módulo principal de RAG
├── testMistral.py            # Script de pruebas
├── requirements.txt          # Dependencias Python
├── documentos/               # Carpeta de documentos a procesar
│   └── ejemplo.pdf
└── chroma_db/               # Base de datos vectorial (generada)
```

### Ejecución básica

```bash
python documentLoader.py
```

El programa:
1. Busca documentos en la carpeta `documentos/`
2. Extrae y procesa el contenido
3. Genera embeddings y los almacena en ChromaDB
4. Inicia un bucle interactivo para consultas

### Ejemplo de uso

```
Ingresa tu consulta de busqueda (o 'salir' para terminar): ¿Cuál es la fecha de vencimiento?

Respuesta con IA:
Según los documentos disponibles, la fecha de vencimiento es...

Fuentes:
- El documento menciona que... (primeros 300 caracteres)
```

---

## Funciones Principales

### `search_and_summarize(query, db_path="chroma_db")`
Ejecuta una consulta RAG completa con contexto inyectado dinámicamente.

**Parámetros:**
- `query` (str): Pregunta del usuario
- `db_path` (str): Ruta de la base de datos vectorial

**Retorno:** Imprime respuesta e identifica fuentes

**Ejemplo:**
```python
from documentLoader import search_and_summarize

search_and_summarize("¿Cuáles son los requisitos principales?")
```

### `process_document(file_path)`
Procesa un archivo y extrae texto segmentado.

**Parámetros:**
- `file_path` (str): Ruta del archivo (PDF, DOCX, TXT)

**Retorno:** Lista de segmentos de texto

**Ejemplo:**
```python
texts = process_document("./documentos/archivo.pdf")
```

### `store_embeddings(texts, db_path="chroma_db")`
Almacena embeddings en ChromaDB.

**Parámetros:**
- `texts` (list): Lista de segmentos de texto
- `db_path` (str): Ruta de la base de datos

**Retorno:** Confirma almacenamiento

**Ejemplo:**
```python
store_embeddings(texts)
```

### `search_documents(query, db_path="chroma_db")`
Búsqueda simple por similitud sin RAG.

**Parámetros:**
- `query` (str): Término de búsqueda
- `db_path` (str): Ruta de la base de datos

**Retorno:** Lista de documentos relevantes

**Ejemplo:**
```python
results = search_documents("búsqueda")
for result in results:
    print(result.page_content)
```

---

## Arquitectura del Sistema

### Flujo de procesamiento

```
Documento original
    ↓
Extracción de texto (extract_text)
    ↓
Segmentación (RecursiveCharacterTextSplitter)
    ↓
Generación de embeddings (HuggingFaceEmbeddings)
    ↓
Almacenamiento vectorial (ChromaDB)
    ├─────────────────┐
    ↓                 ↓
Búsqueda simple    Agente RAG
    ↓                 ↓
Resultados      Mistral + Contexto
                      ↓
                   Respuesta
```

### Componentes principales

| Componente | Función | Tecnología |
|-----------|---------|-----------|
| Extractor de texto | Parseo de documentos | PyMuPDF, python-docx |
| Splitter | Segmentación de texto | LangChain |
| Embedder | Vectorización de texto | HuggingFace Transformers |
| Vector Store | Almacenamiento y búsqueda | Chroma |
| LLM | Generación de respuestas | Ollama + Mistral |
| Agente | Orquestación inteligente | LangChain v1 |

---

## Configuración Avanzada

### Cambiar el modelo de IA
Editar en `documentLoader.py`:
```python
# Línea ~56
mistral_model = ChatOllama(model="neural-chat")  # Cambiar "mistral"
```

### Ajustar parámetros de segmentación
Editar en `documentLoader.py`:
```python
# Línea ~125
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Aumentar para segmentos más grandes
    chunk_overlap=100     # Aumentar para más contexto compartido
)
```

### Cambiar cantidad de documentos recuperados
Editar en `documentLoader.py`:
```python
retrieved_docs = vectorstore.similarity_search(query, k=5)  # Cambiar k
```

---

## Solución de problemas

### Error: "Ollama is not running"
**Solución:** Ejecutar Ollama en otra terminal:
```bash
ollama serve
```

### Error: "Model 'mistral' not found"
**Solución:** Descargar el modelo:
```bash
ollama pull mistral
```

### Error: "CUDA out of memory"
**Solución:** Ollama está usando GPU. Liberar memoria o usar CPU:
```bash
OLLAMA_NUM_GPU=0 ollama serve
```

### Búsquedas lentas
**Causa:** ChromaDB indexando por primera vez
**Solución:** Esperar a que termine o eliminar carpeta `chroma_db/` para reiniciar

---

## Desarrollo

### Agregar nueva herramienta al agente
```python
from langchain.tools import tool

@tool
def mi_herramienta(parametro: str) -> str:
    """Descripción de la herramienta"""
    return "resultado"

# Luego añadir a create_agent:
agent = create_agent(
    model=mistral_model,
    tools=[retrieve_context, mi_herramienta],  # Agregar aquí
    middleware=[prompt_with_context],
)
```

### Agregar nuevo middleware
```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def mi_middleware(request: ModelRequest) -> str:
    # Tu lógica aquí
    return "nuevo_prompt"
```

---