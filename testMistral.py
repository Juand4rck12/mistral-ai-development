import requests

# Ollama endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Payload + prompt
payload = {
    "model" : "mistral",
    "prompt" : "¿Que es la IA?",
    "stream" : False
}

# Enviar request
response = requests.post(OLLAMA_API_URL, json=payload)

# Mostrar respuesta
print(response.json()["response"])