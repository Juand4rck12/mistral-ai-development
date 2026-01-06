import requests

message = input("Digite su prompt: ")

response = requests.post(
    "http://localhost:11434/api/generate",
    json= {
        "model" : "mistral",
        "prompt" : message,
        "stream" : False
    }
)

print(response.json()["response"])