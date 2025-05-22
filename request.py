import requests

url = "http://localhost:8000/translate/stream"
data = {
    "text": "Hello, how are you?",
    "source_language": "en",
    "target_language": "ar"
}
response = requests.post(url, json=data)
print(response.text)