"""
Demonstration: Using OpenAI Python SDK with Sparse Node Harness
Shows how any existing enterprise codebase connects via a one-line change.
"""

# In your application, simply point the standard OpenAI client to the local gateway:
# from openai import OpenAI
# client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")

# response = client.chat.completions.create(
#     model="sparse-harness-auto",
#     messages=[
#         {"role": "user", "content": "Calculate 15% discount on $200 and return net price."}
#     ]
# )
# print(response.choices[0].message.content)

import requests
import json

def test_local_gateway():
    url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "model": "sparse-harness-auto",
        "messages": [
            {"role": "user", "content": "Calculate 15% discount on $200 and return net price."}
        ]
    }
    print("Testing Sparse Node Harness Proxy at:", url)
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            print("Response:", json.dumps(res.json(), indent=2))
        else:
            print("Error:", res.status_code, res.text)
    except Exception as e:
        print("Note: Start the gateway first via 'sparse-harness serve' before running this demo.")

if __name__ == "__main__":
    test_local_gateway()
