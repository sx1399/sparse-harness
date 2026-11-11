"""
Tests for the OpenAI-Compatible API Gateway Proxy
"""

import pytest
from fastapi.testclient import TestClient
from sparse_harness.gateway.proxy import app

client = TestClient(app)

def test_list_models_endpoint():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "sparse-harness-auto"

def test_chat_completions_endpoint():
    payload = {
        "model": "sparse-harness-auto",
        "messages": [
            {"role": "user", "content": "Calculate 15% discount on $200 and return net price."}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert "Verified Result:" in data["choices"][0]["message"]["content"]
    assert data["usage"]["total_tokens"] > 0
    assert "telemetry" in data
