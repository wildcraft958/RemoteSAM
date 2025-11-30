#!/usr/bin/env python3
import requests
import base64
import json

url = "https://aryan-don357--remotesam-inference-remotesamservice-web-infer.modal.run"

with open("assets/demo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Test object detection
payload = {
    "image": image_b64,
    "categories": ["airplane"],
    "task": "object_det"
}

print("Testing object detection...")
response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
