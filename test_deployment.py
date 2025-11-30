import requests
import base64
import json
import sys

# Replace with the actual URL after deployment
# It usually looks like https://<workspace>-<app-name>-<function-name>.modal.run
# For now, I'll take it as an argument or placeholder
if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    print("Usage: python test_deployment.py <url> [image_path]")
    sys.exit(1)

image_path = sys.argv[2] if len(sys.argv) > 2 else "assets/demo.jpg"

try:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    sys.exit(1)

payload = {
    "image": image_b64,
    "prompt": "the airplane on the right",
    "task": "referring_seg"
}

print(f"Sending request to {url}...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Success!")
    result = response.json()
    # print(json.dumps(result, indent=2))
    print("Result received (truncated):", str(result)[:200])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
