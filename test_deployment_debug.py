import requests
import base64
import json
import sys
import numpy as np

if len(sys.argv) > 1:
    url = sys.argv[1]
else:
    print("Usage: python test_deployment_debug.py <url> [image_path] [prompt]")
    sys.exit(1)

image_path = sys.argv[2] if len(sys.argv) > 2 else "assets/demo.jpg"
prompt = sys.argv[3] if len(sys.argv) > 3 else "the airplane on the right"

try:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    sys.exit(1)

payload = {
    "image": image_b64,
    "prompt": prompt,
    "task": "referring_seg"
}

print(f"Sending request to {url}...")
print(f"Image: {image_path}")
print(f"Prompt: '{prompt}'")
print()

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("✓ Success!")
    result = response.json()
    
    if 'result' in result:
        mask = np.array(result['result'])
        print(f"Mask shape: {mask.shape}")
        print(f"Mask data type: {mask.dtype}")
        print(f"Mask min value: {mask.min()}")
        print(f"Mask max value: {mask.max()}")
        print(f"Number of non-zero pixels: {np.count_nonzero(mask)}")
        print(f"Total pixels: {mask.size}")
        print(f"Percentage non-zero: {100.0 * np.count_nonzero(mask) / mask.size:.2f}%")
        
        if np.count_nonzero(mask) > 0:
            print("\n✓ Mask contains segmentation data!")
            # Show some non-zero values
            nonzero_indices = np.nonzero(mask)
            if len(nonzero_indices[0]) > 0:
                sample_idx = min(10, len(nonzero_indices[0]))
                print(f"Sample non-zero values (first {sample_idx}):")
                for i in range(sample_idx):
                    y, x = nonzero_indices[0][i], nonzero_indices[1][i]
                    print(f"  ({y}, {x}): {mask[y, x]}")
        else:
            print("\n⚠ Warning: Mask is all zeros - no segmentation detected")
    else:
        print("Error: No 'result' key in response")
        print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
