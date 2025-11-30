#!/usr/bin/env python3
"""
Comprehensive test for all 8 RemoteSAM tasks
Tests referring_seg, visual_grounding, semantic_seg, multi_label_cls,
image_cls, object_det, object_counting, and image_caption
"""

import requests
import base64
import json
import sys

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_task(url, task_name, payload):
    """Test a single task"""
    print(f"\n{'='*60}")
    print(f"Testing: {task_name}")
    print(f"{'='*60}")
    print(f"Payload (excluding image): {json.dumps({k:v for k,v in payload.items() if k != 'image'}, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS")
            print(f"Response: {json.dumps(result, indent=2)[:500]}...")
            return True, result
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_all_tasks.py <endpoint_url> [image_path]")
        sys.exit(1)
    
    url = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else "assets/demo.jpg"
    
    print(f"\n{'#'*60}")
    print(f"# RemoteSAM - All Tasks Test Suite")
    print(f"#{'#'*60}")
    print(f"Endpoint: {url}")
    print(f"Test Image: {image_path}\n")
    
    # Encode image once
    image_b64 = encode_image(image_path)
    
    results = {}
    
    # Test 1: Referring Segmentation
    results["referring_seg"] = test_task(url, "1. Referring Segmentation", {
        "image": image_b64,
        "prompt": "the airplane on the right",
        "task": "referring_seg"
    })
    
    # Test 2: Visual Grounding
    results["visual_grounding"] = test_task(url, "2. Visual Grounding", {
        "image": image_b64,
        "prompt": "the airplane",
        "task": "visual_grounding"
    })
    
    # Test 3: Semantic Segmentation
    results["semantic_seg"] = test_task(url, "3. Semantic Segmentation", {
        "image": image_b64,
        "categories": ["airplane", "vehicle"],
        "task": "semantic_seg"
    })
    
    # Test 4: Multi-Label Classification
    results["multi_label_cls"] = test_task(url, "4. Multi-Label Classification", {
        "image": image_b64,
        "categories": ["airplane", "car", "building", "ship"],
        "task": "multi_label_cls",
        "parameters": {
            "tau_cls": 0.3,
            "lambda_pool": 0.5
        }
    })
    
    # Test 5: Image Classification
    results["image_cls"] = test_task(url, "5. Image Classification (Scene)", {
        "image": image_b64,
        "categories": ["airport", "parking lot", "stadium", "harbor"],
        "task": "image_cls",
        "parameters": {
            "lambda_pool": 0.5
        }
    })
    
    # Test 6: Object Detection
    results["object_det"] = test_task(url, "6. Object Detection", {
        "image": image_b64,
        "categories": ["airplane", "vehicle"],
        "task": "object_det"
    })
    
    # Test 7: Object Counting
    results["object_counting"] = test_task(url, "7. Object Counting", {
        "image": image_b64,
        "prompt": "airplane",
        "task": "object_counting"
    })
    
    # Test 8: Image Captioning
    results["image_caption"] = test_task(url, "8. Image Captioning", {
        "image": image_b64,
        "categories": ["airplane", "vehicle", "building"],
        "task": "image_caption"
    })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)
    
    for task_name, (success, _) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {task_name}")
    
    print(f"\nTotal: {passed}/{total} tasks passed ({100*passed/total:.1f}%)")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
