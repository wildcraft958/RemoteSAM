#!/usr/bin/env python3
"""
Exhaustive Test Suite for RemoteSAM Modal Deployment
Tests various scenarios, tasks, and edge cases.
"""

import requests
import base64
import json
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class RemoteSAMTester:
    def __init__(self, endpoint_url: str, test_images_dir: str = "assets"):
        self.endpoint_url = endpoint_url
        self.test_images_dir = Path(test_images_dir)
        self.results = {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "errors": [],
            "timings": []
        }
        
    def log_success(self, message: str):
        print(f"{Colors.GREEN}✓{Colors.RESET} {message}")
        
    def log_fail(self, message: str):
        print(f"{Colors.RED}✗{Colors.RESET} {message}")
        
    def log_info(self, message: str):
        print(f"{Colors.BLUE}ℹ{Colors.RESET} {message}")
        
    def log_warning(self, message: str):
        print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def make_request(self, image_b64: str, prompt: str, task: str = "referring_seg") -> Tuple[bool, Optional[Dict], float]:
        """Make API request and return success status, result, and duration"""
        start_time = time.time()
        
        try:
            payload = {
                "image": image_b64,
                "prompt": prompt,
                "task": task
            }
            
            response = requests.post(self.endpoint_url, json=payload, timeout=60)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return True, result, duration
            else:
                return False, {"error": f"HTTP {response.status_code}: {response.text}"}, duration
                
        except Exception as e:
            duration = time.time() - start_time
            return False, {"error": str(e)}, duration
    
    def validate_mask(self, mask: List[List[int]], expected_nonzero: bool = True) -> Tuple[bool, str]:
        """Validate mask structure and content"""
        try:
            mask_array = np.array(mask)
            
            # Check shape
            if len(mask_array.shape) != 2:
                return False, f"Invalid mask shape: {mask_array.shape}, expected 2D array"
            
            # Check values
            unique_values = np.unique(mask_array)
            if not all(v in [0, 1] for v in unique_values):
                return False, f"Invalid mask values: {unique_values}, expected only 0 and 1"
            
            # Check for non-zero pixels
            nonzero_count = np.count_nonzero(mask_array)
            if expected_nonzero and nonzero_count == 0:
                return False, "Mask is all zeros (no segmentation detected)"
            
            total_pixels = mask_array.size
            percentage = 100.0 * nonzero_count / total_pixels if total_pixels > 0 else 0
            
            return True, f"Valid mask: {mask_array.shape}, {nonzero_count} pixels ({percentage:.2f}%)"
            
        except Exception as e:
            return False, f"Mask validation error: {str(e)}"
    
    def validate_bbox(self, bbox: List[float]) -> Tuple[bool, str]:
        """Validate bounding box format"""
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False, f"Invalid bbox format: {bbox}, expected [xmin, ymin, xmax, ymax]"
        
        xmin, ymin, xmax, ymax = bbox
        if xmin >= xmax or ymin >= ymax:
            return False, f"Invalid bbox coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}"
        
        return True, f"Valid bbox: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]"
    
    def run_test(self, test_name: str, image_path: str, prompt: str, task: str = "referring_seg", 
                 expect_success: bool = True, validate_result: bool = True):
        """Run a single test case"""
        self.results["total"] += 1
        test_num = self.results["total"]
        
        print(f"\n{Colors.BOLD}Test #{test_num}: {test_name}{Colors.RESET}")
        print(f"  Image: {image_path}")
        print(f"  Prompt: '{prompt}'")
        print(f"  Task: {task}")
        
        # Encode image
        try:
            img_path = self.test_images_dir / image_path
            if not img_path.exists():
                self.log_fail(f"Image not found: {img_path}")
                self.results["failed"] += 1
                self.results["errors"].append(f"Test #{test_num}: Image not found")
                return
            
            image_b64 = self.encode_image(img_path)
        except Exception as e:
            self.log_fail(f"Failed to encode image: {e}")
            self.results["failed"] += 1
            self.results["errors"].append(f"Test #{test_num}: Image encoding failed")
            return
        
        # Make request
        success, result, duration = self.make_request(image_b64, prompt, task)
        self.results["timings"].append(duration)
        
        # Check success expectation
        if success != expect_success:
            if expect_success:
                self.log_fail(f"Request failed (expected success): {result.get('error', 'Unknown error')}")
                self.results["failed"] += 1
                self.results["errors"].append(f"Test #{test_num}: {test_name} - Request failed")
                return
            else:
                self.log_warning(f"Request succeeded (expected failure)")
        
        if not success:
            self.log_info(f"Request failed as expected: {result.get('error', 'Unknown')}")
            self.results["passed"] += 1
            return
        
        # Validate result
        if validate_result and 'result' in result:
            if task == "referring_seg":
                is_valid, msg = self.validate_mask(result['result'])
                if is_valid:
                    self.log_success(f"Request successful ({duration:.2f}s) - {msg}")
                    self.results["passed"] += 1
                else:
                    self.log_fail(f"Invalid result: {msg}")
                    self.results["failed"] += 1
                    self.results["errors"].append(f"Test #{test_num}: {test_name} - {msg}")
            
            elif task == "visual_grounding":
                bbox = result['result']
                is_valid, msg = self.validate_bbox(bbox)
                if is_valid:
                    self.log_success(f"Request successful ({duration:.2f}s) - {msg}")
                    self.results["passed"] += 1
                else:
                    self.log_fail(f"Invalid result: {msg}")
                    self.results["failed"] += 1
                    self.results["errors"].append(f"Test #{test_num}: {test_name} - {msg}")
            
            elif task == "semantic_seg":
                self.log_success(f"Request successful ({duration:.2f}s)")
                self.results["passed"] += 1
        else:
            self.log_success(f"Request successful ({duration:.2f}s)")
            self.results["passed"] += 1
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}RemoteSAM Deployment Test Suite{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"Endpoint: {self.endpoint_url}")
        print(f"Test Images Directory: {self.test_images_dir}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test Category 1: Referring Segmentation Tests
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Category 1: Referring Segmentation Tests{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        self.run_test(
            "Basic airplane segmentation",
            "demo.jpg",
            "the airplane on the right",
            "referring_seg"
        )
        
        self.run_test(
            "Alternative airplane reference",
            "demo.jpg",
            "the plane",
            "referring_seg"
        )
        
        self.run_test(
            "Specific object attribute",
            "demo.jpg",
            "the white airplane",
            "referring_seg"
        )
        
        # Test Category 2: Visual Grounding Tests
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Category 2: Visual Grounding Tests{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        self.run_test(
            "Bounding box detection - airplane",
            "demo.jpg",
            "the airplane on the right",
            "visual_grounding"
        )
        
        # Test Category 3: Edge Cases and Error Handling
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Category 3: Edge Cases{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        self.run_test(
            "Empty prompt",
            "demo.jpg",
            "",
            "referring_seg",
            validate_result=False
        )
        
        self.run_test(
            "Very long prompt",
            "demo.jpg",
            "the extremely detailed and very specific airplane that is located on the right side of the image with white color and multiple features",
            "referring_seg"
        )
        
        self.run_test(
            "Non-existent object",
            "demo.jpg",
            "the purple elephant",
            "referring_seg"
        )
        
        self.run_test(
            "Ambiguous reference",
            "demo.jpg",
            "the thing",
            "referring_seg"
        )
        
        # Test Category 4: Performance Tests
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Category 4: Performance & Stress Tests{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        # Multiple rapid requests
        for i in range(3):
            self.run_test(
                f"Rapid request #{i+1}",
                "demo.jpg",
                "the airplane",
                "referring_seg"
            )
        
        # Print Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}Test Summary{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        total = self.results["total"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Tests: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.results["timings"]:
            timings = self.results["timings"]
            print(f"\n{Colors.BOLD}Performance Metrics:{Colors.RESET}")
            print(f"  Average Response Time: {np.mean(timings):.2f}s")
            print(f"  Min Response Time: {np.min(timings):.2f}s")
            print(f"  Max Response Time: {np.max(timings):.2f}s")
            print(f"  Median Response Time: {np.median(timings):.2f}s")
        
        if self.results["errors"]:
            print(f"\n{Colors.RED}{Colors.BOLD}Failed Tests:{Colors.RESET}")
            for error in self.results["errors"]:
                print(f"  {Colors.RED}•{Colors.RESET} {error}")
        
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}\n")
        
        # Return exit code based on results
        return 0 if failed == 0 else 1

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_exhaustive.py <endpoint_url> [test_images_dir]")
        print("\nExample:")
        print("  python test_exhaustive.py https://your-app.modal.run assets/")
        sys.exit(1)
    
    endpoint_url = sys.argv[1]
    test_images_dir = sys.argv[2] if len(sys.argv) > 2 else "assets"
    
    tester = RemoteSAMTester(endpoint_url, test_images_dir)
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
