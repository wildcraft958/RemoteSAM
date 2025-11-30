import modal
import sys
import os
import shutil

# Define the image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("libgl1", "libglib2.0-0", "git", "build-essential")
    .run_commands(
        "echo 'torch==1.13.0+cu116' > constraints.txt",
        "echo 'torchvision==0.14.0+cu116' >> constraints.txt",
        "echo 'torchaudio==0.13.0' >> constraints.txt",
        "python -m pip install --upgrade pip",
        "python -m pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116",
        "python -m pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html",
        "python -m pip install -c constraints.txt mmsegmentation==0.17.0 transformers==4.30.2 timm==0.6.11 opencv-python huggingface_hub filelock ftfy regex h5py matplotlib==3.6.1 scikit-image==0.19.3 scipy==1.9.2 tokenizers==0.13.1 yacs einops termcolor pycocotools Pillow scikit-learn pytorch-ignite accelerate shapely torchmetrics yapf==0.40.1 addict fastapi",
    )
    # Add the local directory to the image
    .add_local_dir("/home/bakasur/RemoteSAM", remote_path="/root/RemoteSAM")
)

app = modal.App("remotesam-inference", image=image)

# Define volumes to store the model weights
remotesam_vol = modal.Volume.from_name("remotesam-weights", create_if_missing=True)
bert_vol = modal.Volume.from_name("bert-weights", create_if_missing=True)

@app.cls(
    gpu="A10G",
    scaledown_window=300,
    volumes={
        "/root/pretrained_weights": remotesam_vol,
        "/root/bert_weights": bert_vol,
    },
    timeout=600,
)
class RemoteSAMService:
    @modal.enter()
    def enter(self):
        import traceback
        import warnings
        
        # Suppress deprecation warnings from legacy BERT implementation
        warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*resume_download.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*MMCV.*')
        
        try:
            print("=== Starting RemoteSAM initialization ===")
            # Set working directory to the root of the repo
            os.chdir("/root/RemoteSAM")
            sys.path.append("/root/RemoteSAM")
            print("✓ Working directory set to /root/RemoteSAM")

            import torch
            print(f"✓ PyTorch version: {torch.__version__}")
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"✓ CUDA version: {torch.version.cuda}")

            # Download weights if not present
            from huggingface_hub import hf_hub_download, snapshot_download
            
            weights_path = "/root/pretrained_weights/RemoteSAMv1.pth"
            if not os.path.exists(weights_path):
                print("Downloading RemoteSAM weights...")
                hf_hub_download(
                    repo_id="1e12Leon/RemoteSAM",
                    filename="RemoteSAMv1.pth",
                    local_dir="/root/pretrained_weights",
                    local_dir_use_symlinks=False,
                )
                print("✓ RemoteSAM weights downloaded")
            else:
                print(f"✓ RemoteSAM weights found at {weights_path}")
            
            # Download BERT weights
            bert_path = "/root/bert_weights"
            if not os.path.exists(os.path.join(bert_path, "config.json")):
                print("Downloading BERT weights...")
                snapshot_download(
                    repo_id="bert-base-uncased",
                    local_dir=bert_path,
                    ignore_patterns=["*.msgpack", "*.h5", "*.tflite"],
                )
                bert_vol.commit()
                print("✓ BERT weights downloaded and committed to volume")
            else:
                print(f"✓ BERT weights found at {bert_path}")
                
            # Verify BERT files
            bert_files = os.listdir(bert_path)
            print(f"✓ BERT directory contains: {bert_files}")
            
            # Initialize model
            print("Importing RemoteSAM modules...")
            from tasks.code.model import RemoteSAM, init_demo_model
            print("✓ RemoteSAM modules imported")
            
            device = "cuda"
            checkpoint = weights_path
            
            # Patch sys.argv to avoid argparse errors and pass local BERT path
            print(f"Setting BERT path via sys.argv: {bert_path}")
            sys.argv = ["modal_app.py", "--ck_bert", bert_path]
            print(f"✓ sys.argv set to: {sys.argv}")
            
            print("Calling init_demo_model...")
            print(f"Checkpoint path: {checkpoint}")
            print(f"Checkpoint exists: {os.path.exists(checkpoint)}")
            
            # Load checkpoint to inspect
            checkpoint_data = torch.load(checkpoint, map_location='cpu')
            print(f"✓ Checkpoint loaded, keys: {list(checkpoint_data.keys())}")
            if 'model' in checkpoint_data:
                model_state_keys = list(checkpoint_data['model'].keys())[:10]
                print(f"✓ Model state dict contains {len(checkpoint_data['model'])} keys, first 10: {model_state_keys}")
            
            # Initialize model with proper weight loading
            from tasks.code.model import segmentation
            from args import get_parser
            
            args = get_parser().parse_args()
            args.device = device
            args.window12 = True
            
            print("Creating model architecture...")
            model = segmentation.__dict__["lavt_one"](pretrained='', args=args)
            print("✓ Model architecture created")
            
            print("Loading checkpoint weights...")
            state_dict = checkpoint_data['model']
            
            # Load with error handling
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"✓ Weights loaded - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            if missing_keys:
                print(f"  Missing keys (first 10): {missing_keys[:10]}")
            if unexpected_keys:
                print(f"  Unexpected keys (first 10): {unexpected_keys[:10]}")
            
            model = model.to(device)
            model.eval()  # Set to evaluation mode
            print("✓ Model moved to device and set to eval mode")
            
            self.model_raw = model
            print("✓ init_demo_model completed successfully")
            
            print("Wrapping model with RemoteSAM...")
            self.model = RemoteSAM(self.model_raw, device, use_EPOC=False)
            print("✓ RemoteSAM wrapper created")
            
            print("=== Model initialization completed successfully ===")
        except Exception as e:
            print(f"!!! Error during initialization: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            raise e

    @modal.method()
    def infer(self, image_bytes, prompt, task="referring_seg", categories=None, parameters=None):
        import cv2
        import numpy as np
        from PIL import Image
        import io
        import traceback

        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Parse parameters
            params = parameters or {}
            tau_cls = params.get("tau_cls", 0.5)
            lambda_pool = params.get("lambda_pool", 0.5)
            
            print(f"Running task: {task}")
            
            if task == "referring_seg":
                mask = self.model.referring_seg(image=pil_image, sentence=prompt)
                return {"mask": mask.tolist()}
                
            elif task == "visual_grounding":
                box = self.model.visual_grounding(image=pil_image, sentence=prompt)
                if box is None:
                    return {"bbox": None, "message": "No objects detected"}
                # Convert numpy types to Python native types
                return {"bbox": [float(x) for x in box]}
                
            elif task == "semantic_seg":
                classnames = categories if categories else [p.strip() for p in prompt.split(",")]
                masks = self.model.semantic_seg(image=pil_image, classnames=classnames)
                return {"masks": {k: v.tolist() for k, v in masks.items()}}
                
            elif task == "multi_label_cls":
                classnames = categories if categories else [p.strip() for p in prompt.split(",")]
                result = self.multi_label_classification(pil_image, classnames, tau_cls, lambda_pool)
                return result
                
            elif task == "image_cls":
                classnames = categories if categories else [p.strip() for p in prompt.split(",")]
                result = self.image_classification(pil_image, classnames, lambda_pool)
                return result
                
            elif task == "object_det":
                classnames = categories if categories else [p.strip() for p in prompt.split(",")]
                result = self.object_detection(pil_image, classnames)
                return result
                
            elif task == "object_counting":
                target_class = prompt if prompt else (categories[0] if categories else None)
                if not target_class:
                    raise ValueError("Target class required for object counting")
                result = self.object_counting(pil_image, target_class)
                return result
                
            elif task == "image_caption":
                classnames = categories if categories else [p.strip() for p in prompt.split(",")] if prompt else []
                result = self.image_caption(pil_image, classnames)
                return result
                
            else:
                raise ValueError(f"Unknown task: {task}. Supported: referring_seg, visual_grounding, semantic_seg, multi_label_cls, image_cls, object_det, object_counting, image_caption")
        except Exception as e:
            print("Error during inference:")
            traceback.print_exc()
            raise e
    
    def multi_label_classification(self, image, categories, tau_cls=0.5, lambda_pool=0.5):
        """Multi-label classification via semantic segmentation (Paper Eq. 3)"""
        import numpy as np
        
        results = {}
        # Get probability maps for all categories
        masks, probs = self.model.semantic_seg(image=image, classnames=categories, return_prob=True)
        
        for category in categories:
            prob_map = probs[category]
            
            # Compute confidence via pooling (Equation 3)
            avg_pool = np.mean(prob_map)
            max_pool = np.max(prob_map)
            confidence = lambda_pool * avg_pool + (1 - lambda_pool) * max_pool
            
            results[category] = {
                "score": float(confidence),
                "present": bool(confidence >= tau_cls)
            }
        
        return {"predictions": results}
    
    def image_classification(self, image, categories, lambda_pool=0.5):
        """Image classification - argmax of multi-label scores (Paper Eq. 5)"""
        multi_label_result = self.multi_label_classification(image, categories, tau_cls=0.0, lambda_pool=lambda_pool)
        
        scores = {cat: pred["score"] for cat, pred in multi_label_result["predictions"].items()}
        best_class = max(scores, key=scores.get)
        
        return {
            "predicted_class": best_class,
            "confidence": scores[best_class],
            "all_scores": scores
        }
    
    def object_detection(self, image, categories):
        """Object detection via semantic segmentation + M2B + EPOC (Paper Section 4.2)"""
        import numpy as np
        from utils import M2B
        
        all_detections = []
        
        # Get masks and probabilities for all categories
        masks, probs = self.model.semantic_seg(image=image, classnames=categories, return_prob=True)
        
        for category in categories:
            mask = masks[category]
            prob = probs[category]
            
            if np.sum(mask) == 0:
                continue  # Skip if no pixels detected
            
            # Use M2B to convert mask to bboxes (EPOC disabled for performance)
            boxes = M2B(mask, prob, box_type='hbb')
            
            for box in boxes:
                # Convert numpy types to native Python types for JSON serialization
                all_detections.append({
                    "class": category,
                    "bbox": [float(x) for x in box],  # Convert to native Python floats
                    "confidence": 1.0
                })
        
        return {"detections": all_detections, "count": len(all_detections)}
    
    def object_counting(self, image, target_class):
        """Object counting via detection (Paper Eq. 6)"""
        detection_result = self.object_detection(image, [target_class])
        
        count = sum(1 for d in detection_result["detections"] if d["class"] == target_class)
        
        return {
            "class": target_class,
            "count": count,
            "detections": detection_result["detections"]
        }
    
    def image_caption(self, image, categories):
        """Rule-based image caption generation (Paper Section 4.3)"""
        # Get detection results
        detection_result = self.object_detection(image, categories)
        detections = detection_result["detections"]
        
        if not detections:
            return {"caption": "No objects detected in the image."}
        
        # Count objects by category
        counts = {}
        for det in detections:
            counts[det["class"]] = counts.get(det["class"], 0) + 1
        
        # Generate caption
        caption_parts = []
        caption_parts.append(f"This image contains {len(detections)} objects.")
        
        for category, count in counts.items():
            if count == 1:
                caption_parts.append(f"There is 1 {category}.")
            else:
                caption_parts.append(f"There are {count} {category}s.")
        
        caption = " ".join(caption_parts)
        
        return {
            "caption": caption,
            "object_counts": counts,
            "total_objects": len(detections)
        }

    @modal.fastapi_endpoint(method="POST")
    def web_infer(self, item: dict):
        import base64
        
        image_b64 = item.get("image")
        prompt = item.get("prompt", "")
        task = item.get("task", "referring_seg")
        categories = item.get("categories")
        parameters = item.get("parameters")
        
        image_bytes = base64.b64decode(image_b64)
        
        result = self.infer.local(image_bytes, prompt, task, categories, parameters)
        
        return result

