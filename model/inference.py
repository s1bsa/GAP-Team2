import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from typing import List, Optional, Tuple
from clearml import InputModel, Task, Model
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import model.cnn as cnn_module


# ---------- Utilities ----------

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def discover_images(path: Path, recursive: bool) -> List[Path]:
    """Collect image files from a path."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    paths: List[Path] = []
    if path.is_file():
        if path.suffix.lower() in exts:
            paths.append(path)
    else:
        it = path.rglob("*") if recursive else path.glob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No images found at {path}")
    return sorted(paths)


def build_transform(img_size: int = 256,
                    mean: Tuple[float, float, float] = (0.4664, 0.4891, 0.4104),
                    std: Tuple[float, float, float] = (0.1761, 0.1500, 0.1925)) -> transforms.Compose:
    """Deterministic preprocessing consistent with training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------- Model loading ----------

_CANDIDATE_ARCHS = [
    ("CNN_11", getattr(cnn_module, "CNN_11", None), 256),
    ("Simple_ResNet3", getattr(cnn_module, "Simple_ResNet3", None), 256),
]

def grab_model_names():
    model_names = {}
    model_list = Model.query_models(project_name="Small Group Project - Team 2", only_published=True)
    for model in model_list:
        model_names[model.name] = model.id
    return model_names

def _try_load_arch(arch_ctor, weights, device, num_classes) -> Optional[torch.nn.Module]:
    """Instantiate an arch and try to load weights; return model or None on mismatch."""
    if arch_ctor is None:
        return None
    
    # Try to instantiate with num_classes
    try:
        model = arch_ctor(categories=num_classes).to(device)
    except TypeError:
        try:
            model = arch_ctor(num_classes=num_classes).to(device)
        except TypeError:
            try:
                model = arch_ctor().to(device)
            except Exception:
                return None
    
    try:
        model.load_state_dict(weights, strict=True)
        return model
    except Exception:
        return None


def load_model(weights_path: Path, arch: str = "auto", device: Optional[torch.device] = None, num_classes: int = None) -> Tuple[torch.nn.Module, int]:
    """
    Load a model. If arch='auto', try known CNN_* variants and select the one whose state_dict matches.
    Returns (model, expected_img_size).
    """
    device = device or get_device()
    weights = torch.load(str(weights_path), map_location=device)
    
    # Infer num_classes from weights if not provided
    if num_classes is None:
        num_classes = infer_class_count_from_state_dict(weights_path)
        if num_classes is None:
            num_classes = 38  # default fallback
    
    print(f"Loading model with {num_classes} output classes")

    if arch != "auto":
        # Instantiate the requested architecture
        if not hasattr(cnn_module, arch):
            raise ValueError(f"Unknown architecture '{arch}'. Available: {[n for n,_,_ in _CANDIDATE_ARCHS if getattr(cnn_module, n, None) is not None]}")
        ctor = getattr(cnn_module, arch)
        try:
            model = ctor(categories=num_classes).to(device)
        except TypeError:
            model = ctor().to(device)
        model.load_state_dict(weights)
        img_size = 256 if "256" in arch or arch in {"CNN_2", "CNN_3"} else 32
        model.eval()
        return model, img_size

    # Auto-detect architecture
    for name, ctor, img_size in _CANDIDATE_ARCHS:
        m = _try_load_arch(ctor, weights, device, num_classes)
        if m is not None:
            print(f"Successfully loaded as {name}")
            m.eval()
            return m, img_size

    # Fallback: raise with context
    raise RuntimeError(f"Could not match weights to a known architecture with {num_classes} classes.")


def infer_class_count_from_state_dict(weights_path: Path) -> Optional[int]:
    """Infer number of classes from the classifier's final weight shape if available."""
    sd = torch.load(str(weights_path), map_location="cpu")
    if isinstance(sd, dict):
        # Look for the last linear layer (output layer)
        last_linear_key = None
        for k in reversed(list(sd.keys())):
            if k.endswith(".weight") and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 2:
                last_linear_key = k
                break
        
        if last_linear_key:
            num_classes = sd[last_linear_key].shape[0]
            print(f"Detected {num_classes} classes from weights key: {last_linear_key}")
            return num_classes
    return None


def load_class_names_from_file(num_classes: int) -> List[str]:
    """Load all class names from class_names.txt file."""
    script_dir = Path(__file__).parent
    class_names_path = script_dir / "class_names.txt"
    
    if class_names_path.exists():
        with open(class_names_path, "r") as f:
            all_names = [line.strip() for line in f if line.strip()]
        
        # If model has fewer classes than the full list, it's likely a subset model
        if num_classes < len(all_names):
            print(f"Model has {num_classes} classes but class_names.txt has {len(all_names)} classes")
            print(f"This appears to be a subset model")
            # Return the full list - we'll handle mapping in the caller
            return all_names
        else:
            return all_names[:num_classes]
    
    # Fallback: generate generic names
    return [f"Class_{i}" for i in range(num_classes)]


# ---------- Prediction ----------

@torch.no_grad()
def predict_batch(model: torch.nn.Module,
                  images: List[Image.Image],
                  transform: transforms.Compose,
                  device: torch.device,
                  topk: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model on a list of PIL images. Returns (topk_indices, topk_probs)."""
    tensors = [transform(img) for img in images]
    batch = torch.stack(tensors, dim=0).to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    k = min(topk, probs.shape[1])
    prob_vals, idx = torch.topk(probs, k=k, dim=1)
    return idx.cpu(), prob_vals.cpu()


def predict_image(pil_image: Image.Image,
                  model: torch.nn.Module,
                  transform: transforms.Compose,
                  device: torch.device,
                  topk: int = 1):
    """Convenience API for UI integration."""
    idx, probs = predict_batch(model, [pil_image], transform, device, topk=topk)
    return idx[0], probs[0]


# ---------- CLI ----------

def main(model_name: str, image_path: str, architecture: str = 'auto', img_size: int = None):
    model_names = grab_model_names()
 
    try:
        input_model = InputModel(model_id=model_names[model_name])
    except Exception as e:
        print(f"Error loading model: {e}")
        return []
    
    # Get the local path to the model weights file from ClearML
    weights_path = input_model.get_local_copy()
    device = get_device()

    # Load model - use ClearML model if available, otherwise use args.weights
    model_path = Path(weights_path)
    
    # Infer class count for label mapping
    num_classes = infer_class_count_from_state_dict(model_path)
    if num_classes is None:
        num_classes = 38  # default fallback
    
    print(f"Model has {num_classes} output classes")
    
    # Load model with correct number of classes
    model, default_img_size = load_model(model_path, arch=architecture, device=device, num_classes=num_classes)
    
    # Build transform
    img_size = img_size or default_img_size
    transform = build_transform(img_size=img_size)

    # Load class names from ClearML metadata
    class_names = None
    class_name_mapping = None  # Maps model output indices to actual class names
        
    # Fallback: Load from class_names.txt
    all_class_names = load_class_names_from_file(num_classes)
    
    # Handle subset models (e.g., Tomatoes-only with 10 classes)
    if num_classes < len(all_class_names):
        # Check if this is a tomatoes-only model (last 10 classes)
        if num_classes == 10 and "Tomato" in all_class_names[-1]:
            class_names = all_class_names[-10:]  # Last 10 are tomatoes (indices 28-37)
            print(f"Detected Tomatoes-only model: using classes {28}-{37}")
            print(f"Class names: {class_names[0]} ... {class_names[-1]}")
        else:
            # Generic subset: use first num_classes
            class_names = all_class_names[:num_classes]
            print(f"Using first {num_classes} classes from class_names.txt")
    else:
        class_names = all_class_names
        print(f"Using all {len(class_names)} classes from class_names.txt")
    
    # Ensure we have the right number of class names
    if len(class_names) != num_classes:
        print(f"Warning: Mismatch between model classes ({num_classes}) and loaded names ({len(class_names)})")
        # Pad or truncate as needed
        if len(class_names) < num_classes:
            class_names.extend([f"Class_{i}" for i in range(len(class_names), num_classes)])
        else:
            class_names = class_names[:num_classes]
    
    print(f"Final class mapping: {len(class_names)} classes")
    print(f"  First: {class_names[0]}")
    print(f"  Last:  {class_names[-1]}")
    
    # Collect images
    img_paths = discover_images(Path(image_path), recursive=False)
    
    # Run in reasonably sized mini-batches for speed
    B = 32
    results = []
    batch_imgs = []
    batch_paths = []
    
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
        batch_imgs.append(img)
        batch_paths.append(p)
        
        if len(batch_imgs) == B:
            idx, probs = predict_batch(model, batch_imgs, transform, device, topk=5)
            for _, idx_i, probs_i in zip(batch_paths, idx, probs):
                # Map model outputs to class names
                labels = []
                for j in idx_i.tolist():
                    if 0 <= j < len(class_names):
                        labels.append(class_names[j])
                    else:
                        labels.append(f"Unknown_Class_{j}")
                
                results.append({
                    "top5_labels": labels,
                    "top5_probs": [float(x) for x in probs_i.tolist()],
                })
            batch_imgs.clear()
            batch_paths.clear()

    # Flush tail
    if batch_imgs:
        idx, probs = predict_batch(model, batch_imgs, transform, device, topk=5)
        for _, idx_i, probs_i in zip(batch_paths, idx, probs):
            # Map model outputs to class names
            labels = []
            for j in idx_i.tolist():
                if 0 <= j < len(class_names):
                    labels.append(class_names[j])
                else:
                    labels.append(f"Unknown_Class_{j}")
            
            results.append({
                "top5_labels": labels,
                "top5_probs": [float(x) for x in probs_i.tolist()],
            })
    
    return results