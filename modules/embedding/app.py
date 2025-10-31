# embedding.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import collections

class FaceEmbedder:
    def __init__(self, 
                 model_name='ir_101', 
                 ckpt_path='./modules/embedding/pretrained/adaface_ir101_webface12m.ckpt',
                 device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Build model
        from modules.embedding.model import build_model
        self.model = build_model(model_name)
        
        # Load checkpoint with error handling
        self._load_checkpoint(ckpt_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        
        print("✅ AdaFace model loaded successfully")

    def _load_checkpoint(self, ckpt_path):
        """Load checkpoint with various fallback strategies"""
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            print("Checkpoint keys:", list(checkpoint.keys()))
            
            # Try different possible state dict keys
            state_dict = None
            for key in ['state_dict', 'model_state_dict', 'model', 'network']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"Found state_dict in key: '{key}'")
                    break
            
            if state_dict is None:
                state_dict = checkpoint
                print("Using checkpoint as state_dict directly")
            
            # Clean prefixes from checkpoint keys
            clean_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[len('module.'):]
                if k.startswith('model.'):
                    k = k[len('model.'):]
                clean_state_dict[k] = v

            # Add 'backbone.' prefix to match model structure
            final_state_dict = collections.OrderedDict()
            for k, v in clean_state_dict.items():
                final_state_dict[f'backbone.{k}'] = v

            missing_keys, unexpected_keys = self.model.load_state_dict(final_state_dict, strict=False)

            
            # Debug: Print model and checkpoint structure
            print("\n🔍 Model structure vs Checkpoint structure:")
            model_keys = list(self.model.state_dict().keys())
            checkpoint_keys = list(final_state_dict.keys())
            
            print(f"Model has {len(model_keys)} parameters")
            print(f"Checkpoint has {len(checkpoint_keys)} parameters")
            
            print("\nFirst 5 model keys:", model_keys[:5])
            print("First 5 checkpoint keys:", checkpoint_keys[:5])
            
            # Try loading with strict=False first
            missing_keys, unexpected_keys = self.model.load_state_dict(final_state_dict, strict=False)

            
            print(f"\n🔧 Loading results:")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print("First 5 missing keys:", missing_keys[:5])
            if unexpected_keys:
                print("First 5 unexpected keys:", unexpected_keys[:5])
                
            # If there are issues with fc layer, try to handle them
            if any('fc' in key for key in missing_keys) or any('fc' in key for key in unexpected_keys):
                print("\n🔄 Handling fc layer mismatch...")
                self._handle_fc_mismatch(final_state_dict)
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise

    def _handle_fc_mismatch(self, state_dict):
        """Handle fc layer dimension mismatches"""
        model_dict = self.model.state_dict()
        
        # Filter out incompatible keys
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                else:
                    print(f"Shape mismatch for {k}: model {model_dict[k].shape} vs checkpoint {v.shape}")
                    # Special handling for fc layer
                    if k == 'fc.weight':
                        # If fc weight dimensions don't match, we need to adapt
                        model_in_features = model_dict['fc.weight'].shape[1]
                        checkpoint_in_features = v.shape[1]
                        
                        if model_in_features != checkpoint_in_features:
                            print(f"Adapting fc layer: model expects {model_in_features}, checkpoint has {checkpoint_in_features}")
                            # This usually means the checkpoint was trained with a different input size
                            # We'll skip loading fc weights in this case
                            continue
            else:
                print(f"Key not in model: {k}")
        
        # Load compatible parameters
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print(f"✅ Loaded {len(pretrained_dict)}/{len(state_dict)} parameters")

    def preprocess_face(self, face):
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_rgb)
        return self.transform(face_img).unsqueeze(0).to(self.device)

    def get_embeddings(self, enhanced_faces):
        embeddings = []
        for face in enhanced_faces:
            try:
                img_tensor = self.preprocess_face(face)
                with torch.no_grad():
                    emb = self.model(img_tensor)
                    emb = F.normalize(emb, p=2, dim=1)
                embeddings.append(emb.cpu().numpy().flatten())
            except Exception as e:
                print(f"⚠️ Error in embedding: {e}")
                embeddings.append(np.zeros(512))
        return np.array(embeddings)

    def get_embedding_size(self):
        """Get the embedding dimension"""
        return 512