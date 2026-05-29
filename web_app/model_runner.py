# web_app/model_runner.py
import os
import sys
import time
import re
import pickle
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, DistilBertTokenizer, BertForSequenceClassification
from typing import Dict, Any

MODELS_BASE = r"D:\Code\test_thuanphat_hocsau\main\DEEP_DATA\Main_project\CAREER_GUIDANCE_AI\models\custom_distilbert"

# Add custom_distilbert to path for modeling/configuration imports
if MODELS_BASE not in sys.path:
    sys.path.insert(0, MODELS_BASE)

from modeling_distilbert import DistilBertForCareerPath
from configuration_distilbert import OptimDistilBertConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_LOADED_MODELS = {}

def clean_text_for_dl(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9+#.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_for_ml(text):
    text = clean_text_for_dl(text).lower()
    text = re.sub(r'[^a-z0-9+#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_loaded_models():
    global _LOADED_MODELS
    if not _LOADED_MODELS:
        le_path = os.path.join(MODELS_BASE, "saved_models", "label_encoder.pkl")
        with open(le_path, "rb") as f:
            le = pickle.load(f)
        careers = le.classes_.tolist()
        num_classes = len(careers)
        
        # 1. ML Baseline
        ml_vec_path = os.path.join(MODELS_BASE, "saved_models", "ml_baseline", "vectorizer.pkl")
        ml_model_path = os.path.join(MODELS_BASE, "saved_models", "ml_baseline", "random_forest.pkl")
        with open(ml_vec_path, "rb") as f:
            ml_vec = pickle.load(f)
        with open(ml_model_path, "rb") as f:
            ml_model = pickle.load(f)
            
        # 2. BERT Base
        bert_path = os.path.join(MODELS_BASE, "saved_models", "bert_base", "bert_base.pt")
        tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_base = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
        bert_base.load_state_dict(torch.load(bert_path, map_location=DEVICE))
        bert_base.to(DEVICE).eval()
        
        # 3. Custom DistilBERT
        custom_path = os.path.join(MODELS_BASE, "saved_models", "custom_distilbert", "custom_model.pt")
        tokenizer_distil = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        config = OptimDistilBertConfig()
        config.num_labels = num_classes
        custom_model = DistilBertForCareerPath(config)
        custom_model.load_state_dict(torch.load(custom_path, map_location=DEVICE), strict=False)
        custom_model.to(DEVICE).eval()
        
        _LOADED_MODELS["careers"] = careers
        _LOADED_MODELS["ml_vec"] = ml_vec
        _LOADED_MODELS["ml_model"] = ml_model
        _LOADED_MODELS["bert_tokenizer"] = tokenizer_bert
        _LOADED_MODELS["bert_model"] = bert_base
        _LOADED_MODELS["distil_tokenizer"] = tokenizer_distil
        _LOADED_MODELS["custom_model"] = custom_model
        
    return _LOADED_MODELS

def run_model(model_name: str, input_text: str) -> Dict[str, Any]:
    """
    Load và chạy inference cho 1 model.
    Trả về: {"model": str, "prediction": str, "confidence": float, "inference_ms": float, "error": str|None}
    """
    start = time.perf_counter()
    try:
        models = get_loaded_models()
        careers = models["careers"]
        
        prediction = ""
        confidence = 0.0
        
        if model_name == "ml_baseline":
            ml_text = clean_text_for_ml(input_text)
            vec = models["ml_vec"].transform([ml_text])
            probs = models["ml_model"].predict_proba(vec)[0]
            pred_idx = probs.argmax()
            confidence = float(probs[pred_idx])
            prediction = careers[pred_idx]
            
        elif model_name == "bert_base":
            dl_text = clean_text_for_dl(input_text)
            inputs = models["bert_tokenizer"](
                dl_text, return_tensors="pt", truncation=True, padding=True, max_length=256
            ).to(DEVICE)
            with torch.no_grad():
                outputs = models["bert_model"](**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            confidence = float(probs[pred_idx].item())
            prediction = careers[pred_idx]
            
        elif model_name == "custom_distilbert":
            dl_text = clean_text_for_dl(input_text)
            inputs = models["distil_tokenizer"](
                dl_text, return_tensors="pt", truncation=True, padding=True, max_length=256
            ).to(DEVICE)
            with torch.no_grad():
                outputs = models["custom_model"](inputs['input_ids'], inputs['attention_mask'])
            probs = F.softmax(outputs["logits"], dim=-1)[0]
            pred_idx = probs.argmax().item()
            confidence = float(probs[pred_idx].item())
            prediction = careers[pred_idx]
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "model": model_name,
            "prediction": prediction,
            "confidence": confidence,
            "inference_ms": round(elapsed_ms, 2),
            "error": None
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "model": model_name,
            "prediction": "Error",
            "confidence": 0.0,
            "inference_ms": round(elapsed_ms, 2),
            "error": str(e)
        }

def run_all_models(input_text: str) -> list:
    model_names = ["custom_distilbert", "bert_base", "ml_baseline"]
    return [run_model(name, input_text) for name in model_names]
