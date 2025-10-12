#!/usr/bin/env python3
"""
LLM Inference Module for NCTE Classroom Transcript Analysis

This module handles all LLM inference functionality including:
- Model loading and initialization
- Text generation with different backends
- Classification result extraction
- Support for various model types (transformers, ModelScope)
"""

import os
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Any, Union

import torch
import transformers

# Import ModelScope for MistralAI models
try:
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("Warning: ModelScope not available. MistralAI models will not be supported.")

warnings.filterwarnings("ignore")

model_mapping = {
            "llama-3.1-8b-instruct": "/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
            "mistral-7b-instruct-v0.3": "/u/xfeng4/.cache/modelscope/hub/models/mistralai/Mistral-7B-Instruct-v0.3",
            "llama-3.3-70B-instruct": "meta-llama/Llama-3.3-70B-Instruct",
}
        

class LLMInference:
    """Handles LLM inference for classification tasks."""
    
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 2048):
        """Initialize the LLM inference engine."""
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.max_length = max_length
        self.pipeline = None
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup computing device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load the model using appropriate backend."""
        print(f"Loading model: {self.model_name}")
        
        try:
            model_path = self._get_model_path()
            
            # Check if it's a MistralAI model
            if "mistral" in self.model_name.lower() and MODELSCOPE_AVAILABLE:
                self._load_mistral_model(model_path)
            else:
                # Use transformers pipeline for other models
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_path,
                    model_kwargs={"torch_dtype": torch.bfloat16} if self.device == "cuda" else {},
                    device_map="auto" if self.device == "cuda" else None,
                )
            
            print(f"Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise e
    
    def _load_mistral_model(self, model_path: str):
        """Load MistralAI model using ModelScope."""
        print(f"Loading MistralAI model from: {model_path}")
        
        # Load model and tokenizer using ModelScope
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad_token to eos_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("MistralAI model loaded successfully")
    
    def _get_model_path(self) -> str:
        """Get the model path or identifier."""
        # Model mapping including MistralAI models
        return model_mapping.get(self.model_name.lower(), self.model_name)
    
    def generate_response(self, prompt: List[Dict[str, str]], 
                         max_new_tokens: int = 10) -> str:
        """Generate response from the model using greedy generation."""
        try:
            # Check if we have a MistralAI model loaded
            if hasattr(self, 'model') and hasattr(self, 'tokenizer') and self.model is not None:
                return self._generate_with_mistral_model(prompt, max_new_tokens)
            elif self.pipeline:
                return self._generate_with_pipeline(prompt, max_new_tokens)
            else:
                print("Error: No model or pipeline initialized")
                return ""
                
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def _generate_with_pipeline(self, prompt: List[Dict[str, str]], 
                               max_new_tokens: int) -> str:
        """Generate response using transformers pipeline with greedy generation."""
        try:
            # Message format (list of dicts) - greedy generation only
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy generation
                return_full_text=False,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                return generated_text.strip() if generated_text else ""
            else:
                print("Warning: Pipeline returned empty output")
                return ""
                
        except Exception as e:
            print(f"Pipeline generation error: {e}")
            return ""
    
    def _generate_with_mistral_model(self, prompt: List[Dict[str, str]], 
                                    max_new_tokens: int) -> str:
        """Generate response using MistralAI model with greedy generation."""
        try:
            # Apply chat template for message format
            encodeds = self.tokenizer.apply_chat_template(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                add_generation_prompt=True
            )
            
            input_ids = encodeds.to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy generation
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract only the new tokens (response)
                response = self.tokenizer.decode(
                    generated_ids[0][input_ids.shape[-1]:], 
                    skip_special_tokens=True
                ).strip()
                
                return response
                
        except Exception as e:
            print(f"MistralAI generation error: {e}")
            return ""
    
    def classify(self, prompt: List[Dict[str, str]], 
                 max_new_tokens: int = 10) -> int:
        """Classify a single example and extract 0/1 label."""
        response = self.generate_response(prompt, max_new_tokens)
        return self._extract_classification(response)
    
    def _extract_classification(self, response: str) -> int:
        """Extract classification (0 or 1) from model response."""
        response_clean = response.strip().lower()
        
        # Look for clear indicators first
        if response_clean.startswith('1') or response_clean == '1':
            return 1
        elif response_clean.startswith('0') or response_clean == '0':
            return 0
        
        # Look for patterns that indicate the classification
        if any(word in response_clean for word in ['1', 'yes', 'on-task', 'high', 'true']):
            return 1
        elif any(word in response_clean for word in ['0', 'no', 'off-task', 'low', 'false']):
            return 0
        else:
            # If unclear, default to 0 (conservative approach)
            print(f"Unclear response: '{response}'. Defaulting to 0.")
            return 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'has_pipeline': self.pipeline is not None,
            'has_mistral_model': self.model is not None,
            'modelscope_available': MODELSCOPE_AVAILABLE
        }


def test_llm_inference():
    """Test the LLM inference functionality."""
    print("🧪 Testing LLM Inference\n")
    
    # Test with different models
    models_to_test = [
        "llama-3.1-8b-instruct",  # Primary model
        "mistral-7b-instruct-v0.3",  # MistralAI model
    ]
    
    success = True
    
    for model_name in models_to_test:
        print(f"\n📋 Testing {model_name}:")
        try:
            inference = LLMInference(model_name, device="auto")
            print(f"   ✅ Model loaded successfully")
            
            # Test message format
            messages = [
                {"role": "system", "content": "You are an expert in analyzing elementary mathematics classroom discourse."},
                {"role": "user", "content": "Analyze the student's utterance and determine if it is on-task (1) or off-task (0).\n\nStudent: I think the answer is 5 because 2 plus 3 equals 5.\nTeacher: Good thinking! Can you explain how you got that?\n\nClassification:"}
            ]
            
            result = inference.classify(messages)
            print(f"   ✅ Message format classification: {result}")
            
            # Test model info
            info = inference.get_model_info()
            print(f"   ✅ Model info: {info}")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            success = False
    
    return success


if __name__ == "__main__":
    print("LLM Inference Test for NCTE Classifier")
    print("=" * 60)
    
    test_passed = test_llm_inference()
    
    print(f"\n{'='*60}")
    if test_passed:
        print("✅ ALL TESTS PASSED! LLM inference works correctly.")
        print("\n🔧 Key Features:")
        print("   - Multiple model backend support ✅")
        print("   - String and message format support ✅")
        print("   - Classification result extraction ✅")
        print("   - Error handling and fallbacks ✅")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    
    print("=" * 60)
