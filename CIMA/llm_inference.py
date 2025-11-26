"""
Real LLM inference implementations
Supports OpenAI API, OpenRouter API, and local HuggingFace models
"""
import os
from typing import List, Dict, Optional
import torch
import config


class LLMInference:
    """Unified LLM inference class supporting multiple backends"""
    
    def __init__(self):
        self.mistral_model = None
        self.mistral_tokenizer = None
        
    def _load_mistral_local(self):
        """Load Mistral-7B model locally"""
        if self.mistral_model is not None:
            return
        
        print("Loading Mistral-7B model locally...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_config = config.MODEL_CONFIGS["mistral-7b"]
        model_name = config.MISTRAL_LOCAL_PATH or model_config["name"]
        
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optional quantization
        if model_config.get("load_in_8bit", False):
            self.mistral_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
            )
        else:
            self.mistral_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        
        self.mistral_model.eval()
        print(f"Mistral-7B loaded on device: {self.mistral_model.device}")
    
    def _infer_openai(self, messages: List[Dict], model_name: str = "gpt-4") -> str:
        """
        Inference using OpenAI API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: OpenAI model name
        
        Returns:
            Generated text
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Set it in config.py or as environment variable")
        
        openai.api_key = config.OPENAI_API_KEY
        
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=config.MODEL_CONFIGS["gpt-4"]["temperature"],
            max_tokens=config.MODEL_CONFIGS["gpt-4"]["max_tokens"],
        )
        
        return response.choices[0].message.content.strip()
    
    def _infer_openrouter(self, messages: List[Dict], model_name: str) -> str:
        """
        Inference using OpenRouter API (for Llama-3.1-405B)
        
        Args:
            messages: List of message dicts
            model_name: Model identifier
        
        Returns:
            Generated text
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        # Configure OpenAI client for OpenRouter
        openai.api_base = config.MODEL_CONFIGS["llama-3.1-405b"]["api_base"]
        openai.api_key = config.OPENROUTER_API_KEY
        
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=config.MODEL_CONFIGS["llama-3.1-405b"]["temperature"],
            max_tokens=config.MODEL_CONFIGS["llama-3.1-405b"]["max_tokens"],
        )
        
        return response.choices[0].message.content.strip()
    
    def _infer_mistral_local(self, messages: List[Dict]) -> str:
        """
        Inference using local Mistral-7B model
        
        Args:
            messages: List of message dicts
        
        Returns:
            Generated text
        """
        self._load_mistral_local()
        
        # Format messages for Mistral
        # Mistral uses [INST] tags
        formatted_prompt = self._format_mistral_prompt(messages)
        
        inputs = self.mistral_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.mistral_model.device)
        
        with torch.no_grad():
            outputs = self.mistral_model.generate(
                **inputs,
                max_new_tokens=config.MODEL_CONFIGS["mistral-7b"]["max_tokens"],
                temperature=config.MODEL_CONFIGS["mistral-7b"]["temperature"],
                do_sample=False,
                pad_token_id=self.mistral_tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_text = self.mistral_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _format_mistral_prompt(self, messages: List[Dict]) -> str:
        """
        Format messages for Mistral instruction format
        
        Args:
            messages: List of message dicts
        
        Returns:
            Formatted prompt string
        """
        # Mistral format: <s>[INST] instruction [/INST] response</s>
        formatted = ""
        
        system_msg = ""
        user_msg = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_msg = msg["content"]
        
        if system_msg:
            formatted = f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"
        else:
            formatted = f"<s>[INST] {user_msg} [/INST]"
        
        return formatted
    
    def infer(self, messages: List[Dict], model_type: str = "llama") -> str:
        """
        Main inference function that routes to appropriate backend
        
        Args:
            messages: List of message dicts
            model_type: "gpt", "llama", or "mistral"
        
        Returns:
            Generated text
        """
        if model_type == "gpt":
            return self._infer_openai(messages, "gpt-4")
        elif model_type == "llama":
            return self._infer_openrouter(
                messages,
                config.MODEL_CONFIGS["llama-3.1-405b"]["name"]
            )
        elif model_type == "mistral":
            return self._infer_mistral_local(messages)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def infer_mistral_label_probability(self, prompt: str, label: str) -> float:
        """
        Calculate probability for a specific label continuation in Mistral
        
        Args:
            prompt: Formatted prompt
            label: Candidate label
        
        Returns:
            Probability score for the label
        """
        self._load_mistral_local()
        
        # Tokenize prompt and label separately
        prompt_tokens = self.mistral_tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048
        ).to(self.mistral_model.device)
        
        label_tokens = self.mistral_tokenizer(
            label,
            add_special_tokens=False
        ).input_ids
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.mistral_model(**prompt_tokens)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Get probability for the first token of the label
            # (simplified - could extend to multi-token labels)
            if len(label_tokens) > 0:
                label_token_id = label_tokens[0]
                probs = torch.softmax(logits, dim=-1)
                label_prob = probs[label_token_id].item()
            else:
                label_prob = 0.0
        
        return label_prob
    
    def cleanup(self):
        """Free GPU memory"""
        if self.mistral_model is not None:
            del self.mistral_model
            del self.mistral_tokenizer
            torch.cuda.empty_cache()
            print("Mistral model unloaded")


# Global inference instance
_llm_inference = None


def get_llm_inference() -> LLMInference:
    """Get or create global LLM inference instance"""
    global _llm_inference
    if _llm_inference is None:
        _llm_inference = LLMInference()
    return _llm_inference


def run_llm_inference(messages: List[Dict], model_type: str = "llama") -> str:
    """
    Convenience function for LLM inference
    
    Args:
        messages: List of message dicts
        model_type: "gpt", "llama", or "mistral"
    
    Returns:
        Generated text (single label)
    """
    llm = get_llm_inference()
    return llm.infer(messages, model_type)


def run_mistral_label_probability(prompt: str, label: str) -> float:
    """
    Convenience function for Mistral label probability
    
    Args:
        prompt: Formatted prompt
        label: Candidate label
    
    Returns:
        Probability score
    """
    llm = get_llm_inference()
    return llm.infer_mistral_label_probability(prompt, label)
