"""
LLM wrapper for different providers (OpenAI, Anthropic, HuggingFace, Local)
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMWrapper(ABC):
    """Abstract base class for LLM wrappers"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 2000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_calls = 0
        self.total_tokens = 0
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt"""
        pass
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate and parse JSON response with robust extraction"""
        response = self.generate(prompt, system_prompt)
        
        # Try to extract JSON from response
        try:
            json_str = None
            
            # Strategy 1: Look for JSON in code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                # Strategy 2: Find the largest valid JSON object using regex
                import re
                # Find all potential JSON objects (handling nested braces)
                json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
                matches = re.findall(json_pattern, response, re.DOTALL)
                
                if matches:
                    # Try each match, starting with the longest (most complete)
                    matches.sort(key=len, reverse=True)
                    for match in matches:
                        try:
                            # Test if this is valid JSON
                            json.loads(match)
                            json_str = match
                            break
                        except:
                            continue
                
                # Strategy 3: If still no valid JSON, try the whole response
                if json_str is None:
                    json_str = response.strip()
            
            # Parse JSON
            parsed_json = json.loads(json_str)
            logger.info(f"Successfully parsed JSON response")
            return parsed_json
            
        except Exception as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.error(f"Response preview: {response[:500]}...")
            
            # Fallback: Try to extract key information even if JSON parsing fails
            fallback_data = self._extract_fallback_json(response)
            
            # Always include the raw response for debugging
            fallback_data["error"] = "Failed to parse JSON"
            fallback_data["raw_response"] = response
            
            return fallback_data
    
    def _extract_fallback_json(self, response: str) -> Dict[str, Any]:
        """
        Attempt to extract key information from malformed JSON response
        This is a fallback when JSON parsing completely fails
        """
        import re
        fallback = {}
        
        # Try to extract risk level
        risk_patterns = [
            r'"final_risk_level"\s*:\s*"([^"]+)"',
            r'"risk_level"\s*:\s*"([^"]+)"',
            r'Risk Level[:\s]+([^"\n]+)',
        ]
        for pattern in risk_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                fallback['extracted_risk_level'] = match.group(1).strip()
                break
        
        # Try to extract risk score
        score_patterns = [
            r'"risk_score"\s*:\s*(\d+\.?\d*)',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    fallback['extracted_risk_score'] = float(match.group(1))
                except:
                    pass
                break
        
        # Try to extract confidence
        confidence_patterns = [
            r'"confidence"\s*:\s*"([^"]+)"',
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                fallback['extracted_confidence'] = match.group(1).strip()
                break
        
        logger.info(f"Extracted fallback data: {fallback}")
        return fallback
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics"""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens
        }


class OpenAIWrapper(LLMWrapper):
    """Wrapper for OpenAI API"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7, 
                 max_tokens: int = 2000, api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.total_calls += 1
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicWrapper(LLMWrapper):
    """Wrapper for Anthropic Claude API"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", 
                 temperature: float = 0.7, max_tokens: int = 2000, 
                 api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt if system_prompt else "",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            self.total_calls += 1
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class HuggingFaceWrapper(LLMWrapper):
    """Wrapper for HuggingFace Inference API"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, 
                 max_tokens: int = 2000, api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
        
        api_key = api_key or os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HuggingFace API key not provided")
        
        self.client = InferenceClient(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using HuggingFace API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.total_calls += 1
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise


class LocalLLMWrapper(LLMWrapper):
    """Wrapper for local LLM using transformers"""
    
    def __init__(self, model_path: str, temperature: float = 0.7, 
                 max_tokens: int = 2000, device: str = "cuda"):
        super().__init__(model_path, temperature, max_tokens)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install transformers and torch")
        
        logger.info(f"Loading local model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.device = device
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using local model"""
        # Format prompt with system message
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        self.total_calls += 1
        
        return response


def create_llm_wrapper(provider: str, config: Dict[str, Any]) -> LLMWrapper:
    """Factory function to create LLM wrapper based on provider"""
    
    if provider == "openai":
        return OpenAIWrapper(
            model_name=config.get("model", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            api_key=os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
        )
    
    elif provider == "anthropic":
        return AnthropicWrapper(
            model_name=config.get("model", "claude-3-5-sonnet-20241022"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            api_key=os.getenv(config.get("api_key_env", "ANTHROPIC_API_KEY"))
        )
    
    elif provider == "huggingface":
        return HuggingFaceWrapper(
            model_name=config.get("model"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            api_key=os.getenv(config.get("api_key_env", "HF_API_KEY"))
        )
    
    elif provider == "local":
        return LocalLLMWrapper(
            model_path=config.get("model_path"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            device=config.get("device", "cuda")
        )
    
    elif provider == "llama":
        from .llama_wrapper import LlamaWrapper
        return LlamaWrapper(
            model_path=config.get("model_path", "/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            device=config.get("device", "auto"),
            dtype=config.get("dtype", "bfloat16")
        )
    
    elif provider == "llama_server":
        from .llama_wrapper import LlamaServerWrapper
        return LlamaServerWrapper(
            server_url=config.get("server_url", "http://localhost:8000"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000)
        )
    
    elif provider == "multi_gpu_llama":
        from .multi_gpu_llama_wrapper import MultiGPULlamaWrapper
        return MultiGPULlamaWrapper(
            server_urls=config.get("server_urls", [
                "http://localhost:8000",
                "http://localhost:8001",
                "http://localhost:8002",
                "http://localhost:8003"
            ]),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            strategy=config.get("strategy", "round_robin")
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")





