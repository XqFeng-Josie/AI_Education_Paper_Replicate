"""
Local Llama model wrapper for at-risk student prediction
"""

import torch
import transformers
import logging
from typing import Dict, Any, Optional
from ..models.llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)


class LlamaWrapper(LLMWrapper):
    """Wrapper for local Llama model using transformers pipeline"""
    
    def __init__(self, 
                 model_path: str = "/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
                 temperature: float = 0.7, 
                 max_tokens: int = 2000,
                 device: str = "auto",
                 dtype: str = "bfloat16"):
        """
        Initialize Llama model
        
        Args:
            model_path: Path to local Llama model
            temperature: Sampling temperature
            max_tokens: Maximum new tokens to generate
            device: Device placement ("auto", "cuda", "cpu")
            dtype: Model dtype ("bfloat16", "float16", "float32")
        """
        super().__init__(model_path, temperature, max_tokens)
        
        self.device = device
        self.dtype_str = dtype
        
        # Set dtype
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        logger.info(f"Loading Llama model from {model_path}...")
        logger.info(f"Device: {device}, Dtype: {dtype}")
        
        try:
            # Create pipeline
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": self.dtype},
                device_map=device,
            )
            
            logger.info("✅ Llama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using Llama model
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            Generated text
        """
        # Format messages for Llama chat format
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            # Generate
            outputs = self.pipeline(
                messages,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                do_sample=self.temperature > 0,  # Use sampling if temperature > 0
            )
            
            # Extract response
            response = outputs[0]["generated_text"][-1]['content']
            
            self.total_calls += 1
            # Approximate token count (rough estimate)
            self.total_tokens += len(prompt.split()) + len(response.split())
            
            return response
            
        except Exception as e:
            logger.error(f"Llama generation error: {e}")
            raise
    
    def generate_batch(self, prompts: list, system_prompts: Optional[list] = None) -> list:
        """
        Generate text for multiple prompts in batch
        
        Args:
            prompts: List of user prompts
            system_prompts: List of system prompts (optional)
            
        Returns:
            List of generated texts
        """
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        
        results = []
        for prompt, sys_prompt in zip(prompts, system_prompts):
            result = self.generate(prompt, sys_prompt)
            results.append(result)
        
        return results


class LlamaServerWrapper(LLMWrapper):
    """Wrapper for Llama inference server (HTTP API)"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8000",
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        """
        Initialize Llama server wrapper
        
        Args:
            server_url: URL of the inference server
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__("llama-server", temperature, max_tokens)
        
        self.server_url = server_url.rstrip('/')
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
        
        logger.info(f"Initialized Llama server wrapper: {server_url}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text via server API"""
        
        endpoint = f"{self.server_url}/generate"
        
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = self.requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            self.total_calls += 1
            self.total_tokens += result.get('tokens_used', 0)
            
            return result['generated_text']
            
        except Exception as e:
            logger.error(f"Server API error: {e}")
            raise

