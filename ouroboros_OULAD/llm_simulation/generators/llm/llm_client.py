"""
Client for Llama Server
"""

import requests
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LlamaClient:
    """Client for interacting with Llama server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Test connection
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Connected to Llama server at {base_url}")
            else:
                logger.warning(f"⚠️ Server returned status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Llama server: {e}")
            logger.error(f"Please ensure the server is running: python llm/server/llama_server.py")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        do_sample: bool = True
    ) -> Optional[str]:
        """
        Generate text using Llama
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (agent role)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            
        Returns:
            Generated text or None if failed
        """
        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "do_sample": do_sample
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('generated_text')
            else:
                logger.error(f"Generation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None
    
    def is_healthy(self) -> bool:
        """Check if server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

