"""
LLM client module
Generate text using OpenRouter API
"""
import os
import logging
from typing import Optional, Dict, Any, List
import time

logger = logging.getLogger(__name__)

try:
    import openai
    # 检查是否是新版本OpenAI SDK (v1.0+)
    try:
        from openai import OpenAI
        OPENAI_NEW_VERSION = True
    except ImportError:
        OPENAI_NEW_VERSION = False
except ImportError:
    logger.error("Please install openai library: pip install openai")
    raise


class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    # Rate limiting: 5 seconds between requests
    RATE_LIMIT_INTERVAL = 5.0
    # Error retry: sleep 1 minute (60 seconds) after error
    ERROR_RETRY_SLEEP = 60.0
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-3.1-70b-instruct",
        base_url: str = "https://openrouter.ai/api/v1",
        fallback_model: Optional[str] = None
    ):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key (if None, read from environment variable OPENROUTER_API_KEY)
            model: Model name to use (default: meta-llama/llama-3.1-70b-instruct)
            base_url: API base URL (default: https://openrouter.ai/api/v1)
            fallback_model: Fallback model name if primary model fails (optional)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Please set environment variable or pass api_key parameter.\n"
                "Get API key: https://openrouter.ai/keys"
            )
        
        self.model = model
        self.fallback_model = fallback_model
        self.current_model = model  # Initialize current_model before _test_connection
        self.base_url = base_url
        
        # Rate limiting: track last request time
        self._last_request_time = 0.0
        
        # Initialize OpenAI client (support both old and new versions)
        if OPENAI_NEW_VERSION:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            # Old version SDK
            openai.api_key = self.api_key
            openai.api_base = self.base_url
            self.client = None
        
        self._test_connection()
        
    def _test_connection(self):
        """Test connection to OpenRouter API"""
        try:
            # Test connection with a simple request
            messages = [
                {"role": "user", "content": "Hello"}
            ]
            if OPENAI_NEW_VERSION:
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    max_tokens=10
                )
            else:
                response = openai.ChatCompletion.create(
                    model=self.current_model,
                    messages=messages,
                    max_tokens=10
                )
            logger.info(f"✅ Connected to OpenRouter API, model: {self.current_model}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to OpenRouter API: {e}")
            logger.error("Please check:")
            logger.error("1. Is the API key correctly set?")
            logger.error("2. Is the network connection normal?")
            logger.error("3. Is the model name correct?")
            raise
    
    def _wait_for_rate_limit(self):
        """Wait to respect rate limit (5 seconds between requests)"""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self.RATE_LIMIT_INTERVAL:
            sleep_time = self.RATE_LIMIT_INTERVAL - time_since_last_request
            logger.debug(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _switch_to_fallback_model(self):
        """Switch to fallback model if available"""
        if self.fallback_model and self.current_model != self.fallback_model:
            logger.warning(f"Switching from {self.current_model} to fallback model {self.fallback_model}")
            self.current_model = self.fallback_model
            return True
        return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        do_sample: bool = True,
        retry_times: int = 3
    ) -> Optional[str]:
        """
        Generate text using OpenRouter API with rate limiting and error retry
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (agent role)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (handled automatically by OpenRouter)
            retry_times: Number of retries
            
        Returns:
            Generated text, returns None on failure
        """
        # Build message list
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Reset to primary model at start
        self.current_model = self.model
        
        for attempt in range(retry_times):
            try:
                # Rate limiting: wait 5 seconds between requests
                self._wait_for_rate_limit()
                
                # Make API call
                if OPENAI_NEW_VERSION:
                    response = self.client.chat.completions.create(
                        model=self.current_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    generated_text = response.choices[0].message.content.strip()
                else:
                    response = openai.ChatCompletion.create(
                        model=self.current_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    generated_text = response.choices[0].message.content.strip()
                
                return generated_text
                    
            except Exception as e:
                logger.warning(f"Generation failed (attempt {attempt + 1}/{retry_times}): {e}")
                
                # If not the last attempt, retry
                if attempt < retry_times - 1:
                    # Try switching to fallback model if available
                    if attempt == 1 and self._switch_to_fallback_model():
                        logger.info(f"Retrying with fallback model: {self.current_model}")
                    else:
                        # Sleep 1 minute before retry
                        logger.info(f"Waiting {self.ERROR_RETRY_SLEEP} seconds before retry...")
                        time.sleep(self.ERROR_RETRY_SLEEP)
        
        return None
    
    def is_healthy(self) -> bool:
        """Check if API is available"""
        try:
            messages = [{"role": "user", "content": "test"}]
            if OPENAI_NEW_VERSION:
                response = self.client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    max_tokens=5
                )
            else:
                response = openai.ChatCompletion.create(
                    model=self.current_model,
                    messages=messages,
                    max_tokens=5
                )
            return True
        except:
            return False


# For backward compatibility, keep LlamaClient as an alias
LlamaClient = OpenRouterClient

