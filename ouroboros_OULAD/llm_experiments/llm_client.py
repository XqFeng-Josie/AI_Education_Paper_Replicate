"""
Unified LLM Client
Supports OpenRouter API and local Llama server
"""

import os
import time
import logging
import requests
import threading
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: int = 60,
                 temperature: float = 0.1,
                 max_tokens: int = 2048):
        """
        Initialize LLM client
        
        Args:
            max_retries: Maximum number of retries on failure
            retry_delay: Delay in seconds between retries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        """
        Implementation of text generation (to be overridden by subclasses)
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (None = use default)
            max_tokens: Maximum tokens (None = use default)
            
        Returns:
            Generated text
        """
        pass
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: Optional[float] = None, 
                max_tokens: Optional[int] = None) -> str:
        """
        Generate text with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (None = use default)
            max_tokens: Maximum tokens (None = use default)
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If generation fails after all retries
        """
        # Use default values if not specified
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self._generate_impl(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {e}")
        
        raise RuntimeError(f"API call failed after {self.max_retries} attempts") from last_exception


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "meta-llama/llama-3.1-70b-instruct",
                 base_url: str = "https://openrouter.ai/api/v1",
                 max_retries: int = 3,
                 retry_delay: int = 60,
                 temperature: float = 0.1,
                 max_tokens: int = 2048):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model name
            base_url: API base URL
            max_retries: Maximum number of retries
            retry_delay: Delay in seconds between retries
            temperature: Default sampling temperature
            max_tokens: Default maximum tokens
        """
        super().__init__(max_retries, retry_delay, temperature, max_tokens)
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Please set environment variable or pass api_key parameter.")
        
        self.model = model
        self.base_url = base_url
        
        # Try to import OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.use_openai = True
        except ImportError:
            logger.warning("OpenAI package not available, will use requests")
            self.client = None
            self.use_openai = False
        
        logger.info(f"Initialized OpenRouterClient with model: {self.model}")
    
    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        """Generate text using OpenRouter API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.use_openai:
            # Use OpenAI client
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        else:
            # Use requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()


class LocalLlamaClient(BaseLLMClient):
    """Local Llama server client"""
    
    def __init__(self,
                 server_url: str = "http://localhost:8001",
                 max_retries: int = 3,
                 retry_delay: int = 60,
                 temperature: float = 0.1,
                 max_tokens: int = 2048,
                 timeout: int = 300):
        """
        Initialize Local Llama client
        
        Args:
            server_url: URL of the local Llama server
            max_retries: Maximum number of retries
            retry_delay: Delay in seconds between retries
            temperature: Default sampling temperature
            max_tokens: Default maximum tokens
            timeout: Request timeout in seconds
        """
        super().__init__(max_retries, retry_delay, temperature, max_tokens)
        
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        
        # Check if server is running
        self._check_health()
        
        logger.info(f"Initialized LocalLlamaClient connected to: {self.server_url}")
    
    def _check_health(self):
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            logger.info(f"✅ Local Llama server is healthy at {self.server_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to local Llama server at {self.server_url}: {e}")
            logger.error("Please ensure the server is running. Start it with:")
            logger.error("  bash start_llama_server.sh")
            raise ConnectionError(f"Cannot connect to local Llama server: {e}")
    
    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        """Generate text using local Llama server"""
        data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "do_sample": True if temperature > 0 else False
        }
        
        response = requests.post(
            f"{self.server_url}/generate",
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError("Server returned failure")
        
        return result["generated_text"].strip()


class MultiServerLlamaClient(BaseLLMClient):
    """Multi-server local Llama client with load balancing"""
    
    def __init__(self,
                 server_urls: list = None,
                 base_port: int = 8000,
                 num_servers: int = 4,
                 max_retries: int = 3,
                 retry_delay: int = 60,
                 temperature: float = 0.1,
                 max_tokens: int = 2048,
                 timeout: int = 300,
                 load_balance_strategy: str = "round_robin"):
        """
        Initialize Multi-Server Llama client with load balancing
        
        Args:
            server_urls: List of server URLs (if None, auto-generate from base_port)
            base_port: Base port for auto-generating URLs (default: 8000)
            num_servers: Number of servers (for auto-generation)
            max_retries: Maximum number of retries per server
            retry_delay: Delay in seconds between retries
            temperature: Default sampling temperature
            max_tokens: Default maximum tokens
            timeout: Request timeout in seconds
            load_balance_strategy: Load balancing strategy ("round_robin" or "random")
        """
        super().__init__(max_retries, retry_delay, temperature, max_tokens)
        
        # Generate server URLs if not provided
        if server_urls is None:
            server_urls = [f"http://localhost:{base_port + i}" for i in range(num_servers)]
        
        self.server_urls = [url.rstrip('/') for url in server_urls]
        self.timeout = timeout
        self.load_balance_strategy = load_balance_strategy
        self.current_server_index = 0
        self.lock = threading.Lock()  # Lock for thread-safe operations
        
        # Check health of all servers
        self.healthy_servers = []
        self._check_all_servers_health()
        
        if len(self.healthy_servers) == 0:
            raise ConnectionError("No healthy servers available!")
        
        logger.info(f"Initialized MultiServerLlamaClient with {len(self.healthy_servers)}/{len(self.server_urls)} healthy servers")
        logger.info(f"Healthy servers: {self.healthy_servers}")
        logger.info(f"Load balance strategy: {self.load_balance_strategy}")
    
    def _check_all_servers_health(self):
        """Check health of all servers"""
        import threading
        
        def check_single_server(url):
            try:
                response = requests.get(f"{url}/health", timeout=5)
                response.raise_for_status()
                return url, True
            except Exception as e:
                logger.warning(f"Server {url} is not healthy: {e}")
                return url, False
        
        # Use threading to check all servers in parallel
        threads = []
        results = []
        
        def worker(url):
            result = check_single_server(url)
            results.append(result)
        
        for url in self.server_urls:
            thread = threading.Thread(target=worker, args=(url,))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        self.healthy_servers = [url for url, healthy in results if healthy]
    
    def _get_next_server(self) -> str:
        """Get next server URL based on load balancing strategy"""
        import random
        
        if self.load_balance_strategy == "round_robin":
            # Thread-safe round-robin using modulo
            with self.lock:
                server = self.healthy_servers[self.current_server_index % len(self.healthy_servers)]
                self.current_server_index += 1
            return server
        elif self.load_balance_strategy == "random":
            return random.choice(self.healthy_servers)
        else:
            return self.healthy_servers[0]
    
    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> str:
        """Generate text using multi-server load balancing"""
        # Try servers in order until one succeeds
        tried_servers = set()
        last_exception = None
        
        while len(tried_servers) < len(self.healthy_servers):
            server_url = self._get_next_server()
            
            # Skip if already tried
            if server_url in tried_servers:
                continue
            
            tried_servers.add(server_url)
            
            try:
                data = {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "do_sample": True if temperature > 0 else False
                }
                
                response = requests.post(
                    f"{server_url}/generate",
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                if not result.get("success", False):
                    raise RuntimeError("Server returned failure")
                
                return result["generated_text"].strip()
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Server {server_url} failed: {e}, trying next server...")
                continue
        
        # All servers failed
        raise RuntimeError(f"All {len(self.healthy_servers)} servers failed. Last error: {last_exception}")


def create_llm_client(
    provider: str = "openrouter",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    server_url: Optional[str] = None,
    server_urls: Optional[list] = None,
    base_port: int = 8000,
    num_servers: int = 1,
    max_retries: int = 3,
    retry_delay: int = 60,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    load_balance_strategy: str = "round_robin"
) -> BaseLLMClient:
    """
    Factory function to create LLM client
    
    Args:
        provider: LLM provider ("openrouter", "local", or "multi_local")
        api_key: API key (for OpenRouter)
        model: Model name (for OpenRouter)
        server_url: Server URL (for single local Llama)
        server_urls: List of server URLs (for multi_local)
        base_port: Base port for auto-generating URLs (for multi_local)
        num_servers: Number of servers (for multi_local, default: 1)
        max_retries: Maximum number of retries
        retry_delay: Delay in seconds between retries
        temperature: Default sampling temperature
        max_tokens: Default maximum tokens
        load_balance_strategy: Load balancing strategy for multi_local ("round_robin" or "random")
        
    Returns:
        LLM client instance
        
    Raises:
        ValueError: If provider is not supported
        
    Examples:
        >>> # OpenRouter client
        >>> client = create_llm_client(
        ...     provider="openrouter",
        ...     model="meta-llama/llama-3.1-70b-instruct"
        ... )
        
        >>> # Single local Llama client
        >>> client = create_llm_client(
        ...     provider="local",
        ...     server_url="http://localhost:8001"
        ... )
        
        >>> # Multi-server local Llama client (auto-generate URLs)
        >>> client = create_llm_client(
        ...     provider="multi_local",
        ...     base_port=8000,
        ...     num_servers=4
        ... )
        
        >>> # Multi-server local Llama client (explicit URLs)
        >>> client = create_llm_client(
        ...     provider="multi_local",
        ...     server_urls=["http://localhost:8000", "http://localhost:8001"]
        ... )
    """
    provider = provider.lower()
    
    if provider == "openrouter":
        return OpenRouterClient(
            api_key=api_key,
            model=model or "meta-llama/llama-3.1-70b-instruct",
            max_retries=max_retries,
            retry_delay=retry_delay,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == "local":
        return LocalLlamaClient(
            server_url=server_url or "http://localhost:8001",
            max_retries=max_retries,
            retry_delay=retry_delay,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif provider == "multi_local":
        return MultiServerLlamaClient(
            server_urls=server_urls,
            base_port=base_port,
            num_servers=num_servers,
            max_retries=max_retries,
            retry_delay=retry_delay,
            temperature=temperature,
            max_tokens=max_tokens,
            load_balance_strategy=load_balance_strategy
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openrouter', 'local', or 'multi_local'")


if __name__ == "__main__":
    """Test the LLM clients"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM clients")
    parser.add_argument("--provider", type=str, choices=["openrouter", "local"], 
                       default="local", help="LLM provider")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-70b-instruct",
                       help="Model name (for OpenRouter)")
    parser.add_argument("--server_url", type=str, default="http://localhost:8001",
                       help="Server URL (for local)")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                       help="Test prompt")
    
    args = parser.parse_args()
    
    # Create client
    logging.basicConfig(level=logging.INFO)
    client = create_llm_client(
        provider=args.provider,
        model=args.model,
        server_url=args.server_url
    )
    
    # Test generation
    print(f"\n{'='*60}")
    print(f"Testing {args.provider} client")
    print(f"{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    
    response = client.generate(
        prompt=args.prompt,
        system_prompt="You are a helpful assistant."
    )
    
    print(f"Response:\n{response}")
    print(f"\n{'='*60}")

