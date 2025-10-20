"""
Multi-GPU Load Balancing Client for Llama Servers

自动在多个GPU服务器间分发请求，实现负载均衡
"""

import requests
import logging
import time
from typing import List, Optional
from threading import Lock
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiGPULlamaClient:
    """
    多GPU负载均衡客户端
    
    支持两种负载均衡策略：
    1. round_robin: 轮询分配
    2. random: 随机分配
    """
    
    def __init__(self, server_urls: List[str], strategy: str = "round_robin", timeout: int = 300):
        """
        Args:
            server_urls: 服务器URL列表，例如 ["http://localhost:8000", "http://localhost:8001"]
            strategy: 负载均衡策略 ("round_robin" 或 "random")
            timeout: 请求超时时间（秒）
        """
        self.server_urls = server_urls
        self.strategy = strategy
        self.timeout = timeout
        self.current_index = 0
        self.lock = Lock()
        
        # 检查服务器状态
        self.check_servers()
        
        logger.info(f"MultiGPULlamaClient initialized with {len(self.server_urls)} servers")
        logger.info(f"Strategy: {self.strategy}")
    
    def check_servers(self):
        """检查所有服务器是否健康"""
        healthy_servers = []
        
        for url in self.server_urls:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    healthy_servers.append(url)
                    logger.info(f"✓ Server {url} is healthy")
                else:
                    logger.warning(f"✗ Server {url} returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"✗ Server {url} is not accessible: {e}")
        
        if not healthy_servers:
            raise RuntimeError("No healthy servers available!")
        
        self.server_urls = healthy_servers
        logger.info(f"Active servers: {len(self.server_urls)}")
    
    def get_next_server(self) -> str:
        """根据策略获取下一个服务器URL"""
        with self.lock:
            if self.strategy == "round_robin":
                server = self.server_urls[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.server_urls)
                return server
            elif self.strategy == "random":
                return random.choice(self.server_urls)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 2000,
                 max_retries: int = 3) -> str:
        """
        生成文本，自动负载均衡到不同GPU
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            max_retries: 最大重试次数
            
        Returns:
            生成的文本
        """
        
        request_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(max_retries):
            server_url = self.get_next_server()
            
            try:
                logger.debug(f"Attempt {attempt + 1}: Sending request to {server_url}")
                
                start_time = time.time()
                response = requests.post(
                    f"{server_url}/generate",
                    json=request_data,
                    timeout=self.timeout
                )
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✓ Request completed in {elapsed_time:.2f}s using {server_url}")
                    return result["generated_text"]
                else:
                    logger.error(f"Server {server_url} returned error: {response.status_code}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying with another server...")
                        continue
                    else:
                        raise RuntimeError(f"All retries failed. Last status: {response.status_code}")
            
            except requests.exceptions.Timeout:
                logger.error(f"Timeout when calling {server_url}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with another server...")
                    continue
                else:
                    raise RuntimeError("All retries failed due to timeout")
            
            except Exception as e:
                logger.error(f"Error calling {server_url}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying with another server...")
                    continue
                else:
                    raise RuntimeError(f"All retries failed. Last error: {e}")
        
        raise RuntimeError("Failed to generate text after all retries")
    
    def get_stats(self):
        """获取所有服务器的统计信息"""
        stats = []
        for url in self.server_urls:
            try:
                response = requests.get(f"{url}/", timeout=5)
                if response.status_code == 200:
                    stats.append({
                        "url": url,
                        "status": response.json()
                    })
            except Exception as e:
                stats.append({
                    "url": url,
                    "error": str(e)
                })
        return stats


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # 创建客户端（假设在4个GPU上启动了4个服务器）
    server_urls = [
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:8002",
        "http://localhost:8003"
    ]
    
    client = MultiGPULlamaClient(server_urls, strategy="round_robin")
    
    # 测试生成
    prompt = "Explain machine learning in simple terms."
    system_prompt = "You are a helpful AI assistant."
    
    print("Generating response...")
    response = client.generate(prompt, system_prompt)
    print(f"Response: {response}")
    
    # 查看服务器状态
    print("\nServer Stats:")
    for stat in client.get_stats():
        print(f"  {stat}")

