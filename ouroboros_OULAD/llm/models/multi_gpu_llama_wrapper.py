"""
Multi-GPU LLM Wrapper for experiment integration
整合多GPU客户端到实验框架
"""

import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

from multi_gpu_client import MultiGPULlamaClient
from .llm_wrapper import LLMWrapper
import logging

logger = logging.getLogger(__name__)


class MultiGPULlamaWrapper(LLMWrapper):
    """
    Multi-GPU Llama Wrapper
    
    自动在多个GPU服务器间负载均衡
    """
    
    def __init__(self, server_urls: list, temperature: float = 0.7, 
                 max_tokens: int = 2000, strategy: str = "round_robin"):
        """
        Args:
            server_urls: 服务器URL列表，例如：
                ["http://localhost:8000", "http://localhost:8001", "http://localhost:8002"]
            temperature: 温度参数
            max_tokens: 最大token数
            strategy: 负载均衡策略 ("round_robin" 或 "random")
        """
        # Initialize base class with a descriptive model name
        model_name = f"multi_gpu_llama_{len(server_urls)}x"
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        
        self.server_urls = server_urls
        self.strategy = strategy
        
        # 初始化多GPU客户端
        self.client = MultiGPULlamaClient(
            server_urls=server_urls,
            strategy=strategy,
            timeout=300
        )
        
        logger.info(f"MultiGPULlamaWrapper initialized with {len(self.client.server_urls)} servers")
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """生成文本"""
        try:
            response = self.client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 更新统计
            self.total_calls += 1
            # 粗略估算token数
            self.total_tokens += len(prompt.split()) + len(response.split())
            
            return response
        
        except Exception as e:
            logger.error(f"MultiGPU generation error: {e}")
            raise
    
    def get_server_stats(self):
        """获取所有服务器状态"""
        return self.client.get_stats()

