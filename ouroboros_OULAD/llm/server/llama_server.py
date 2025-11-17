"""
FastAPI server for Llama model inference

Usage:
    python llama_server.py --model_path /path/to/llama --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import transformers
import torch
import logging
from typing import Optional
import argparse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama Inference Server")

# Global model pipeline
pipeline = None


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    do_sample: bool = True


class GenerateResponse(BaseModel):
    generated_text: str
    tokens_used: int
    success: bool


def load_model(model_path: str, device: str = "auto", dtype: str = "bfloat16"):
    """Load Llama model"""
    global pipeline
    
    logger.info(f"Loading model from {model_path}...")
    
    # Set dtype
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch_dtype},
        device_map=device,
    )
    
    logger.info("✅ Model loaded successfully")


@app.get("/")
def root():
    """Health check"""
    return {
        "status": "running",
        "model_loaded": pipeline is not None
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate text endpoint"""
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format messages
        messages = []
        
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": request.prompt
        })
        print(messages)
        print("*"*100)
        # Generate
        outputs = pipeline(
            messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            do_sample=request.do_sample,
        )
        # Extract response
        generated_text = outputs[0]["generated_text"][-1]['content']
        print(generated_text)
        print("*"*100)
        # Estimate tokens (rough)
        tokens_used = len(request.prompt.split()) + len(generated_text.split())
        
        return GenerateResponse(
            generated_text=generated_text,
            tokens_used=tokens_used,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Llama Inference Server")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/u/xfeng4/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        help="Path to Llama model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model_path, args.device, args.dtype)
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

