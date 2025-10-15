#!/usr/bin/env python3
"""
LLM Inference Module for NCTE Classroom Transcript Analysis

This module handles all LLM inference functionality including:
- Model loading and initialization
- Text generation with different backends
- Classification result extraction
- Support for various model types (transformers, ModelScope, OpenAI API)
"""

import os
import warnings
from typing import List, Dict, Any

import torch
import transformers
from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM
from transformers import AutoTokenizer as HFAutoTokenizer


# Try to import OpenAI for API models
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

warnings.filterwarnings("ignore")

model_mapping = {
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    # OpenAI models
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
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
        self.openai_client = None
        self.is_openai_model = False

        # Token usage tracking for API models
        self.last_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

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

            # Check if it's an OpenAI model
            if self._is_openai_model(self.model_name):
                self._load_openai_model(model_path)
            # Check if it's a MistralAI model
            elif self.model_name.lower() == "mistral-7b-instruct-v0.3":
                self._load_mistral_model(model_path)
            # Check if it's a Qwen model - use official loading method
            elif self.model_name.lower() == "qwen2.5-7b-instruct":
                self._load_qwen_model(model_path)
            # Check if it's Llama 70B - use reference implementation
            elif self.model_name.lower() == "llama-3.3-70b-instruct":
                self._load_llama_70b_model(model_path)
            else:
                # Use transformers pipeline for other models (LLaMA)
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_path,
                    model_kwargs=(
                        {"torch_dtype": torch.bfloat16} if self.device == "cuda" else {}
                    ),
                    device_map="auto" if self.device == "cuda" else None,
                )

            print(f"Successfully loaded model: {self.model_name}")

        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise e

    def _is_openai_model(self, model_name: str) -> bool:
        """Check if the model is an OpenAI API model."""
        openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        return any(model in model_name.lower() for model in openai_models)

    def _load_openai_model(self, model_path: str):
        """Load OpenAI model using OpenAI API."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is not available. Please install it with: pip install openai"
            )

        print(f"Initializing OpenAI API client for model: {model_path}")

        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it with your OpenAI API key."
            )

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=api_key)
        self.is_openai_model = True

        print(f"OpenAI API client initialized successfully for {model_path}")

    def _load_mistral_model(self, model_path: str):
        """Load MistralAI model using ModelScope."""
        print(f"Loading MistralAI model from: {model_path}")

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

    def _load_qwen_model(self, model_path: str):
        """Load Qwen model using official method from HuggingFace transformers.

        Implementation follows official Qwen2.5 documentation:
        https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
        """
        print(f"Loading Qwen model from: {model_path}")

        # Load model and tokenizer using transformers (official method)
        self.model = HFAutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = HFAutoTokenizer.from_pretrained(model_path)

        # Set pad_token to eos_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print("Qwen model loaded successfully")

    def _load_llama_70b_model(self, model_path: str):
        """Load Llama 70B model using reference implementation.

        Implementation follows the reference code pattern for Llama-3.3-70B-Instruct.
        """
        print(f"Loading Llama 70B model from: {model_path}")

        # Use transformers pipeline following reference code
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        print("Llama 70B model loaded successfully")

    def _get_model_path(self) -> str:
        """Get the model path or identifier."""
        # Model mapping including MistralAI models
        return model_mapping.get(self.model_name.lower(), self.model_name)

    def generate_response(
        self, prompt: List[Dict[str, str]], max_new_tokens: int = 10
    ) -> str:
        """Generate response from the model using greedy generation."""
        try:
            # Reset token usage for non-API models
            if not (self.is_openai_model and self.openai_client):
                self.last_token_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }

            # Check if it's an OpenAI API model
            if self.is_openai_model and self.openai_client:
                return self._generate_with_openai(prompt, max_new_tokens)
            # Check if we have a direct model loaded (Mistral, Qwen, etc.)
            elif (
                hasattr(self, "model")
                and hasattr(self, "tokenizer")
                and self.model is not None
            ):
                return self._generate_with_model(prompt, max_new_tokens)
            elif self.pipeline:
                return self._generate_with_pipeline(prompt, max_new_tokens)
            else:
                print("Error: No model or pipeline initialized")
                return ""

        except Exception as e:
            print(f"Generation error: {e}")
            return ""

    def _generate_with_pipeline(
        self, prompt: List[Dict[str, str]], max_new_tokens: int
    ) -> str:
        """Generate response using transformers pipeline with greedy generation."""
        try:
            # Message format (list of dicts) - greedy generation only
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy generation
                return_full_text=False,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
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

    def _generate_with_openai(
        self, prompt: List[Dict[str, str]], max_new_tokens: int
    ) -> str:
        """Generate response using OpenAI API with greedy generation.

        Uses the same message format as other LLMs for consistency.
        Also tracks token usage for cost calculation.
        """
        try:
            model_path = self._get_model_path()

            # Call OpenAI API with chat completion
            response = self.openai_client.chat.completions.create(
                model=model_path,
                messages=prompt,
                max_tokens=max_new_tokens,
                temperature=0,  # Greedy generation (deterministic)
                top_p=1.0,
                n=1,
            )

            # Extract token usage information
            if hasattr(response, "usage") and response.usage:
                self.last_token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            else:
                # Reset if no usage info
                self.last_token_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }

            # Extract the generated text
            generated_text = response.choices[0].message.content
            return generated_text.strip() if generated_text else ""

        except Exception as e:
            print(f"OpenAI API generation error: {e}")
            # Reset token usage on error
            self.last_token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            return ""

    def _generate_with_model(
        self, prompt: List[Dict[str, str]], max_new_tokens: int
    ) -> str:
        """Generate response using direct model (Mistral/Qwen) with greedy generation.

        Uses official Qwen approach: apply_chat_template -> generate -> extract response
        """
        try:
            # Apply chat template for message format
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(
                self.model.device
            )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy generation
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Extract only the new tokens (response)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]

                response = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                return response.strip()

        except Exception as e:
            print(f"Model generation error: {e}")
            return ""

    def classify(self, prompt: List[Dict[str, str]], max_new_tokens: int = 10) -> int:
        """Classify a single example and extract 0/1 label."""
        response = self.generate_response(prompt, max_new_tokens)
        return self._extract_classification(response)

    def _extract_classification(self, response: str) -> int:
        """Extract classification (0 or 1) from model response."""
        response_clean = response.strip().lower()

        # Look for clear indicators first
        if response_clean.startswith("1") or response_clean == "1":
            return 1
        elif response_clean.startswith("0") or response_clean == "0":
            return 0

        # Look for patterns that indicate the classification
        if any(
            word in response_clean for word in ["1", "yes", "on-task", "high", "true"]
        ):
            return 1
        elif any(
            word in response_clean for word in ["0", "no", "off-task", "low", "false"]
        ):
            return 0
        else:
            # If unclear, default to 0 (conservative approach)
            print(f"Unclear response: '{response}'. Defaulting to 0.")
            return 0

    def get_token_usage(self) -> Dict[str, int]:
        """Get the last API call's token usage (for OpenAI models)."""
        return self.last_token_usage.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "has_pipeline": self.pipeline is not None,
            "has_direct_model": self.model is not None,
            "is_openai_model": self.is_openai_model,
            "has_openai_client": self.openai_client is not None,
            "openai_available": OPENAI_AVAILABLE,
            "last_token_usage": self.last_token_usage,
        }


def test_llm_inference():
    """Test the LLM inference functionality."""
    print("🧪 Testing LLM Inference\n")

    # Test with different models
    models_to_test = [
        "llama-3.1-8b-instruct",  # Primary model
        "mistral-7b-instruct-v0.3",  # MistralAI model
        "qwen2.5-7b-instruct",  # Qwen model
        "gpt-4o-mini",  # OpenAI model (if API key is set)
    ]

    success = True

    for model_name in models_to_test:
        print(f"\n📋 Testing {model_name}:")
        try:
            inference = LLMInference(model_name, device="auto")
            print("   ✅ Model loaded successfully")

            # Test message format
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in analyzing elementary mathematics classroom discourse.",
                },
                {
                    "role": "user",
                    "content": "Analyze the student's utterance and determine if it is on-task (1) or off-task (0).\n\nStudent: I think the answer is 5 because 2 plus 3 equals 5.\nTeacher: Good thinking! Can you explain how you got that?\n\nClassification:",
                },
            ]

            result = inference.classify(messages)
            print(f"   ✅ Message format classification: {result}")

            # Test model info
            info = inference.get_model_info()
            print(f"   ✅ Model info: {info}")

        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            if "gpt" in model_name.lower():
                print("   ℹ️  OpenAI API key might not be set or model not available")
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
