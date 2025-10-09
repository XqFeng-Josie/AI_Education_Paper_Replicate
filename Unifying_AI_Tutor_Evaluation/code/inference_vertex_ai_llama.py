import json
import numpy as np
import os
import random
import time
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Dict, Any

# Google Cloud and OpenAI imports
import vertexai
import openai
from google.auth import default

from prompt.prompt import prompt_MathDial, prompt_Bridge

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Get access token from gcloud
def get_token():
    import subprocess
    try:
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to get access token: {e}") from e
    except FileNotFoundError as exc:
        raise Exception("gcloud CLI not found. Please install Google Cloud SDK.") from exc
    
class VertexAILlamaInference:
    def __init__(self, project_id: str = None, location: str = "us-central1", 
                 model_name: str = "meta/llama3-8b-instruct-maas", bucket_name: str = None):
        """
        Initialize Vertex AI Llama inference client using OpenAI-compatible interface
        
        Args:
            project_id: Google Cloud Project ID. If None, will try to get from environment
            location: Google Cloud region (default: us-central1)
            model_name: Model name in Vertex AI (default: meta/llama3-8b-instruct-maas)
            bucket_name: GCS bucket name for staging (optional)
        """
        # Get project ID from environment if not provided
        if project_id is None:
            try:
                _, project_id = default()
            except Exception as e:
                raise ValueError(f"Could not determine project ID. Please set GOOGLE_CLOUD_PROJECT environment variable or pass project_id parameter. Error: {e}") from e
        
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.bucket_name = bucket_name
        
        # Initialize Vertex AI
        bucket_uri = f"gs://{bucket_name}" if bucket_name else None
        vertexai.init(project=project_id, location=location, staging_bucket=bucket_uri)
        
        # Setup authentication and OpenAI client
        self._setup_openai_client()
        
        # Initialize prompt templates
        self.prompt_MathDial = prompt_MathDial()
        self.prompt_Bridge = prompt_Bridge()
        
        print("Initialized Vertex AI Llama client with OpenAI interface:")
        print(f"  Project ID: {self.project_id}")
        print(f"  Location: {self.location}")
        print(f"  Model: {self.model_name}")
    
    def _setup_openai_client(self):
        """Setup OpenAI client pointing to Vertex AI endpoint"""
        try:
            # Get access token from gcloud CLI
            access_token = get_token()
            print("✓ Successfully obtained access token from gcloud CLI")
            
            # Initialize OpenAI client (pointing to Vertex AI endpoint)
            self.client = openai.OpenAI(
                base_url=f"https://{self.location}-aiplatform.googleapis.com/v1beta1/"
                         f"projects/{self.project_id}/locations/{self.location}/endpoints/openapi/chat/completions?",
                api_key=access_token
            )
            print("✓ OpenAI client initialized successfully")
            
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
            print("\n🔧 Please ensure you are authenticated with gcloud:")
            print("1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
            print("2. Authenticate: gcloud auth login")
            print("3. Set project: gcloud config set project YOUR_PROJECT_ID")
            print("4. Verify: gcloud auth print-access-token")
            raise

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 1.0) -> str:
        """
        Generate response using OpenAI client connected to Vertex AI
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Generated response text
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use OpenAI client to call Vertex AI
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Extract assistant response
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                else:
                    print("Warning: Empty response received")
                    return ""
                    
            except Exception as e:
                error_str = str(e)
                print(f"Error generating response (attempt {attempt + 1}/{max_retries}): {error_str}")
                
                # Check if it's an authentication error
                if "401" in error_str or "UNAUTHENTICATED" in error_str:
                    print("Authentication error detected. Refreshing access token...")
                    try:
                        # Refresh the access token and recreate client
                        self._setup_openai_client()
                        print("✓ Access token refreshed")
                        continue  # Retry with new token
                    except Exception as refresh_error:
                        print(f"Failed to refresh token: {refresh_error}")
                        if attempt == max_retries - 1:
                            return f"Authentication Error: {str(e)}"
                        continue
                
                # For other errors, wait before retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    return f"Error after {max_retries} attempts: {str(e)}"
        
        return "Error: Maximum retries exceeded"

    def load_data(self, dataset_file: str) -> List[Dict[str, Any]]:
        """
        Load dataset from JSON file
        
        Args:
            dataset_file: Path to dataset JSON file
            
        Returns:
            List of data samples
        """
        with open(dataset_file, "r", encoding="utf-8") as fp:
            json_data = json.load(fp)
        print(f"Loaded {len(json_data)} samples from {dataset_file}")
        return json_data

    def load_existing_results(self, output_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load existing results from output file
        
        Args:
            output_file: Path to output file
            
        Returns:
            Dictionary mapping conversation_id+Split to result data
        """
        existing_results = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    key = item['conversation_id'] + item['Split']
                    existing_results[key] = item
                print(f"Loaded {len(existing_results)} existing results from {output_file}")
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")
        return existing_results

    def save_single_result(self, result_data: Dict[str, Any], output_file: str) -> None:
        """
        Save single result to output file (append mode)
        
        Args:
            result_data: Single result to save
            output_file: Path to output file
        """
        # Load existing results
        existing_results = []
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
            except:
                existing_results = []
        
        # Add new result
        existing_results.append(result_data)
        
        # Save back to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)

    def inference(self, args, json_data: List[Dict[str, Any]]) -> None:
        """
        Run inference on dataset with incremental processing
        
        Args:
            args: Command line arguments
            json_data: Dataset to process
        """
        # Load existing results to avoid reprocessing
        existing_results = self.load_existing_results(args.output_file)
        
        # Filter out already processed items
        to_process = []
        for cur_data in json_data:
            key = cur_data['conversation_id'] + cur_data['Split']
            if key not in existing_results:
                to_process.append(cur_data)
        
        print(f"Total samples: {len(json_data)}")
        print(f"Already processed: {len(existing_results)}")
        print(f"To process: {len(to_process)}")
        
        if not to_process:
            print("All samples already processed!")
            return
        
        processed_count = len(existing_results)
        
        for x in tqdm(range(len(to_process)), desc="Processing samples"):
            cur_data = to_process[x]
            
            try:
                # Generate prompt based on data source
                if cur_data['Data'] == "MathDial":
                    messages = self.prompt_MathDial.gen_prompt(cur_data)
                elif cur_data['Data'] == "Bridge":
                    messages = self.prompt_Bridge.gen_prompt(cur_data)
                else:
                    raise ValueError(f"Invalid data source: {cur_data['Data']}")
                
                # Generate response
                response = self.generate_response(messages, max_tokens=1024)
                
                # Store result
                result_data = {
                    'conversation_id': cur_data['conversation_id'],
                    'Split': cur_data['Split'],
                    'prompt': messages,
                    'result': response
                }
                
                # Save immediately
                self.save_single_result(result_data, args.output_file)
                processed_count += 1
                
                print(f"Processed and saved sample {processed_count}/{len(json_data)}: {cur_data['conversation_id']}")
                
                # Sleep to avoid quota / rate limit issues (following t.py pattern)
                if x < len(to_process) - 1:  # Don't sleep after the last item
                    time.sleep(61)
                    
            except Exception as e:
                print(f"Error processing sample {cur_data['conversation_id']}: {e}")
                # Continue with next sample instead of failing completely
                continue
        
        print(f"Inference completed! Total processed: {processed_count}/{len(json_data)}")
        print(f"Results saved to {args.output_file}")


def main():
    """Main function to run inference"""
    parser = ArgumentParser(description="Vertex AI Llama Inference with OpenAI Client")
    parser.add_argument("--dataset_file", type=str, 
                       default="../data/MRBench/MRBench_V1.json",
                       help="Path to dataset JSON file")
    parser.add_argument("--project_id", type=str, default="geminitest-420802",
                       help="Google Cloud Project ID (optional, will try to get from environment)")
    parser.add_argument("--location", type=str, default="us-central1",
                       help="Google Cloud region")
    parser.add_argument("--model_name", type=str, default="meta/llama-3.1-8b-instruct-maas",
                       help="Vertex AI model name")
    parser.add_argument("--bucket_name", type=str, default=None,
                       help="GCS bucket name for staging (optional)")
    parser.add_argument("--output_file", type=str, default="../data/MRBench/MRBench_V1_VertexAI_llama-3.1-8b-instruct-maas.json",
                       help="Output file path (optional, will be auto-generated)")
    
    args = parser.parse_args()
    
    print("="*100)
    print("Vertex AI Llama Inference with OpenAI Client Configuration:")
    print(f"  Dataset file: {args.dataset_file}")
    print(f"  Output file: {args.output_file}")
    print(f"  Project ID: {args.project_id}")
    print(f"  Location: {args.location}")
    print(f"  Model name: {args.model_name}")
    print(f"  Bucket name: {args.bucket_name}")
    print("="*100)
    
    try:
        # Initialize inference client
        inference_client = VertexAILlamaInference(
            project_id=args.project_id,
            location=args.location,
            model_name=args.model_name,
            bucket_name=args.bucket_name
        )
        
        # Load data
        json_data = inference_client.load_data(args.dataset_file)
        
        # Run inference
        inference_client.inference(args, json_data)
        
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
