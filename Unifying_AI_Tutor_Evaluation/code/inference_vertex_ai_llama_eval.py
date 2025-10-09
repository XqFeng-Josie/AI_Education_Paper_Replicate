import os
import json
import numpy as np
import random
import re
import sys
import time
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Dict, Any

# Google Cloud and OpenAI imports
import vertexai
import openai
from google.auth import default

from prompt.prompt_llm_eval import gen_eval_prompt

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# # Google Colab authentication (only if running inside Colab)
# if "google.colab" in sys.modules:
#     from google.colab import auth
#     auth.authenticate_user()

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

class VertexAILlamaEvalInference:
    def __init__(self, project_id: str = None, location: str = "us-central1", 
                 model_name: str = "meta/llama3-405b-instruct-maas", bucket_name: str = None):
        """
        Initialize Vertex AI Llama evaluation inference client using OpenAI-compatible interface
        
        Args:
            project_id: Google Cloud Project ID. If None, will try to get from environment
            location: Google Cloud region (default: us-central1)
            model_name: Model name in Vertex AI (default: meta/llama3-405b-instruct-maas)
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
        
        print("Initialized Vertex AI Llama evaluation client with OpenAI interface:")
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

    def extract_feedback_and_result(self, case_text: str) -> Dict[str, Any]:
        """
        Extract the single paragraph containing 'Feedback:' and return {'feedback': str, 'number': int|None}
        
        Args:
            case_text: Generated response text
            
        Returns:
            Dictionary with feedback text and extracted number
        """
        m = re.search(r'Feedback:\s*([\s\S]*?)(?=\n\s*Score:|\Z)', case_text, flags=re.IGNORECASE)
        if not m:
            return {"feedback": case_text.strip(), "number": None}
        
        fb_part = m.group(1).strip()

        num = None
        for pat in (
            r'\[RESULT\]\s*(\d+)',        # [RESULT] 1
            r'\[Score\s*(\d+)\]',        # [Score 3]
            r'\[\s*(\d+)\]',             # [2]
            r'(\d+)\.?\s*$',             # tail number 3 or 1.
        ):
            mm = re.search(pat, fb_part, flags=re.IGNORECASE)
            if mm:
                num = int(mm.group(1))
                break
        
        if num is None:
            mm = re.search(r'Score:\s*(\d+)', case_text, flags=re.IGNORECASE)
            if mm:
                num = int(mm.group(1))

        # Clean the feedback text: remove all the marked form
        cleaned = fb_part
        cleaned = re.sub(r'\[Score\s*\d+\]\.?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[RESULT\]\s*\d+\.?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[\s*\d+\]\.?\s*', '', cleaned)
        cleaned = re.sub(r'\n\s*Score:\s*\d+\s*$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+\d{1,2}\.?\s*$', '', cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.rstrip('.,;:')

        if cleaned == "" or num is None:
            print("*"*100)
            print("Warning: Could not extract proper feedback or score:")
            print(case_text)
            print("*"*100)

        return {"feedback": cleaned, "number": num}

    def load_data(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load dataset from JSON file
        
        Args:
            dataset_path: Path to dataset JSON file
            
        Returns:
            List of data samples
        """
        with open(dataset_path, "r", encoding="utf-8") as fp:
            json_data = json.load(fp)
        print(f"Loaded {len(json_data)} samples from {dataset_path}")
        return json_data

    def save_evaluation_results(self, json_data: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save evaluation results to file
        
        Args:
            json_data: Data with evaluation results
            output_file: Path to output file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    def inference(self, args, json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run evaluation inference on dataset with incremental processing
        
        Args:
            args: Command line arguments
            json_data: Dataset to process
            
        Returns:
            Processed data with evaluation results
        """
        rubric_key = f'eval_{args.rubric_name}_result'
        
        # Filter out already processed items for this rubric
        to_process = []
        already_processed = 0
        
        for cur_data in json_data:
            if rubric_key not in cur_data:
                to_process.append(cur_data)
            else:
                already_processed += 1
        
        print(f"Total samples: {len(json_data)}")
        print(f"Already processed for {args.rubric_name}: {already_processed}")
        print(f"To process: {len(to_process)}")
        
        if not to_process:
            print(f"All samples already processed for {args.rubric_name}!")
            return json_data
        
        processed_count = already_processed
        
        for x in tqdm(range(len(to_process)), desc=f"Evaluating {args.rubric_name}"):
            cur_data = to_process[x]
            
            try:
                messages = gen_eval_prompt(cur_data, args.rubric_name)
                
                # Generate response
                response = self.generate_response(messages, max_tokens=1024)
                
                # Extract feedback and score
                feedback_and_result = self.extract_feedback_and_result(response)
                
                # Add evaluation result to current data
                cur_data[rubric_key] = {
                    "result": response,
                    "feedback": feedback_and_result['feedback'],
                    "number": feedback_and_result['number']
                }
                processed_count += 1
                
                # Save results immediately after each processing
                self.save_evaluation_results(json_data, args.output_file)
                
                print(f"Processed and saved {args.rubric_name} for sample {processed_count}/{len(json_data)}: {cur_data.get('conversation_id', 'unknown')}")
                
                # Sleep to avoid quota / rate limit issues (following t.py pattern)
                if x < len(to_process) - 1:  # Don't sleep after the last item
                    time.sleep(61)
                    
            except Exception as e:
                print(f"Error processing sample for {args.rubric_name}: {e}")
                # Continue with next sample instead of failing completely
                continue
        
        print(f"Evaluation for {args.rubric_name} completed! Total processed: {processed_count}/{len(json_data)}")
        return json_data

    def inference_original(self, args, json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run evaluation on original model responses with incremental processing
        
        Args:
            args: Command line arguments
            json_data: Dataset to process
            
        Returns:
            Processed data with evaluation results
        """
        total_evaluations = 0
        to_process_evaluations = 0
        already_processed_evaluations = 0
        
        # Count total evaluations needed and already processed
        for cur_data in json_data:
            for model_name, value in cur_data['anno_llm_responses'].items():
                total_evaluations += 1
                if ("annotation_eval" in value and 
                    args.rubric_name in value['annotation_eval']):
                    already_processed_evaluations += 1
                else:
                    to_process_evaluations += 1
        
        print(f"Total evaluations: {total_evaluations}")
        print(f"Already processed for {args.rubric_name}: {already_processed_evaluations}")
        print(f"To process: {to_process_evaluations}")
        
        if to_process_evaluations == 0:
            print(f"All evaluations already processed for {args.rubric_name}!")
            return json_data
        
        processed_count = already_processed_evaluations
        
        for x in tqdm(range(len(json_data)), desc=f"Evaluating original responses for {args.rubric_name}"):
            cur_data = json_data[x]
            
            for model_name, value in cur_data['anno_llm_responses'].items():
                # Skip if already processed
                if ("annotation_eval" in value and 
                    args.rubric_name in value['annotation_eval']):
                    continue
                
                try:
                    response = value['response']
                    temp_data = {
                        "conversation_history": cur_data['conversation_history'],
                        "result": response,
                    }
                    messages = gen_eval_prompt(temp_data, args.rubric_name)
                    
                    # Generate evaluation response
                    eval_response = self.generate_response(messages, max_tokens=1024)
                    
                    # Extract feedback and score
                    feedback_and_result = self.extract_feedback_and_result(eval_response)
                    
                    # Add evaluation result
                    if "annotation_eval" not in value:
                        value['annotation_eval'] = {}
                    value['annotation_eval'][args.rubric_name] = {
                        "result": eval_response,
                        "feedback": feedback_and_result['feedback'],
                        "number": feedback_and_result['number']
                    }
                    processed_count += 1
                    
                    # Save results immediately after each processing
                    self.save_evaluation_results(json_data, args.output_file)
                    
                    print(f"Processed and saved {args.rubric_name} for {model_name} on sample {processed_count}/{total_evaluations}: {cur_data.get('conversation_id', 'unknown')}")
                    
                    # Sleep to avoid quota / rate limit issues (following t.py pattern)
                    if processed_count < total_evaluations:  # Don't sleep after the last item
                        time.sleep(61)
                        
                except Exception as e:
                    print(f"Error processing evaluation for {model_name} on {args.rubric_name}: {e}")
                    # Continue with next evaluation instead of failing completely
                    continue
        
        print(f"Original evaluation for {args.rubric_name} completed! Total processed: {processed_count}/{total_evaluations}")
        return json_data


def main():
    """Main function to run evaluation inference"""
    parser = ArgumentParser(description="Vertex AI Llama Evaluation Inference with OpenAI Client")
    parser.add_argument("--dataset_file", type=str, 
                       default="../data/MRBench/MRBench_V1_Mistral-7B-Instruct-v0.1.json",
                       help="Path to dataset JSON file")
    parser.add_argument("--mode", type=str, default="generate", 
                       choices=["generate", "original"],
                       help="Mode: 'generate' for new responses, 'original' for existing responses")
    parser.add_argument("--project_id", type=str, default="geminitest-420802",
                       help="Google Cloud Project ID (optional, will try to get from environment)")
    parser.add_argument("--location", type=str, default="us-central1",
                       help="Google Cloud region")
    parser.add_argument("--model_name", type=str, default="meta/llama-3.1-8b-instruct-maas",
                       help="Vertex AI model name")
    parser.add_argument("--bucket_name", type=str, default=None,
                       help="GCS bucket name for staging (optional)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path (optional, will be auto-generated)")
    
    args = parser.parse_args()
    
    # Auto-generate output file name if not provided
    if args.output_file is None:
        input_file = args.dataset_file.split("/")[-1].replace(".json", "")
        output_dir = os.path.dirname(args.dataset_file)
        args.output_file = os.path.join(output_dir, f"{input_file}_vertex_ai_llama_eval.json")
    
    print("="*100)
    print("Vertex AI Llama Evaluation Inference with OpenAI Client Configuration:")
    print(f"  Dataset file: {args.dataset_file}")
    print(f"  Output file: {args.output_file}")
    print(f"  Mode: {args.mode}")
    print(f"  Project ID: {args.project_id}")
    print(f"  Location: {args.location}")
    print(f"  Model name: {args.model_name}")
    print(f"  Bucket name: {args.bucket_name}")
    print("="*100)
    
    try:
        # Initialize evaluation inference client
        eval_client = VertexAILlamaEvalInference(
            project_id=args.project_id,
            location=args.location,
            model_name=args.model_name,
            bucket_name=args.bucket_name
        )
        
        if args.mode == "generate":
            # Load MRBench data for conversation history
            MRBench_V1_data = eval_client.load_data("../data/MRBench/MRBench_V1.json")
            conversation_history_map = {
                data['conversation_id'] + data['Split']: data['conversation_history'] 
                for data in MRBench_V1_data
            }
            
            # Load dataset
            json_data = eval_client.load_data(args.dataset_file)
            for data in json_data:
                data['conversation_history'] = conversation_history_map[data['conversation_id'] + data['Split']]
            
            # Define evaluation rubrics
            # rubric_name_list = [
            #     "mistake_identification", "mistake_location", "revealing_of_the_answer", 
            #     "providing_guidance", "coherence", "actionability", "tutor_tone", "humanlikeness"
            # ]
            rubric_name_list = [
                "providing_guidance"
            ]
            
            # Run evaluation for each rubric
            for rubric_name in rubric_name_list:
                print("*"*100)
                print(f"Evaluating {rubric_name}...")
                print("*"*100)
                args.rubric_name = rubric_name
                json_data = eval_client.inference(args, json_data)
            
            print("Evaluation completed.")
            
        else:
            # Original mode - evaluate existing responses
            MRBench_V1_data = eval_client.load_data(args.dataset_file)
            
            # Define evaluation rubrics
            rubric_name_list = [
                "mistake_identification", "mistake_location", "revealing_of_the_answer",
                "providing_guidance", "coherence", "actionability", "tutor_tone", "humanlikeness"
            ]
            
            # Run evaluation for each rubric
            for rubric_name in rubric_name_list:
                print("*"*100)
                print(f"Evaluating {rubric_name}...")
                print("*"*100)
                args.rubric_name = rubric_name
                MRBench_V1_data = eval_client.inference_original(args, MRBench_V1_data)
            
            print("Original evaluation completed.")
        
    except Exception as e:
        print(f"Error during evaluation inference: {e}")
        raise


if __name__ == "__main__":
    main()
