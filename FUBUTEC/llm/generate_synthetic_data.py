"""
Main script: Generate synthetic student data using LLM
"""
import sys
import os
import argparse
import logging
import pandas as pd
from pathlib import Path

# Add llm_data_generation module to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_data_generation import OpenRouterClient, StudentDataGenerator, DataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_callback(current, total):
    """Progress callback"""
    if current % 10 == 0 or current == total:
        print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic student data using LLM')
    parser.add_argument(
        '--n_students',
        type=int,
        default=1000,
        help='Number of students to generate (default: 1000)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='OpenRouter API key (if not provided, will read from environment variable OPENROUTER_API_KEY)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/llama-3.3-70b-instruct:free',
        help='Model name to use (default: meta-llama/llama-3.3-70b-instruct:free)'
    )
    parser.add_argument(
        '--fallback_model',
        type=str,
        default='meta-llama/llama-3.3-70b-instruct',
        help='Fallback model name if primary model fails (default: meta-llama/llama-3.3-70b-instruct)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='../data/student-por.csv',
        help='Original data path (default: ../data/student-por.csv)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='../data/student-por-synthetic.csv',
        help='Output file path (default: ../data/student-por-synthetic.csv)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume generation from existing file (skip already generated records)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate data quality after generation'
    )
    
    args = parser.parse_args()
    
    # Check original data file
    if not os.path.exists(args.data_path):
        logger.error(f"Original data file does not exist: {args.data_path}")
        return 1
    
    # Load original data
    logger.info(f"Loading original data: {args.data_path}")
    original_data = pd.read_csv(args.data_path, sep=';')
    logger.info(f"Original data shape: {original_data.shape}")
    
    # Initialize LLM client
    logger.info(f"Connecting to OpenRouter API, model: {args.model}")
    if args.fallback_model:
        logger.info(f"Fallback model: {args.fallback_model}")
    try:
        llm_client = OpenRouterClient(
            api_key=args.api_key, 
            model=args.model,
            fallback_model=args.fallback_model
        )
    except ValueError as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return 1
    
    if not llm_client.is_healthy():
        logger.error("OpenRouter API is not available, please check API key and network connection")
        return 1
    
    # Initialize data generator
    logger.info("Initializing data generator...")
    generator = StudentDataGenerator(llm_client, original_data)
    
    # Check resume status
    if args.resume:
        existing_records, existing_count = generator.load_existing_records(args.output_path)
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing records, will resume generation")
        else:
            logger.info("No existing records found, starting fresh generation")
    
    # Generate data with real-time saving and resume support
    logger.info(f"Starting to generate {args.n_students} synthetic records...")
    logger.info(f"Real-time saving enabled: records will be saved immediately to {args.output_path}")
    logger.info("Rate limiting: 5 seconds between requests")
    logger.info("Error retry: 1 minute sleep after errors")
    
    records = generator.generate_batch(
        args.n_students, 
        progress_callback=progress_callback,
        output_path=args.output_path,
        resume=args.resume
    )
    
    if not records:
        logger.error("Failed to generate any data")
        return 1
    
    logger.info(f"Generation completed. Total records: {len(records)}")
    logger.info(f"All records have been saved to: {args.output_path}")
    
    # Validate data (if requested)
    if args.validate:
        logger.info("Validating generated data...")
        synthetic_data = pd.read_csv(args.output_path, sep=';')
        validator = DataValidator(original_data)
        results = validator.validate(synthetic_data)
        validator.print_validation_report(results)
    
    logger.info("Completed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

