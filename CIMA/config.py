"""
Configuration file for CIMA experiment
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_PATH = os.path.join(DATA_DIR, "dataset.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
DEV_PATH = os.path.join(DATA_DIR, "dev.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# Labels
LABELS = ["Guess", "Question", "Affirmation", "Other"]
LABEL_MAPPING = {i: label for i, label in enumerate(LABELS)}

# Split ratios
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# API Keys (set these as environment variables or replace with your keys)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-BDZkUNPtZ- QPMfJG55WbNkmus7IupVQt -T9olUe0RlovaZwRWF-In- owr6EZoyNFe1ITk9mkvXT3BlbkFJnl y1NOa5FbBGd4yHXjh3vnJHKWBYfnqS FB0cq47RMO1nPt2kVHjJ919RIMesQV aPgOsTpKMN8A")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")  # For Llama-3.1-405B

# Model configurations
MODEL_CONFIGS = {
    "gpt-4": {
        "name": "gpt-4",
        "api_type": "openai",
        "temperature": 0,
        "max_tokens": 10,
    },
    "llama-3.1-405b": {
        "name": "meta-llama/llama-3.1-405b-instruct",
        "api_type": "openrouter",  # or "openai" if using OpenAI-compatible endpoint
        "api_base": "https://openrouter.ai/api/v1",
        "temperature": 0,
        "max_tokens": 10,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "api_type": "local",  # Local deployment
        "device": "cuda",  # or "cpu"
        "temperature": 0,
        "max_tokens": 10,
        "load_in_8bit": False,  # Set to True for 8-bit quantization
    },
}

# Inference backend selection
MISTRAL_LOCAL_PATH = None  # Set to local model path if you have downloaded weights

# Prompts
GPT4_SYSTEM_PROMPT = """You're observing a student learning Italian prepositions.
Classify their response into one out of 4 categories: [Guess, Question, Affirmation, Other].
Only return the label corresponding to one of the four categories."""

GPT4_USER_PROMPT = "Utterance: {utterance}"

MISTRAL_PROMPT = """Scenario: You're observing a student learning Italian prepositions.
Student Utterance: {utterance}
Student Action: {label}"""
