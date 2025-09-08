from ctransformers import AutoModelForCausalLM
import os

# Configuration

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_TYPE = "mistral"

# ---------------------

def check_model_file(path):
    """Checks if the model file exists and is not empty."""
    if not os.path.exists(path):
        print(f"ERROR: Model file not found at '{path}'")
        return False
    if os.path.getsize(path) < 1024 * 1024: # Less than 1MB
        print(f"ERROR: Model file at '{path}' is too small. It might be corrupted or a placeholder.")
        return False
    print(f"SUCCESS: Model file found at '{path}' and has a valid size.")
    return True

def run_test():
    """Runs the isolated test to load the LLM."""
    if not check_model_file(MODEL_PATH):
        return

    print("\nAttempting to load the model directly with ctransformers...")
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=MODEL_PATH,
            model_type=MODEL_TYPE,
            gpu_layers=0
        )
        print("\n--- SUCCESS ---")
        print("The LLM was loaded successfully!")
        print("This means the ctransformers library and the model file are both working correctly.")
        print("The problem is likely a hidden conflict with LangChain.")
    except Exception as e:
        print("\n--- FAILURE ---")
        print("The LLM failed to load even with a direct call.")
        print(f"Error details: {e}")
        print("\nThis suggests one of two problems:")
        print("1. The model file itself is corrupted or incomplete.")
        print("2. There is a deep installation issue with the ctransformers library (e.g., missing system dependencies).")

if __name__ == "__main__":
    run_test()