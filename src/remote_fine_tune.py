import os
import json
import time
from together import Together
# The specific types like FineTuningJob and FileType are often accessed via 
# the client or are part of the main 'together' package in modern SDK versions.
# We only need the core Together client import.

# --- CONFIGURATION ---
# The Together SDK automatically looks for the TOGETHER_API_KEY environment variable.
# We will check for it explicitly to provide a clear error message if it's missing.

# Together AI model identifier for Qwen 3 4B
BASE_MODEL = "Qwen-3-4B"

# Local path to your training dataset
# Using os.path.join for cross-platform compatibility
TRAIN_SET_PATH = os.path.join(
    'C:', 
    os.sep, 
    'Users', 
    'brian', 
    'Documents', 
    'Random', 
    'scibowl-gpt', 
    'data', 
    'train', 
    'mit_ess.jsonl'
)

# A descriptive name for your fine-tuning job
JOB_NAME = "qwen3-4b-scibowl-ess-11182025"

def run_fine_tuning_job():
    """
    Handles the entire fine-tuning workflow: upload, job creation, and monitoring.
    """
    # 1. Check for API Key in environment
    together_api_key = os.environ.get("TOGETHER_API_KEY")
    if not together_api_key:
        print("CRITICAL ERROR: The TOGETHER_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        print("For instructions, see the chat response below.")
        return

    # Initialize the client using the environment variable
    TOGETHER_CLIENT = Together(api_key=together_api_key)

    print("--- Together AI Fine-Tuning Service ---")
    print(f"Target Model: {BASE_MODEL}")
    print(f"Dataset Path: {TRAIN_SET_PATH}\n")

    # 2. Upload the training file
    print(f"1. Uploading dataset file: {os.path.basename(TRAIN_SET_PATH)}...")
    try:
        # FIX: Changed 'file_type' to 'purpose' with the standard value 'fine-tune' 
        # to resolve the 'unexpected keyword argument' error.
        training_file = TOGETHER_CLIENT.files.upload(
            file=TRAIN_SET_PATH,
            purpose="fine-tune"
        )
        file_id = training_file.id
        print(f"   -> Upload successful. File ID: {file_id}")
    except Exception as e:
        print(f"CRITICAL ERROR during file upload: {e}")
        return

    # 3. Create the Fine-Tuning Job
    print("2. Creating fine-tuning job...")
    try:
        job = TOGETHER_CLIENT.fine_tuning.create(
            training_file=file_id, 
            model=BASE_MODEL,
            hyperparameters={
                "n_epochs": 3,
                "batch_size": 2,
                "learning_rate": "2e-5"
            }
        )
        job_id = job.id
        # Together's Python SDK includes a utility function for blocking until completion
        print(f"   -> Job created successfully. Job ID: {job_id}")
        print(f"   -> Fine-tuned model name will be: {job.fine_tuned_model_name}")
    except Exception as e:
        print(f"CRITICAL ERROR creating job: {e}")
        return

    # 4. Monitor the Job Status
    print("\n3. Monitoring job status (Polling every 30 seconds)...")
    
    # Custom monitoring loop using together.fine_tuning.retrieve for a better
    # feedback loop than the blocking wait_for_job method.
    while True:
        try:
            status = TOGETHER_CLIENT.fine_tuning.retrieve(job_id=job_id)
            job_status = status.status.value
            
            if job_status in ["completed", "error", "cancelled"]:
                break
            
            # Print a concise status update
            print(f"   - Status: {job_status.capitalize()} (Step: {status.steps.value if status.steps else 'N/A'})")
            time.sleep(30) # Wait 30 seconds before next poll

        except Exception as e:
            print(f"Error during status retrieval: {e}")
            time.sleep(30)
    
    final_job_status = TOGETHER_CLIENT.fine_tuning.retrieve(job_id=job_id)
    
    print("\n--- JOB FINISHED ---")
    print(f"Job ID: {final_job_status.id}")
    print(f"Status: {final_job_status.status.value}")
    
    if final_job_status.status.value == "completed":
        print(f"SUCCESS! Your fine-tuned model is available under the name:")
        print(f"Model Name: {final_job_status.fine_tuned_model_name}")
        print("\nUse this model name in the Together API for inference.")
    else:
        print(f"Job failed. Reason: {final_job_status.error.message if final_job_status.error else 'Unknown error.'}")
        
    print("\n--- Cleanup ---")
    print(f"File ID {file_id} will be retained for 30 days unless deleted manually.")


if __name__ == "__main__":
    run_fine_tuning_job()