import os
import sys
import torch
import glob
import zipfile

# --- Configuration ---
# You can change these values

# 1. Name your model/experiment. This will be the name of your final files.
experiment_name = "my_voice_model"

# 2. Set the path to your dataset folder.
dataset_path = "dataset"

# 3. Training Parameters
total_training_epochs = 150  # For 10-15 mins of audio, 150-200 is a good range.
save_every_epoch = 10      # How often to save a checkpoint.
batch_size = 8             # Decrease this if you get "CUDA out of memory" errors.
# --- End of Configuration ---


def run_preprocessing():
    """
    Prepares the dataset for training.
    """
    print("\n>>> Step 1: Starting dataset preprocessing...")
    
    from infer.modules.train.preprocess import preprocess_trainset
    
    sample_rate = 40000
    output_logs_path = "logs"
    num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if not os.path.exists(output_logs_path):
        os.makedirs(output_logs_path)
        
    preprocess_trainset(dataset_path, sample_rate, num_processes, output_logs_path, experiment_name)
    
    print(f"\n✅ Preprocessing complete! Processed files are in 'logs/{experiment_name}'.")

def run_training():
    """
    Trains the RVC model.
    """
    print("\n>>> Step 2: Starting model training...")
    
    from infer.modules.train.train import click_train
    
    rvc_version = "v2"
    pitch_extraction_algorithm = "rmvpe"
    sample_rate = 40000
    
    # Check for available GPUs
    gpu_ids = "0"
    if not torch.cuda.is_available():
        print("WARNING: No NVIDIA GPU detected. Training will be extremely slow on CPU.")
        gpu_ids = "" # Fallback to CPU if no GPU
    
    click_train(
        experiment_name,
        sample_rate,
        rvc_version,
        save_every_epoch,
        total_training_epochs,
        batch_size,
        True,  # Save only the latest checkpoint
        gpu_ids,
        pitch_extraction_algorithm,
        False, # Don't cache training set to GPU
        False  # Don't save small final model
    )
    
    print("\n✅ Training finished!")

def run_index_training():
    """
    Builds the feature index file.
    """
    print("\n>>> Step 3: Building the feature index...")
    
    from infer.modules.train.train import train_index
    
    rvc_version = "v2"
    log_path = os.path.join("logs", experiment_name)
    
    train_index(log_path, rvc_version)
    
    print("\n✅ Feature index built successfully!")

def package_model():
    """
    Finds the final model files and zips them for easy use.
    """
    print("\n>>> Step 4: Packaging your final model...")
    
    # Find the trained model .pth file
    pth_files = glob.glob(f"weights/{experiment_name}*.pth")
    if not pth_files:
        print("❌ Could not find the trained .pth file. Check logs for errors.")
        return
        
    latest_pth_file = max(pth_files, key=os.path.getctime)
    
    # Find the trained .index file
    index_files = glob.glob(f"logs/{experiment_name}/*.index")
    if not index_files:
        print("❌ Could not find the trained .index file. Check logs for errors.")
        return
        
    latest_index_file = index_files[0]
    
    # Create a zip file with both model files
    zip_filename = f"{experiment_name}_RVC_v2_model.zip"
    print(f"Creating zip file: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(latest_pth_file, os.path.basename(latest_pth_file))
        zipf.write(latest_index_file, os.path.basename(latest_index_file))
        
    print(f"\n✅ Model packaged successfully into '{zip_filename}'!")
    print("You can find your final model in this zip file in your project folder.")


if __name__ == "__main__":
    # Add the project directory to the Python path to allow imports
    sys.path.append(os.getcwd())
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print(f"❌ Error: The '{dataset_path}' folder is missing or empty.")
        print("Please create it and place your .wav audio files inside.")
    else:
        run_preprocessing()
        run_training()
        run_index_training()
        package_model()