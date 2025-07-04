import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
AUX_WEIGHT = 0.3

# Early stopping
PATIENCE = 4

# Data paths
TRAIN_DIR = "/content/dataset/Comys_Hackathon5/Task_A/train"
VAL_DIR = "/content/dataset/Comys_Hackathon5/Task_A/val"
TEST_DIR = "/content/dataset/Comys_Hackathon5/Task_A/test"

# Model paths
BEST_MODEL_PATH = "checkpoint_best.pth"
FINAL_MODEL_PATH = "TASK_A_MODEL.pth"
RESULTS_FILE = "results.csv"

print(f"Using device: {DEVICE}")