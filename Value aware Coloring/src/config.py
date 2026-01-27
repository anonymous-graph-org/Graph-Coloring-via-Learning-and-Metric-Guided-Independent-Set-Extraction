# Configurations for training and evaluation

HIDDEN_DIM = 32  # Hidden dimension
NUM_LAYERS = 20  # Deeper model
NUM_MAPS = 32    # Number of probability maps
EPOCHS = 200     # Training epochs
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
TIME_BUDGET = 60  # Time budget for prediction (seconds)
SEED = 42

# Paths
REDUCED_LABELS_FILE = "src/reduced_labels.pickle"
MODEL_SAVE_PATH = "model_parameters/value_aware_gcn.pth"
OUTPUT_FILE = "output_color.txt"
TEST_DATA_PATH = "model_parameters/value_test_data.pickle"
RESULTS_PATH = "model_parameters/value_test_results.pickle"