import os

ORIG_INPUT_DATASET = 'food11'
BASE_PATH = 'dataset'

# directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

#  list of class label names
CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles/Pasta",
    "Rice", "Seafood",
    "Soup",
    "Vegetable/Fruit"
]

# set the batch size when fine-tuning
BATCH_SIZE = 16

# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "food11.model"])

# define the path to the output training history plots
UNFROZEN_PLOT_PATH = os.path.sep.join(["output", "unfrozen.png"])
WARMUP_PLOT_PATH = os.path.sep.join(["output", "warmup.png"])