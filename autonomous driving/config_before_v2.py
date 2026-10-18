"""Settings shared by the custom trainer and PiRacer driver."""

from pathlib import Path


# Files and folders
project_dir = Path(__file__).resolve().parent
data_dir = Path.home() / "mycar" / "new_data"
model_dir = project_dir / "models"
model_path = model_dir / "custom_pilot.h5"
car_dir = Path.home() / "mycar"

# Camera input must match the images recorded by the PiRacer.
image_height = 120
image_width = 160
image_channels = 3

# Training
random_seed = 42
batch_size = 128
max_epochs = 60
learning_rate = 1e-3
early_stopping_patience = 7

# Driving
# This multiplies the model's learned throttle. Begin conservatively.
ai_throttle_multiplier = 0.35
drive_loop_hz = 20

train_size = 0.8
val_size = 0.1
test_size = 0.1
