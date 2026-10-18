"""Settings shared by the custom trainer and PiRacer driver."""

from pathlib import Path


# Files and folders
project_dir = Path(__file__).resolve().parent
data_dir = Path.home() / "mycar" / "obstacle_varied_data"
model_dir = project_dir / "models"
model_path = model_dir / "custom_obstacle_varied_v2.pt"
car_dir = Path.home() / "mycar"

# Camera input must match the images recorded by the PiRacer.
image_height = 120
image_width = 160
image_channels = 3

# Training
random_seed = 42
batch_size = 128
max_epochs = 100
learning_rate = 1e-3
early_stopping_patience = 7

# Reduce overconfidence in the dominant full-left and center steering bins.
steering_label_smoothing = 0.05
steering_class_weight_power = 0.5
steering_class_weight_max = 2.5

# Driving
# This multiplies the model's learned throttle. Begin conservatively.
ai_throttle_multiplier = 0.35
drive_loop_hz = 20

# Decode the categorical distributions continuously and smooth live commands.
prediction_temperature = 1.0
steering_smoothing_alpha = 0.35
throttle_smoothing_alpha = 0.20
steering_max_step = 0.20

train_size = 0.8
val_size = 0.1
test_size = 0.1
