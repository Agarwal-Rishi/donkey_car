from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config as settings

from model import (
    AutonomousDriver,
    steering_bins,
    throttle_bins,
    image_height,
    image_width,
    throttle_max,
)


model_path = settings.model_path


class CustomPilot:
    def __init__(self, checkpoint_path=None):
        self.model = AutonomousDriver()
        self.last_steering = None
        self.last_throttle = None
        if checkpoint_path is not None:
            self.load(checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.last_steering = None
        self.last_throttle = None

        print(f"Loaded best epoch: {checkpoint['epoch'] + 1}")
        print(f"Validation loss: {checkpoint['val_loss']}")

    def run(self, image_array):
        image = Image.fromarray(image_array).convert("RGB")
        image = image.resize((image_width, image_height))

        image_array = np.array(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            steering_logits, throttle_logits = self.model(image_tensor)

        temperature = settings.prediction_temperature
        steering_probabilities = torch.softmax(
            steering_logits / temperature, dim=1
        )
        throttle_probabilities = torch.softmax(
            throttle_logits / temperature, dim=1
        )

        steering_values = torch.linspace(-1.0, 1.0, steering_bins)
        throttle_values = torch.linspace(0.0, throttle_max, throttle_bins)
        raw_steering = torch.sum(
            steering_probabilities[0] * steering_values
        ).item()
        raw_throttle = torch.sum(
            throttle_probabilities[0] * throttle_values
        ).item()

        if self.last_steering is None:
            steering = raw_steering
            throttle = raw_throttle
        else:
            steering = (
                (1.0 - settings.steering_smoothing_alpha) * self.last_steering
                + settings.steering_smoothing_alpha * raw_steering
            )
            steering = float(np.clip(
                steering,
                self.last_steering - settings.steering_max_step,
                self.last_steering + settings.steering_max_step,
            ))
            throttle = (
                (1.0 - settings.throttle_smoothing_alpha) * self.last_throttle
                + settings.throttle_smoothing_alpha * raw_throttle
            )

        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, throttle_max))

        self.last_steering = steering
        self.last_throttle = throttle

        return steering, throttle


if __name__ == "__main__":
    images_dir = settings.data_dir / "images"
    image_path = next(images_dir.glob("*.jpg"))

    image_array = np.array(Image.open(image_path).convert("RGB"))

    pilot = CustomPilot(model_path)
    steering, throttle = pilot.run(image_array)

    print(f"Image: {image_path.name}")
    print(f"Predicted steering: {steering}")
    print(f"Predicted throttle: {throttle}")
