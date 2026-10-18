from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import (
    AutonomousDriver,
    steering_bins,
    throttle_bins,
    image_height,
    image_width,
    throttle_max,
)


model_path = Path(__file__).parent / "models" / "custom_cat_v1.pt"


class CustomPilot:
    def __init__(self, checkpoint_path=None):
        self.model = AutonomousDriver()
        if checkpoint_path is not None:
            self.load(checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

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

        steering_bin = torch.argmax(steering_logits, dim=1).item()
        throttle_bin = torch.argmax(throttle_logits, dim=1).item()

        steering = -1.0 + steering_bin * 2.0 / (steering_bins - 1)
        throttle = throttle_bin * throttle_max / (throttle_bins - 1)

        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, throttle_max))

        return steering, throttle


if __name__ == "__main__":
    images_dir = Path.home() / "mycar" / "new_data" / "images"
    image_path = next(images_dir.glob("*.jpg"))

    image_array = np.array(Image.open(image_path).convert("RGB"))

    pilot = CustomPilot(model_path)
    steering, throttle = pilot.run(image_array)

    print(f"Image: {image_path.name}")
    print(f"Predicted steering: {steering}")
    print(f"Predicted throttle: {throttle}")
