import json
import numpy as np
from PIL import Image
import config
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import config as settings
from torch.utils.data import DataLoader, Dataset
from model import AutonomousDriver, steering_bins, throttle_bins, throttle_max

class DrivingDataset(Dataset):
    def __init__(self, images, actions):
        self.images = images
        self.actions = actions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        image = image.permute(2, 0, 1).float() / 255.0

        angle = self.actions[index, 0]
        throttle = self.actions[index, 1]

        steering_bin = round((angle + 1.0) * (steering_bins - 1) / 2.0)
        throttle_bin = round(throttle * (throttle_bins - 1) / throttle_max)

        steering_bin = max(0, min(steering_bins - 1, steering_bin))
        throttle_bin = max(0, min(throttle_bins - 1, throttle_bin))

        return (
            image,
            torch.tensor(steering_bin, dtype=torch.long),
            torch.tensor(throttle_bin, dtype=torch.long),
        )

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

tub = config.data_dir
images = []
actions = []

for catalog in sorted(tub.glob("catalog_*.catalog")):
    for line in catalog.read_text().splitlines():
        record = json.loads(line)
        image_path = tub / "images" / record["cam/image_array"]
        images.append(np.array(Image.open(image_path)))
        actions.append([record["user/angle"], record["user/throttle"]])

images = np.array(images)
actions = np.array(actions, dtype=np.float32)

train_end = int(len(images) * settings.train_size)
val_end = train_end + int(len(images) * settings.val_size)
test_end = val_end + int(len(images) * settings.test_size)

train_images = images[:train_end]
train_actions = actions[:train_end]
val_images = images[train_end:val_end]
val_actions = actions[train_end:val_end]
test_images = images[val_end:test_end]
test_actions = actions[val_end:test_end]

print(f"train images: {len(train_images)}")
print(f"train actions: {len(train_actions)}")
print(f"val images: {len(val_images)}")
print(f"val actions: {len(val_actions)}")
print(f"test images: {len(test_images)}")
print(f"test actions: {len(test_actions)}")

train_dataset = DrivingDataset(train_images, train_actions)
val_dataset = DrivingDataset(val_images, val_actions)
test_dataset = DrivingDataset(test_images, test_actions)

train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True)

print(f"train loader: {len(train_loader)}")
print(f"val loader: {len(val_loader)}")
print(f"test loader: {len(test_loader)}")


model = AutonomousDriver()

optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
save_path = Path("models/custom_cat_v1.pt")
save_path.parent.mkdir(exist_ok=True)

for epoch in range(settings.max_epochs):
    model.train()
    for images, steering_targets, throttle_targets in train_loader:
        optimizer.zero_grad()

        steering_logits, throttle_logits = model(images)

        steering_loss = criterion(steering_logits, steering_targets)
        throttle_loss = criterion(throttle_logits, throttle_targets)
        loss = steering_loss + throttle_loss

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


    model.eval()
    total_val_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, steering_targets, throttle_targets in val_loader:
            steering_logits, throttle_logits = model(images)

            steering_loss = criterion(steering_logits, steering_targets)
            throttle_loss = criterion(throttle_logits, throttle_targets)
            loss = steering_loss + throttle_loss

            batch_size = images.size(0)
            total_val_loss += loss.item() * batch_size
            total_samples += batch_size

    average_val_loss = total_val_loss / total_samples
    print(f"Validation loss: {average_val_loss}")

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": best_val_loss,
        }, save_path)

        print(f"Saved new best model to {save_path}")


checkpoint = torch.load(save_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

total_test_loss = 0.0
total_test_samples = 0

with torch.no_grad():
    for images, steering_targets, throttle_targets in test_loader:
        steering_logits, throttle_logits = model(images)

        steering_loss = criterion(steering_logits, steering_targets)
        throttle_loss = criterion(throttle_logits, throttle_targets)
        loss = steering_loss + throttle_loss

        batch_size = images.size(0)
        total_test_loss += loss.item() * batch_size
        total_test_samples += batch_size

average_test_loss = total_test_loss / total_test_samples

print(f"Loaded best epoch: {checkpoint['epoch'] + 1}")
print(f"Best validation loss: {checkpoint['val_loss']}")
print(f"Test loss: {average_test_loss}")
