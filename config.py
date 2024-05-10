import os

csv_path = os.path.join(
    "CodingChallenge_v2", "car_imgs_4000.csv")
images_dir = os.path.join(
    "CodingChallenge_v2", "imgs")

train_chunk = 0.7
valid_chunk = 0.15
test_chunk = 0.15

batch_size = 4
epochs = 50
model_checkpoint = "last_chkpt.pth"