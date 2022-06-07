model_load = True
model_gen = "Generator"
model_path_gen = "output/Pix2Pix/gen.pth.tar"
model_disc = "Discriminator"
model_path_disc = "output/Pix2Pix/disc.pth.tar"

dataset_name = "Map"
dataset_root = "DATASETS/maps"

save_image_path = "Test/maps"

image_size = [256, 256]
batch_size = 16
epochs = 500
l1_lambda = 100
learning_rate = 2e-4
