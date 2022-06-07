import torch
from torchvision.utils import save_image
from tqdm import tqdm
import Config
from Augmentation import train_augmentation, val_augmentation
from Generator import Generator
from Dataset import Map

def save_images(gen, valloader, save_image_path, device):
    gen.eval()
    num = 0

    with torch.no_grad():
        for x, y in tqdm(valloader):
            num += 1
            x, y = x.to(device), y.to(device)
            y_fake = gen(x)
            x = x * 0.5 + 0.5
            y_fake = y_fake * 0.5 + 0.5
            save_image(torch.cat([x, y, y_fake], 0), f"{save_image_path}/{num}.png") # remove normalization
    gen.train()

def save_checkpoint(gen, opt_gen, e, model_path_gen):
    torch.save({
            'model_state_dict': gen.state_dict(),
            'optim_state_dict': opt_gen.state_dict(),
            'epoch': e,
        }, model_path_gen)


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gen = eval(model_gen)().to(device)
    valset = eval(dataset_name)(root_dir=dataset_root, split="val", transform=val_augmentation)
    valloader = DataLoader(valset, batch_size=1, shuffle=False)
    save_images(gen, valloader, save_image_path, device)