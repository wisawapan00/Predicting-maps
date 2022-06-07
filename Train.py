import time
import torch
import multiprocessing as mp
from tabulate import tabulate
from torch.utils.data import DataLoader
from pathlib import Path
import torch.optim as optim
from torch.utils.data import RandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import Config
import Augmentation
import Dataset

def main():
    start = time.time()
    best_mIoU = 0.0
    epoch = 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers = mp.cpu_count()

    trainset = eval(dataset_name)(root_dir=dataset_root, split="train", transform=train_augmentation)
    valset = eval(dataset_name)(root_dir=dataset_root, split="val", transform=val_augmentation)

    disc = eval(model_disc)().to(device)
    gen = eval(model_gen)().to(device)

    sampler = RandomSampler(trainset)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)

    iters_per_epoch = len(trainset) // batch_size
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    disc_scaler = GradScaler()
    gen_scaler = GradScaler()

    if model_load:
        checkpoint_gen = torch.load(model_path_gen)
        checkpoint_disc = torch.load(model_path_disc)

        gen.load_state_dict(checkpoint_gen['model_state_dict'])
        disc.load_state_dict(checkpoint_disc['model_state_dict'])

        opt_gen.load_state_dict(checkpoint_gen['optim_state_dict'])
        opt_disc.load_state_dict(checkpoint_disc['optim_state_dict'])

        epoch = checkpoint_gen['epoch']+1
        print("Model successfully loaded!")


    for e in range(epoch, epochs):
        disc.train()
        gen.train()

        D_fake = 0.0
        D_real = 0.0
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch: [{e + 1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {learning_rate:.8f} D_fake: {D_fake:.3f} D_real: {D_real:.3f}")

        for iter, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            
            # Train Discriminator
            with autocast():
                y_fake = gen(x)
                D_real = disc(x, y)
                D_real_loss = loss_fn(D_real, torch.ones_like(D_real))
                D_fake = disc(x, y_fake.detach())
                D_fake_loss = loss_fn(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            disc.zero_grad()
            disc_scaler.scale(D_loss).backward()
            disc_scaler.step(opt_disc)
            disc_scaler.update()

            # Train generator
            with torch.cuda.amp.autocast():
                D_fake = disc(x, y_fake)
                G_fake_loss = loss_fn(D_fake, torch.ones_like(D_fake))
                L1 = l1_loss(y_fake, y) * l1_lambda
                G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            gen_scaler.scale(G_loss).backward()
            gen_scaler.step(opt_gen)
            gen_scaler.update()

            D_real=torch.sigmoid(D_real).mean().item()
            D_fake=torch.sigmoid(D_fake).mean().item()

            pbar.set_description(
                f"Epoch: [{e + 1}/{epochs}] Iter: [{iter + 1}/{iters_per_epoch}] LR: {learning_rate:.8f} D_fake: {D_fake:.3f} D_real: {D_real:.3f}")

        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)

        if e % 50 == 0: save_images(gen, valloader, save_image_path, device)
        save_checkpoint(gen, opt_gen, e, model_path_gen)
        save_checkpoint(disc, opt_disc, e, model_path_disc)


    end = time.gmtime(time.time() - start)

    print('Total Training Time', time.strftime("%H:%M:%S", end))

if __name__ == '__main__':
    main()