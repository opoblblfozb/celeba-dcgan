import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
from PIL import Image

from dcgan import get_netD, get_netG
from utils import get_data_loader
import wandb


def get_trainer(
        data_root,
        ngpu,
        lr,
        beta1,
        num_epochs,
        image_size,
        batch_size,
        workers,
        nz,
        nc,
        ndf,
        ngf,
        pjname,
        group):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    netD = get_netD(ngpu=ngpu, device=device, nc=nc, ndf=ndf)
    netG = get_netG(ngpu=ngpu, device=device, nz=nz, ngf=ngf, nc=nc)
    print(workers)
    dataloader = get_data_loader(
        data_root,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers)
    config = {
        "netD": netD,
        "netG": netG,
        "device": device,
        "lr": lr,
        "beta1": beta1,
        "num_epochs": num_epochs,
        "dataloader": dataloader,
        "nz": nz,
        "pjname": pjname,
        "group": group,
    }
    trainer = Trainer(**config)
    return trainer


class Trainer:
    def __init__(
            self,
            netD,
            netG,
            device,
            lr,
            beta1,
            num_epochs,
            dataloader,
            nz,
            pjname,
            group):
        self.netD = netD
        self.netG = netG
        self.device = device
        self.lr = lr
        self.beta1 = beta1
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.nz = nz
        self.fixed_z = torch.randn(64, self.nz, 1, 1, device=self.device)
        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()
        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=self.lr, betas=(
                self.beta1, 0.999))
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.lr, betas=(
                self.beta1, 0.999))

        wandb.init(project=pjname, group=group)
        wandb.watch(self.netD)
        wandb.watch(self.netG)

    def generate_tensor_image(self):
        return self.netG(self.fixed_z)

    def generate_image(self):
        tensor = self.generate_tensor_image()
        arr = np.transpose(
            vutils.make_grid(
                tensor,
                padding=2,
                normalize=True).cpu())
        return Image.fromarray(arr)

    def train_netD(self, data):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        self.netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(self.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), self.real_label,
                           dtype=torch.float, device=self.device)
        # Forward pass real batch through D
        output = self.netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with
        # previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()
        return errD

    def train_netG(self, data):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        # Format batch
        real_cpu = data[0].to(self.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), self.real_label,
                           dtype=torch.float, device=self.device)
        # fake labels are real for generator cost
        label = torch.full((b_size,), self.real_label,
                           dtype=torch.float, device=self.device)
        # Since we just updated D, perform another forward pass of all-fake
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        # Generate fake image batch with G
        fake = self.netG(noise)
        # batch through D
        output = self.netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()
        return errG

    def train(self):

        # Create batch of latent vectors that we will use to visualize
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                errD = self.train_netD(data)
                errG = self.train_netG(data)
                image = self.generate_image()
                wandb.log({"errD": errD, "errG": errG,
                          "image": wandb.Image(image)})
