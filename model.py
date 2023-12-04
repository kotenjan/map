import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import itertools
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np
import json
import os
from device import Device
from generator import Generator
from discriminator import Discriminator
from dataset import SatelliteToMapDataset
from buffer import ImageBuffer
from torch.nn.functional import conv2d
from torch.autograd import Variable
from math import exp


class GAN:
    def __init__(self, dataset_root_dir, val_dataset_root_dir, batch_size=2, current_epoch=0, num_epochs=1000, lr=0.0001, betas=(0.5, 0.999), resize=256, input_channels=3, output_channels=3):
        
        if current_epoch > 0:
            with open("models/errors.json") as f:
                self.errors = json.load(f)
            with open("models/val_errors.json") as f:
                self.val_errors = json.load(f)
            self.best_val_loss = min(self.val_errors.values(), key=lambda x: x['val_loss'])['val_loss']
            self.best_ssim_loss = max(self.val_errors.values(), key=lambda x: x['ssim'])['ssim']
            self.wait = self.val_errors[max(self.val_errors.keys())]['wait']
            print(f"self.wait: {self.wait}")
        else:
            self.errors = {}
            self.val_errors = {}    
            self.best_val_loss = np.inf
            self.best_ssim_loss = 0
            self.wait = 0
            print(f"self.wait: {self.wait}")
            
        self.device = Device().device
        self.val_dataset = SatelliteToMapDataset(root_dir=val_dataset_root_dir, resize=resize)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.patience = 16  # Number of epochs to wait for improvement before stopping
        self.dataset = SatelliteToMapDataset(root_dir=dataset_root_dir, resize=resize)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.window = self.create_window(11, 3).to(self.device)
        self.G_A = Generator(input_channels, output_channels).to(self.device)
        self.G_B = Generator(input_channels, output_channels).to(self.device)
        self.D_A = Discriminator(input_channels).to(self.device)
        self.D_B = Discriminator(input_channels).to(self.device)
        self.optimizer_G = optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=lr, betas=betas)
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=lr, betas=betas)
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=lr, betas=betas)
        self.scaler = GradScaler()
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch
        self.fake_A_buffer = ImageBuffer()
        self.fake_B_buffer = ImageBuffer()
        os.makedirs('plots', exist_ok=True)
        
        if self.current_epoch > 0:
            self.load_models(self.current_epoch)
        
    # Adversarial Loss
    def adversarial_loss(self, output, target_is_real):
        target = torch.ones_like(output) if target_is_real else torch.zeros_like(output)
        return nn.MSELoss()(output, target)

    # Cycle Consistency Loss
    def cycle_consistency_loss(self, real_image, reconstructed_image, lambda_cycle=10):
        return lambda_cycle * nn.L1Loss()(real_image, reconstructed_image)

    # Identity Loss (Optional)
    def identity_loss(self, real_image, same_image, lambda_identity=5):
        return lambda_identity * nn.L1Loss()(real_image, same_image)
    
    def load_models(self, epoch):
        self.G_A.load_state_dict(torch.load(f'all_models/G_A_epoch_{epoch - 1}.pth'))
        self.G_B.load_state_dict(torch.load(f'all_models/G_B_epoch_{epoch - 1}.pth'))
        self.D_A.load_state_dict(torch.load(f'all_models/D_A_epoch_{epoch - 1}.pth'))
        self.D_B.load_state_dict(torch.load(f'all_models/D_B_epoch_{epoch - 1}.pth'))
        print(f"Loaded models from epoch {epoch}")

    def save_plots(self, real_A, real_B, fake_A, fake_B, epoch, batch_index):
        # Function to convert tensor to numpy image
        def to_numpy(tensor):
            # Normalize to [0, 1] and then scale to [0, 255]
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            return image

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        # Normalize and plot real_A
        image = to_numpy(real_A)
        image = (image - image.min()) / (image.max() - image.min())
        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Real A")
        axs[0, 0].axis('off')

        # Normalize and plot fake_A
        image = to_numpy(fake_A)
        image = (image - image.min()) / (image.max() - image.min())
        axs[0, 1].imshow(image)
        axs[0, 1].set_title("AI Generated A From B")
        axs[0, 1].axis('off')

        # Normalize and plot real_B
        image = to_numpy(real_B)
        image = (image - image.min()) / (image.max() - image.min())
        axs[1, 0].imshow(image)
        axs[1, 0].set_title("Real B")
        axs[1, 0].axis('off')

        # Normalize and plot fake_B
        image = to_numpy(fake_B)
        image = (image - image.min()) / (image.max() - image.min())
        axs[1, 1].imshow(image)
        axs[1, 1].set_title("AI Generated B From A")
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'plots/epoch_{epoch}_batch_{batch_index}.png')
        plt.savefig(f'progress.png')
        plt.close()
        
    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    # SSIM computation method
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        if img1.size(1) != channel:
            raise ValueError('Input images must have the same number of channels')

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def validate(self, dataloader):
        self.G_A.eval()
        self.G_B.eval()
        total_loss_G = 0
        total_ssim = 0

        for i, batch in enumerate(dataloader):
            real_A = batch['satellite_image'].to(self.device)
            real_B = batch['map_image'].to(self.device)

            with torch.no_grad():
                fake_B = self.G_A(real_A)
                ssim_val = self.ssim(fake_B, real_B, window=self.window, window_size=11, channel=3, size_average=True)

                # Identity loss
                loss_identity_A = self.identity_loss(self.G_B(real_A), real_A)

                # GAN loss
                loss_GAN_A2B = self.adversarial_loss(self.D_B(fake_B), True)  # True for real label

                # Cycle consistency loss
                reconstructed_A = self.G_B(fake_B)
                loss_cycle_ABA = self.cycle_consistency_loss(real_A, reconstructed_A)

                # Total generator loss
                total_loss_G += (loss_GAN_A2B + loss_cycle_ABA + loss_identity_A)
                total_ssim += ssim_val

        avg_loss_G = total_loss_G / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        return float(avg_loss_G), float(avg_ssim)

    def train(self):
        
        for epoch in range(self.current_epoch, self.num_epochs):
            
            epoch_errors_g = {}
            epoch_errors_d_a = {}
            epoch_errors_d_b = {}
            
            for i, batch in enumerate(self.dataloader):
                print(f"Epoch: {epoch}")
                print(f"    Batch: {i}")

                real_A = batch['satellite_image'].to(self.device)
                real_B = batch['map_image'].to(self.device)

                ######################
                # Update Generators
                ######################
                self.optimizer_G.zero_grad()

                # Generate fake images
                fake_B = self.G_A(real_A)
                fake_A = self.G_B(real_B)
                
                # Plot the output of the current batch
                self.save_plots(real_A[0], real_B[0], fake_A[0], fake_B[0], epoch, i)

                # Identity loss
                loss_identity_A = self.identity_loss(self.G_B(real_A), real_A)
                loss_identity_B = self.identity_loss(self.G_A(real_B), real_B)

                # GAN loss
                loss_GAN_A2B = self.adversarial_loss(self.D_B(fake_B), True)  # True for real label
                loss_GAN_B2A = self.adversarial_loss(self.D_A(fake_A), True)  # True for real label

                # Cycle consistency loss
                reconstructed_A = self.G_B(fake_B)
                reconstructed_B = self.G_A(fake_A)
                loss_cycle_ABA = self.cycle_consistency_loss(real_A, reconstructed_A)
                loss_cycle_BAB = self.cycle_consistency_loss(real_B, reconstructed_B)

                # Total generator loss
                total_loss_G = (loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_identity_A + loss_identity_B)

                print(f"    Total Generator Loss: {total_loss_G}")
                
                epoch_errors_g[i] = {
                    "loss_GAN_A2B": float(loss_GAN_A2B),
                    "loss_GAN_B2A": float(loss_GAN_B2A),
                    "loss_cycle_ABA": float(loss_cycle_ABA),
                    "loss_cycle_BAB": float(loss_cycle_BAB),
                    "loss_identity_A": float(loss_identity_A),
                    "loss_identity_B": float(loss_identity_B),
                }

                total_loss_G.backward()
                self.optimizer_G.step()

                ######################
                # Update Discriminators
                ######################
                self.optimizer_D_A.zero_grad()
                self.optimizer_D_B.zero_grad()

                # Real loss
                real_loss_D_A = self.adversarial_loss(self.D_A(real_A), True)
                real_loss_D_B = self.adversarial_loss(self.D_B(real_B), True)

                # Fake loss
                fake_A = self.fake_A_buffer.get_images(fake_A)
                fake_B = self.fake_B_buffer.get_images(fake_B)

                fake_loss_D_A = self.adversarial_loss(self.D_A(fake_A.detach()), False)
                fake_loss_D_B = self.adversarial_loss(self.D_B(fake_B.detach()), False)

                # Total loss for discriminators
                total_loss_D_A = (real_loss_D_A + fake_loss_D_A) / 2
                total_loss_D_B = (real_loss_D_B + fake_loss_D_B) / 2
                
                print(f"    Total Discriminator A (Satellite to Map) Loss: {total_loss_D_A}")
                print(f"    Total Discriminator B (Map to Satellite) Loss: {total_loss_D_B}")
                
                epoch_errors_d_a[i] = float(total_loss_D_A)
                epoch_errors_d_b[i] = float(total_loss_D_B)

                total_loss_D_A.backward()
                total_loss_D_B.backward()

                self.optimizer_D_A.step()
                self.optimizer_D_B.step()

                # Free up memory
                del real_A, real_B, fake_A, fake_B, reconstructed_A, reconstructed_B
                
                self.errors[epoch] = {"G": epoch_errors_g, "D_A": epoch_errors_d_a, "D_B": epoch_errors_d_b}
                with open('models/errors.json', 'w') as f:
                    json.dump(self.errors, f)
                
            val_loss, ssim_loss = self.validate(self.val_dataloader)
            print(f"Validation Loss: {val_loss}")
            print(f"SSIM Loss: {ssim_loss}")
            print(f"Wait: {self.wait}")
            
            
            torch.save(self.G_A.state_dict(), f'all_models/G_A_epoch_{epoch}.pth')
            torch.save(self.G_B.state_dict(), f'all_models/G_B_epoch_{epoch}.pth')
            torch.save(self.D_A.state_dict(), f'all_models/D_A_epoch_{epoch}.pth')
            torch.save(self.D_B.state_dict(), f'all_models/D_B_epoch_{epoch}.pth')
            print(f"Saved models for epoch {epoch}")

            # Check for improvement
            if val_loss < self.best_val_loss or ssim_loss > self.best_ssim_loss:
                self.best_val_loss = min(val_loss, self.best_val_loss)
                self.best_ssim_loss = max(ssim_loss, self.best_ssim_loss)
                self.wait = 0  # reset wait
                print(f"self.wait: {self.wait}")
                torch.save(self.G_A.state_dict(), 'models/best_G_A.pth')
                torch.save(self.G_B.state_dict(), 'models/best_G_B.pth')
                torch.save(self.D_A.state_dict(), 'models/best_D_A.pth')
                torch.save(self.D_B.state_dict(), 'models/best_D_B.pth')
            else:
                self.wait += 1
                print(f"self.wait: {self.wait}")
                if self.wait >= self.patience:
                    print("Early stopping triggered")
                    break
                
            self.val_errors[epoch] = {"ssim": ssim_loss, "val_loss": val_loss, "wait": self.wait}
            with open('models/val_errors.json', 'w') as f:
                json.dump(self.val_errors, f)
                
            print("==================================================================\n\n")
                
            torch.cuda.empty_cache()
