import os
import numpy as np
# import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import utils
from tqdm import tqdm
from utils.formatter import format_best_checkpoint_name, format_current_checkpoint_name
from piq import ssim

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def transform_perceptual(img):
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.repeat(1, 3, 1, 1)
    img = transform(img)
    return img

def get_perceptual(λ_perceptual, epoch, percpt_scale):
    warmup = 1/percpt_scale
    if epoch < warmup:
        return λ_perceptual * percpt_scale * epoch 
    else:
        return λ_perceptual
    
    
# def show_imgs(images, titles=["Input", "Output", "Label"]):
#     plt.figure(figsize=(20,20))
#     for ind, img in enumerate(images):
#         plt.subplot(1, len(images), ind+1)
#         plt.imshow(utils.make_grid(images[ind]).cpu().numpy().transpose((1,2,0)))
#         plt.title(titles[ind])
#     plt.show()

def train(model, optimizer, dataloader, losses, λ_loss):
    model.train()
    total_loss = 0.0
    total_losses = {}
    for l in losses:
        total_losses[l] = 0.0
    
    for data in dataloader:
        # Get images
        images, labels = data["image"].float().cuda(), data["label"].float().cuda()

        optimizer.zero_grad()
        output = model(images)
        
        # Calculate loss
        loss      = 0
        this_loss = {}
        for l in losses:
            this_loss[l] = losses[l](output, labels)
            loss += this_loss[l] * λ_loss[l]
            
        loss.backward()
        optimizer.step()
        
        # Store losses
        total_loss  += loss.item() 
        for l in losses:
            total_losses[l] += this_loss[l].item()
    
    total_loss /= len(dataloader)
    for l in losses:
        total_losses[l] /= len(dataloader)
    
    return total_loss, total_losses

def evaluate(model, dataloader, losses, λ_loss):
    model.eval()
    total_loss = 0.0
    total_losses = {}
    for l in losses:
        total_losses[l] = 0.0
    
    with torch.no_grad():
        for data in dataloader:
            # Get images
            images, labels = data["image"].float().cuda(), data["label"].float().cuda()
            output = model(images)

            # Calculate loss
            loss      = 0
            this_loss = {}
            for l in losses:
                this_loss[l] = losses[l](output, labels)
                loss += this_loss[l] * λ_loss[l]

            # Store losses
            total_loss  += loss.item() 
            for l in losses:
                total_losses[l] += this_loss[l].item()    

    total_loss /= len(dataloader)
    for l in losses:
        total_losses[l] /= len(dataloader)
    
    return total_loss, total_losses

def train_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, n_epochs, name, losses, λ_loss):
    best_loss = np.inf
    
    for epoch in range(n_epochs):
        print('\nEpoch ', epoch)
        print('Learning rate :', get_lr(optimizer))
        
        ###### Training ######
        train_total, train_all = train(model, optimizer, train_dataloader, losses, λ_loss)
        output = f"Training Loss: {train_total:.4f}. "
        for loss in train_all:
            output += f"{loss}: {train_all[loss]:.4f}, "
        print(output)

        ###### Evaluation ######
        val_total, val_all = evaluate(model, test_dataloader, losses, λ_loss)
        output = f"Validation Loss: {val_total:.4f}. "
        for loss in val_all:
            output += f"{loss}: {val_all[loss]:.4f}, "
        print(output)

        if val_total < best_loss:
            best_loss = val_total
            torch.save(model.state_dict(), name + '_best.pth')
            print('Best model saved at', name + '_best.pth')

        torch.save(model.state_dict(), name + '_current.pth')
        scheduler.step(val_total)
        
    return model
