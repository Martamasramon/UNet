import torch
import matplotlib.pyplot as plt
import numpy as np

def format_image(img):
    return np.squeeze((img).cpu().numpy())

def visualize_results(model, dataset, device, name, batch_size=5, seed=1):
    fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(3*2,3*batch_size))
    axes[0,0].set_title('Original (input)')
    axes[0,1].set_title('Reconstructed (output)')
    np.random.seed(seed)
    indices = np.random.choice(np.arange(len(dataset)),batch_size,replace=False)

    model.eval()
    with torch.no_grad():
        for i,ind in enumerate(indices):
            sample = dataset[ind]
            img    = sample["image"].unsqueeze(0).float().to(device)
                    
            # Use model to get prediction
            img_recon = model(img)

            im0 =axes[i,0].imshow(format_image(img),        cmap='gray',vmin=0,vmax=1)
            axes[i,0].axis('off')
            plt.colorbar(im0, ax=axes[i,0])
            im1 =axes[i,1].imshow(format_image(img_recon),  cmap='gray',vmin=0,vmax=1)
            axes[i,1].axis('off')
            plt.colorbar(im1, ax=axes[i,1])
            
        fig.tight_layout(pad=0.25)
        plt.savefig(f'./results/image_{name}.jpg')
        plt.close()