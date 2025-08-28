import torch
import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import structural_similarity   as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import mean_squared_error      as mse_metric

def compute_metrics(pred, gt):
    pred_np = pred.squeeze().cpu().numpy()
    gt_np   = gt.squeeze().cpu().numpy()
    
    mse  = mse_metric (gt_np, pred_np)
    psnr = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim = ssim_metric(gt_np, pred_np, data_range=1.0)
    return mse, psnr, ssim

def evaluate_results(model, dataloader, device):
    mse_list, psnr_list, ssim_list = [], [], []
    for batch in dataloader:
        imgs = batch.to(device)
        
        with torch.no_grad():
            pred, _ = model(imgs)
        
        for j in range(pred.size(0)):
            mse, psnr, ssim = compute_metrics(pred[j], imgs[j])
            mse_list.append(mse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        return

    print(f'Average MSE:  {np.mean(mse_list):.6f}')
    print(f'Average PSNR: {np.mean(psnr_list):.2f}')
    print(f'Average SSIM: {np.mean(ssim_list):.4f}')
    
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
            img    = sample.unsqueeze(0).float().to(device)
                    
            # Use model to get prediction
            img_recon, _ = model(img)

            im0 =axes[i,0].imshow(format_image(img),        cmap='gray',vmin=0,vmax=1)
            axes[i,0].axis('off')
            plt.colorbar(im0, ax=axes[i,0])
            im1 =axes[i,1].imshow(format_image(img_recon),  cmap='gray',vmin=0,vmax=1)
            axes[i,1].axis('off')
            plt.colorbar(im1, ax=axes[i,1])
            
        fig.tight_layout(pad=0.25)
        plt.savefig(f'./results/image_{name}.jpg')
        plt.close()