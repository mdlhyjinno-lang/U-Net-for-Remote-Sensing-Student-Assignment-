import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def calculate_metrics(mask):
    """Calculate evaluation metrics for the mask"""
    metrics = {
        'mask_area_ratio': np.sum(mask) / (mask.shape[0] * mask.shape[1]),
        'mask_pixel_count': int(np.sum(mask)),
        'mask_height': mask.shape[0],
        'mask_width': mask.shape[1],
        'mask_mean': float(np.mean(mask)),
        'mask_std': float(np.std(mask)),
        'mask_max': float(np.max(mask)),
        'mask_min': float(np.min(mask))
    }
    return metrics


def create_overlay_image(img, mask):
    """Create an overlay image with original image and mask"""
    # Convert PIL Image to numpy array
    img_np = np.array(img)
    
    # Create mask with transparency
    mask_rgb = np.zeros_like(img_np)
    mask_rgb[mask == 1] = [255, 0, 0]  # Red color for mask
    
    # Convert to PIL Images
    img_pil = Image.fromarray(img_np)
    mask_pil = Image.fromarray(mask_rgb)
    
    # Create overlay
    overlay = img_pil.copy()
    overlay.paste(mask_pil, (0, 0), Image.fromarray((mask * 128).astype(np.uint8)))  # 50% transparency
    
    return overlay


def create_result_image(img, mask, overlay):
    """Create a combined result image with original, mask, and overlay"""
    # Get dimensions
    width, height = img.size
    
    # Create new image with 3 columns
    result = Image.new('RGB', (width * 3, height))
    
    # Paste images
    result.paste(img, (0, 0))
    result.paste(mask_to_image(mask), (width, 0))
    result.paste(overlay, (width * 2, 0))
    
    # Add labels
    draw = ImageDraw.Draw(result)
    labels = ['Original', 'Mask', 'Overlay']
    for i, label in enumerate(labels):
        draw.text((width * i + 10, 10), label, fill=(255, 255, 255))
    
    return result


def plot_three_images(img, mask, overlay):
    """Plot three images: original, mask, and overlay"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].set_title('Original Image')
    ax[0].imshow(img)
    ax[0].axis('off')
    
    ax[1].set_title('Black and White Mask')
    ax[1].imshow(mask, cmap='gray')
    ax[1].axis('off')
    
    ax[2].set_title('Overlay Image')
    ax[2].imshow(overlay)
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory with timestamp subfolder
    output_dir = f"prediction_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        original_img = img.copy()

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        # Create overlay image
        overlay = create_overlay_image(img, mask)
        
        # Create combined result image
        result_img = create_result_image(img, mask, overlay)
        
        # Calculate metrics
        metrics = calculate_metrics(mask)
        metrics['image_file'] = fn
        metrics['model_file'] = args.model
        metrics['scale_factor'] = args.scale
        metrics['threshold'] = args.mask_threshold
        metrics['timestamp'] = timestamp
        
        # Get base filename without path
        base_fn = os.path.basename(fn)
        name_without_ext = os.path.splitext(base_fn)[0]
        
        # Save individual images
        if not args.no_save:
            # Original image
            original_path = os.path.join(output_dir, f"{name_without_ext}_original_{timestamp}.jpg")
            original_img.save(original_path)
            
            # Mask image (black and white)
            mask_path = os.path.join(output_dir, f"{name_without_ext}_mask_{timestamp}.jpg")
            mask_img = mask_to_image(mask)
            mask_img.save(mask_path)
            
            # Overlay image
            overlay_path = os.path.join(output_dir, f"{name_without_ext}_overlay_{timestamp}.jpg")
            overlay.save(overlay_path)
            
            # Combined result image
            result_path = os.path.join(output_dir, f"{name_without_ext}_result_{timestamp}.jpg")
            result_img.save(result_path)

            logging.info(f"Original image saved to {original_path}")
            logging.info(f"Mask saved to {mask_path}")
            logging.info(f"Overlay saved to {overlay_path}")
            logging.info(f"Combined result saved to {result_path}")
        
        # Save metrics to txt file
        metrics_path = os.path.join(output_dir, f"{name_without_ext}_metrics_{timestamp}.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Prediction Metrics for {fn}\n")
            f.write(f"=" * 50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"Metrics saved to {metrics_path}")
        
        # Print metrics to console
        print(f"\n=== Metrics for {fn} ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("=" * 50)

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_three_images(img, mask, overlay)
