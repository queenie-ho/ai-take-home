import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
import os

# Import model classes
from src.models.mnist_gan_model import MNISTGANModel
from src.models.modules.generators import Generator
from src.models.modules.discriminators import Discriminator


def load_trained_model(checkpoint_path: str, device: str = 'cpu') -> MNISTGANModel:
    """
    Load a trained GAN model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded MNISTGANModel
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    hparams = checkpoint['hyper_parameters']
    
    # Create generator and discriminator with saved hyperparameters
    generator = Generator(
        n_classes=hparams.get('n_classes', 10),
        latent_dim=hparams.get('latent_dim', 100),
        channels=hparams.get('channels', 1),
        img_size=hparams.get('img_size', 32)
    )
    
    discriminator = Discriminator(
        n_classes=hparams.get('n_classes', 10),
        channels=hparams.get('channels', 1),
        img_size=hparams.get('img_size', 32)
    )
    
    # Filter out generator and discriminator from hparams to avoid conflicts
    filtered_hparams = {k: v for k, v in hparams.items() 
                       if k not in ['generator', 'discriminator']}
    
    # Create the model
    model = MNISTGANModel(
        generator=generator,
        discriminator=discriminator,
        **filtered_hparams
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully on {device}")
    return model


def generate_images(model: MNISTGANModel, 
                   num_samples: int = 10, 
                   specific_digits: list = None,
                   device: str = 'cpu') -> tuple:
    """
    Generate images using the trained GAN.
    
    Args:
        model: Trained GAN model
        num_samples: Number of images to generate per digit
        specific_digits: List of specific digits to generate (0-9). If None, generates all digits
        device: Device to run inference on
    
    Returns:
        Tuple of (generated_images, labels)
    """
    model.eval()
    
    if specific_digits is None:
        specific_digits = list(range(10))  # Generate all digits 0-9
    
    generated_images = []
    labels = []
    
    with torch.no_grad():
        for digit in specific_digits:
            for _ in range(num_samples):
                # Generate random noise
                z = torch.randn(1, model.hparams.latent_dim, device=device)
                
                # Create label tensor
                label = torch.tensor([digit], device=device)
                
                # Generate image
                fake_img = model.generator(z, label)
                
                generated_images.append(fake_img.squeeze().cpu().numpy())
                labels.append(digit)
    
    return np.array(generated_images), np.array(labels)


def save_images_grid(images: np.ndarray, 
                    labels: np.ndarray, 
                    save_path: str = "generated_images.png",
                    grid_size: tuple = None):
    """
    Save generated images in a grid format.
    
    Args:
        images: Array of generated images
        labels: Array of corresponding labels
        save_path: Path to save the image grid
        grid_size: Tuple (rows, cols) for grid layout. If None, auto-calculated
    """
    num_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = min(10, num_images)
        rows = (num_images + cols - 1) // cols
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle("Generated MNIST Images", fontsize=16)
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot images
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        
        if i < num_images:
            # Denormalize image (from [-1, 1] to [0, 1])
            img = (images[i] + 1) / 2
            img = np.clip(img, 0, 1)
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f"Digit: {labels[i]}")
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image grid saved to {save_path}")
    plt.close()


def save_individual_images(images: np.ndarray, 
                          labels: np.ndarray, 
                          output_dir: str = "generated_individual"):
    """
    Save each generated image as a separate PNG file.
    
    Args:
        images: Array of generated images
        labels: Array of corresponding labels
        output_dir: Directory to save individual images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Denormalize image (from [-1, 1] to [0, 1])
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        # Convert to 8-bit image
        img_8bit = (img * 255).astype(np.uint8)
        
        # Save as PNG
        img_pil = Image.fromarray(img_8bit, mode='L')
        filename = f"digit_{label}_sample_{i:03d}.png"
        img_pil.save(os.path.join(output_dir, filename))
    
    print(f"Individual images saved to {output_dir}/")


def test_discriminator_confidence(model: MNISTGANModel, 
                                 images: np.ndarray, 
                                 labels: np.ndarray, 
                                 device: str = 'cpu'):
    """
    Test discriminator confidence on generated images.
    
    Args:
        model: Trained GAN model
        images: Generated images
        labels: Corresponding labels
        device: Device to run inference on
    """
    model.eval()
    
    confidences = []
    
    with torch.no_grad():
        for img, label in zip(images, labels):
            # Convert to tensor
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
            label_tensor = torch.tensor([label], device=device)
            
            # Get discriminator output
            validity = model.discriminator(img_tensor, label_tensor)
            confidence = torch.sigmoid(validity).item()
            confidences.append(confidence)
    
    avg_confidence = np.mean(confidences)
    print(f"\nDiscriminator Analysis:")
    print(f"Average confidence on generated images: {avg_confidence:.4f}")
    print(f"Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
    
    return confidences


def main():
    parser = argparse.ArgumentParser(description="Test trained MNIST GAN model")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to the trained model checkpoint (.ckpt file)")
    parser.add_argument("--output_dir", "-o", type=str, default="test_output",
                       help="Output directory for generated images")
    parser.add_argument("--num_samples", "-n", type=int, default=5,
                       help="Number of samples to generate per digit")
    parser.add_argument("--digits", "-d", type=int, nargs='+', default=None,
                       help="Specific digits to generate (e.g., --digits 0 1 2)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--grid_only", action="store_true",
                       help="Save only grid image, not individual images")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load the trained model
        model = load_trained_model(args.checkpoint, device)
        
        # Generate images
        print(f"\nGenerating images...")
        images, labels = generate_images(
            model, 
            num_samples=args.num_samples,
            specific_digits=args.digits,
            device=device
        )
        sqrt_size = int(np.sqrt(images.shape[1]))
        save_imgs = images.reshape(int(images.shape[0]), sqrt_size, sqrt_size)
                
        # Save images as grid
        grid_path = os.path.join(args.output_dir, "generated_grid.png")
        save_images_grid(save_imgs, labels, grid_path)
                
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())