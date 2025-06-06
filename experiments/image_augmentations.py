from typing import Union, Literal, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

CIFAR_IMAGE_DIMENSIONS = (32, 32, 3)

# These should match your global transform setup
resize_size = (224, 224)
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_with_feature_extraction(model_name: str = "resnet18") -> nn.Module:
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        return_nodes = {"avgpool": "feat"}
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        return_nodes = {"avgpool": "feat"}
    else:
        raise ValueError("Only resnet18 and resnet50 are supported.")
    model = model.to(device).eval()
    extractor = create_feature_extractor(model, return_nodes=return_nodes)
    return extractor

def compute_embedding(model: nn.Module, img_tensor: torch.Tensor) -> torch.Tensor:
    feat = model(img_tensor)["feat"]
    if feat.ndim > 2:
        feat = torch.flatten(feat, start_dim=1)
    return feat  # Shape: [1, d]


def compute_jacobian(model: nn.Module, img: Union[np.ndarray, Image.Image]) -> torch.Tensor:
    # Step 1: Prepare original image tensor (32x32)
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8)) if img.max() <= 1 else Image.fromarray(img)
    img = img.resize((32, 32))
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, C]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, 32, 32]
    img_tensor.requires_grad_(True)

    # Step 2: Resize + normalize inside autograd graph
    resized = torch.nn.functional.interpolate(img_tensor, size=resize_size, mode='bilinear', align_corners=False)
    transformed = normalize(resized.squeeze(0)).unsqueeze(0)

    # Step 3: Forward through embedding model
    embedding = compute_embedding(model, transformed)
    d = embedding.shape[1]
    I = img_tensor.numel()

    # Step 4: Compute Jacobian
    J = torch.zeros(d, I, device=device)
    for i in range(d):
        grad = torch.autograd.grad(embedding[0, i], img_tensor, retain_graph=True)[0]
        J[i] = grad.flatten()

    return J  # Shape: [d, 3*32*32]


def compute_max_eigvec(J: torch.Tensor, Q: torch.Tensor, l2_norm: float) -> torch.Tensor:
    M = J.T @ Q @ J
    eigvals, eigvecs = torch.linalg.eigh(M)
    v = eigvecs[:, -1]
    return (v / v.norm()) * l2_norm  # L2-normalized perturbation


def compute_pseudoinverse_delta(J: torch.Tensor, target_vec: torch.Tensor) -> torch.Tensor:
    # Solves J @ delta = target_vec using least squares
    # torch.linalg.lstsq expects (A, B) to solve A @ X ≈ B
    solution = torch.linalg.lstsq(J, target_vec.unsqueeze(1))
    delta = solution.solution.squeeze()
    return delta

def compute_regularized_delta(J: torch.Tensor, target_vec: torch.Tensor, l2_lambda: float=0.1) -> torch.Tensor:
    """
    Solve (JᵀJ + λI)⁻¹ Jᵀ target_vec for delta (ridge regression)
    """
    JT = J.T
    I = torch.eye(J.shape[1], device=J.device)
    delta = torch.linalg.solve(JT @ J + l2_lambda * I, JT @ target_vec)
    return delta


def augment_image_embedding(
        model_name: str = "resnet18",
        img: Union[np.ndarray, Image.Image] = None,
        Q: Union[np.ndarray, torch.Tensor] = None,
        mode: Literal["max", "pseudo", "both"] = "max",
        l2_norm: float = 1.0,
        target_vec: Optional[torch.Tensor] = None
) -> dict:
    """
    Compute a perturbation delta for an input image that either:
    - Maximizes the Q-norm of the change in embedding space (mode='max')
    - Matches a user-specified vector in embedding space (mode='pseudo')
    - Computes both at once (mode='both')

    Args:
        model_name: 'resnet18' or 'resnet50'
        img: np.ndarray or PIL.Image (RGB image)
        Q: PSD matrix of shape [d, d]
        mode: One of ['max', 'pseudo', 'both']
        l2_norm: L2 norm to scale perturbation (used in 'max' or 'both')
        target_vec: Embedding space direction to reproduce (used in 'pseudo' or 'both')

    Returns:
        dict: Keys can include 'delta_max', 'delta_pseudo', 'jacobian'
    """
    model = load_model_with_feature_extraction(model_name)
    # img_tensor = preprocess_image(img)
    Q = torch.tensor(Q, dtype=torch.float32).to(device) if isinstance(Q, np.ndarray) else Q.to(device)
    target_vec = target_vec.to(device) if target_vec is not None else None

    J = compute_jacobian(model, img)

    results = {'jacobian': J}
    if mode in ["max", "both"]:
        delta_max = compute_max_eigvec(J, Q, l2_norm)
        results['delta_max'] = delta_max.reshape(CIFAR_IMAGE_DIMENSIONS)
    if mode in ["pseudo", "both"]:
        if target_vec is None:
            raise ValueError("target_vec must be provided in 'pseudo' or 'both' mode.")
        delta_pseudo = compute_regularized_delta(J, target_vec)
        results['delta_pseudo'] = delta_pseudo.reshape(CIFAR_IMAGE_DIMENSIONS)

    return results


def plot_cifar_image(img: np.ndarray, title: str = None):
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def save_cifar_image(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)


def load_cifar_image_from_file(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def add_tennis_ball(img: np.ndarray, portion: float = 0.05) -> np.ndarray:
    """
    Adds a yellow circle (~5% area) near the top-right corner.
    """
    output = img.copy()
    h, w = output.shape[:2]
    radius = int(np.sqrt(portion * h * w / np.pi))  # area ~5% of total
    center = (w - radius - 2, radius + 2)  # top-right with padding

    # OpenCV uses BGR, not RGB
    yellow_bgr = (0, 255, 255)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.circle(output_bgr, center, radius, yellow_bgr, -1)
    return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)


def merge_top_right_patch(img: np.ndarray, donor_img: np.ndarray, pct: float = 0.1) -> np.ndarray:
    """
    Overwrite top-right ~10% of pixels using donor image.
    """
    output = img.copy()
    h, w = output.shape[:2]
    patch_area = int(pct * h * w)
    patch_size = int(np.sqrt(patch_area))

    # Compute patch boundaries
    x0 = w - patch_size
    y0 = 0
    output[y0:y0 + patch_size, x0:x0 + patch_size] = donor_img[y0:y0 + patch_size, x0:x0 + patch_size]
    return output


def average_with_image(img: np.ndarray, donor_img: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Returns a weighted average of original and donor image.
    alpha = weight of donor image.
    """
    return np.clip((1 - alpha) * img + alpha * donor_img, 0, 255).astype(np.uint8)


def add_high_freq_noise(img: np.ndarray, epsilon: float = 10.0, seed: int = None) -> np.ndarray:
    """
    Add small high-frequency perturbations in the Fourier domain.
    """
    rng = np.random.default_rng(seed)

    img_f = np.fft.fft2(img, axes=(0, 1))
    perturb = (rng.standard_normal(img.shape) + 1j * rng.standard_normal(img.shape)) * epsilon
    perturbed = img_f + perturb
    img_noisy = np.fft.ifft2(perturbed, axes=(0, 1)).real
    return np.clip(img_noisy, 0, 255).astype(np.uint8)


def add_low_rank_perturbation(img: np.ndarray, rank: int = 1, scale: float = 10.0, seed: int = None) -> np.ndarray:
    """
    Adds a low-rank matrix (outer product) across all channels.
    """
    rng = np.random.default_rng(seed)

    h, w, c = img.shape
    u = rng.standard_normal((h, rank))
    v = rng.standard_normal((w, rank))
    low_rank = (u @ v.T)[..., None].repeat(c, axis=-1) * scale
    perturbed = img.astype(float) + low_rank
    return np.clip(perturbed, 0, 255).astype(np.uint8)
