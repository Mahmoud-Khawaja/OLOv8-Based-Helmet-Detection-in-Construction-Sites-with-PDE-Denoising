import numpy as np
import cv2
from google.colab import files
from google.colab.patches import cv2_imshow
import io
from PIL import Image
import matplotlib.pyplot as plt

# 1. Bilateral Filtering Function
def bilateral_filter(img, d=15, sigma_color=100, sigma_space=100):
    # Convert RGBA to RGB if the image has 4 channels
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to 3-channel RGB
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

# 2. ROF TVD Denoising using Euler-Lagrange Equation (Fixed)
def rof_tvd_denoise(img, f_bilateral, lambda_reg=0.05, max_iter=40, tol=1e-5):
    """Refined ROF TVD denoising with better edge preservation."""
    u = img.copy().astype(np.float32)
    u_prev = u.copy()

    for _ in range(max_iter):
        grad_x = np.gradient(u, axis=1)
        grad_y = np.gradient(u, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)  
        div_grad = (np.gradient(grad_x / (grad_magnitude + 1e-8), axis=1) +
                    np.gradient(grad_y / (grad_magnitude + 1e-8), axis=0))
        u = u - 0.1 * (div_grad - lambda_reg * (u - f_bilateral))

        # Stop if change is small
        if np.linalg.norm(u - u_prev) < tol:
            break

        u_prev = u.copy()

    u = np.nan_to_num(u, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(u, 0, 255).astype(np.uint8)

def non_local_means_denoise(img, h=10):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

# 4. Full Hybrid Denoising Process
def hybrid_denoising(img, bilateral_diameter=25, sigma_color=100, sigma_space=100, lambda_reg=0.03, h=10):
    # Step 1: Apply bilateral filtering
    f_bilateral = bilateral_filter(img, d=bilateral_diameter, sigma_color=sigma_color, sigma_space=sigma_space)

    # Step 2: Apply ROF TVD Denoising
    denoised_img = rof_tvd_denoise(img, f_bilateral, lambda_reg=lambda_reg)

    # Step 3: Apply non-local means for additional noise reduction
    denoised_img = non_local_means_denoise(denoised_img, h=h)

    return denoised_img

uploaded = files.upload()

image_name = next(iter(uploaded))

img = cv2.imdecode(np.frombuffer(uploaded[image_name], np.uint8), cv2.IMREAD_UNCHANGED)

img = img.astype(np.uint8)

if len(img.shape) == 3 and img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

if img is None:
    print("Image not found or could not be loaded.")
else:
    print("Image successfully uploaded and loaded.")

Image.open(io.BytesIO(uploaded[image_name])).show()

denoised_image = hybrid_denoising(img, bilateral_diameter=25, sigma_color=100, sigma_space=100, lambda_reg=0.05, h=10)

cv2.imwrite('noisy_image.jpg', img)
cv2.imwrite('denoised_image.jpg', denoised_image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
plt.title("Noisy Image")
plt.axis('off') 

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB) if len(denoised_image.shape) == 3 else denoised_image, cmap='gray')
plt.title("Denoised Image")
plt.axis('off')

plt.show()
