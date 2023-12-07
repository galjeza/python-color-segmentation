import cv2
import numpy as np
from matplotlib import pyplot as plt


def segmentiraj_barvo(slika_rgb, barva, val_min=0.1, sat_min=0.1):
    slika_hsv = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)    
    color_ranges = {
        'rdeca': [((0, int(sat_min * 255), int(val_min * 255)), (15, 255, 255)),
                  ((165, int(sat_min * 255), int(val_min * 255)), (180, 255, 255))],  # 330/2 = 165
        'rumena': [((15, int(sat_min * 255), int(val_min * 255)), (45, 255, 255))],  # 30/2 = 15, 90/2 = 45
        'zelena': [((45, int(sat_min * 255), int(val_min * 255)), (75, 255, 255))],  # 90/2 = 45, 150/2 = 75
        'turkizna': [((75, int(sat_min * 255), int(val_min * 255)), (105, 255, 255))],
        'modra': [((105, int(sat_min * 255), int(val_min * 255)), (135, 255, 255))],  # 150/2 = 75, 210/2 = 105
        'vijolicna': [((135, int(sat_min * 255), int(val_min * 255)), (165, 255, 255))],  # 210/2 = 105, 270/2 = 135
    }

    # Initialize a mask that selects nothing
    mask = np.zeros(slika_hsv.shape[:2], dtype=np.uint8)

    # Process the ranges for the specified color
    for (lower_bound, upper_bound) in color_ranges[barva]:
        # Create a temporary mask for each range and combine them
        temp_mask = cv2.inRange(slika_hsv, lower_bound, upper_bound)
        mask = cv2.bitwise_or(mask, temp_mask)

    # Ensure the mask is binary
    mask[mask > 0] = 1
    return mask

slika_bgr = cv2.imread('./data/img1.jpg')
slika_rgb = cv2.cvtColor(slika_bgr, cv2.COLOR_BGR2RGB)
mask_rumena = segmentiraj_barvo(slika_rgb, 'rumena')  

# kjer je maska true , naj bo slika_rgb, kjer je false je grayscale
slika_gray = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2GRAY)
slika_gray_3ch = cv2.cvtColor(slika_gray, cv2.COLOR_GRAY2RGB)
slika_combined = np.where(mask_rumena[:, :, np.newaxis]==1, slika_rgb, slika_gray_3ch)


plt.figure(figsize=(15, 5))

# Prvotna slika
plt.subplot(1, 3, 1)
plt.imshow(slika_rgb)
plt.title('Original Image')
plt.axis('off')

# Maska
plt.subplot(1, 3, 2)
plt.imshow(mask_rumena, cmap='gray')
plt.title('Mask for Selected Color')
plt.axis('off')

# kombinirana slika
plt.subplot(1, 3, 3)
plt.imshow(slika_combined)
plt.title('Combined Image')
plt.axis('off')
plt.show()


rdeci_kanal = slika_rgb[:, :, 0][mask_rumena]
zeleni_kanal = slika_rgb[:, :, 1][mask_rumena]
modri_kanal = slika_rgb[:, :, 2][mask_rumena]


fig, axs = plt.subplots(3, 1, figsize=(10, 10))
plt.xlabel("vrednost pixslov")
plt.ylabel("stevilo pixslov")
plt.grid()
axs[0].hist(rdeci_kanal.ravel(), bins=256, color='red')
axs[0].set_title('Histogram Rdečega Kanala')
axs[0].grid()
plt.grid()
axs[1].hist(zeleni_kanal.ravel(), bins=256, color='green')
axs[1].set_title('Histogram Zelenega Kanala')
axs[1].grid()

axs[2].hist(modri_kanal.ravel(), bins=256, color='blue')
axs[2].set_title('Histogram Modrega Kanala')
axs[2].grid()

plt.tight_layout()
plt.show()





