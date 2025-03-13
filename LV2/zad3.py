import numpy as np
import matplotlib.pyplot as plt

image = plt.imread('road.jpg')


bright_image = image + 30
bright_image[bright_image > 230] = 230

height, width, _ = image.shape
second_quarter_image = image[:, width // 2 - width // 4 : width // 2]

rotated_image = np.rot90(image, -1) 

mirrored_image = np.fliplr(image)  

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(image)
axes[0, 0].set_title('Originalna slika')
axes[0, 0].axis('off')

axes[0, 1].imshow(bright_image)
axes[0, 1].set_title('Posvijetljena slika')
axes[0, 1].axis('off')

axes[1, 0].imshow(second_quarter_image)
axes[1, 0].set_title('Druga četvrtina po širini')
axes[1, 0].axis('off')

axes[1, 1].imshow(mirrored_image)
axes[1, 1].set_title('Zrcaljena slika')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
