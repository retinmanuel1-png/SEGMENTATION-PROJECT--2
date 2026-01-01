import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


image_path = input("Enter the image file path: ")


image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Please check the path.")
    exit()


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

k = int(input("Enter number of segments (clusters): "))

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(pixel_values)


centers = np.uint8(kmeans.cluster_centers_)
labels = kmeans.labels_

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented_image)
plt.axis("off")

plt.show()
