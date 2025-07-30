import numpy as np
from imaris_ims_file_reader.ims import ims
import napari
import cv2
from skimage import morphology
import time

run_visualization = False

start_time = time.time()

# Load .ims file
file_path = "TESTI.ims"
data = ims(file_path)

# Extract shape: (T, C, Z, Y, X)
assert data.shape[0] == 1  # single timepoint
t = 0

# Extract 4 channels as 3D volumes (Z, Y, X)
channels = [np.array(data[t, c, :, :, :]) for c in range(data.shape[1])]

if run_visualization:
    # Launch Napari viewer
    viewer = napari.Viewer()

    # Add each channel as a separate grayscale 3D volume
    for idx, channel_data in enumerate(channels):
        viewer.add_image(
            channel_data,
            name=f"Channel {idx}",
            colormap="gray",
            blending="additive",  # you can try 'translucent' too
            rendering="mip",      # maximum intensity projection (3D)
        )

    napari.run()

gray = np.array(data[0, 0, :, :, :]).squeeze()
gray = gray.astype(np.float32) / 2**16

gray = 40* gray ** 1.5 

# Perform basic 3D morphological operations on the grayscale volume
struct_elem = morphology.ball(2)  # spherical structuring element with radius 1
gray_dilated = morphology.dilation(gray, struct_elem)
gray_eroded = morphology.erosion(gray, struct_elem)
gray_opened = morphology.opening(gray, struct_elem)
gray_closed = morphology.closing(gray, struct_elem)

# Visualize Z-projection of the original and processed volumes using OpenCV
gray_sum = gray.sum(axis=0)
gray_dilated_sum = gray_dilated.sum(axis=0)
gray_eroded_sum = gray_eroded.sum(axis=0)
gray_opened_sum = gray_opened.sum(axis=0)
gray_closed_sum = gray_closed.sum(axis=0)

elapsed_time = time.time()-start_time

print(f"Computation ready in {elapsed_time:.2f} seconds")

cv2.imshow("gray_projection", gray_sum)
cv2.imshow("dilated_projection", gray_dilated_sum)
cv2.imshow("eroded_projection", gray_eroded_sum)
cv2.imshow("opened_projection", gray_opened_sum)
cv2.imshow("closed_projection", gray_closed_sum)
k = cv2.waitKey(0)
