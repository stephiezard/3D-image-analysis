import numpy as np
from imaris_ims_file_reader.ims import ims
import napari
import cv2

run_visualization = False

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

gray = np.array(data[0, 0, :,:,:]).squeeze()
gray = gray.astype(np.float32)/2**16
gray_sum = gray.sum(axis=0)
cv2.imshow("gray_projection", gray_sum)
k = cv2.waitKey(0)

