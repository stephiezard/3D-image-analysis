import numpy as np
from imaris_ims_file_reader.ims import ims
import napari

# Load .ims file
file_path = "TESTI.ims"
data = ims(file_path)

# Extract shape: (T, C, Z, Y, X)
assert data.shape[0] == 1  # single timepoint
t = 0

# Extract 4 channels as 3D volumes (Z, Y, X)
channels = [np.array(data[t, c, :, :, :]) for c in range(data.shape[1])]

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


