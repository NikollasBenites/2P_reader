import tifffile as tifffile
import napari
import numpy as np
from pathlib import Path


def load_tiff(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        data = tif.asarray()  # shape should be (333, 1024, 1024)
        tags = tif.pages[0].tags
        xres = tags["XResolution"].value  # (num, den)
        yres = tags["YResolution"].value  # (num, den)

    # convert (num, den) -> float
    x_ppunit = xres[0] / xres[1]
    y_ppunit = yres[0] / yres[1]

    # pixel size in "unit" per pixel is 1 / (pixels per unit)
    px_size_x = 1.0 / x_ppunit
    px_size_y = 1.0 / y_ppunit

    # build scale:
    # we don't know z-step from metadata, so just put 1 for now
    scale = (1.0, px_size_y, px_size_x)

    print("data shape:", data.shape)
    print("scale going to napari:", scale)

    viewer = napari.Viewer()
    viewer.add_image(
        data,
        name="2p_stack",
        scale=scale,
        contrast_limits=(
            np.percentile(data, 1),
            np.percentile(data, 99),
        ),
        colormap="gray",
    )
    napari.run()


if __name__ == "__main__":
    # EDIT THIS
    tiff_path = Path("/Volumes/VCN_Connectome_DC015901_ReadWrite/ExpansionMicroscopy/2Photon/2P_PreExp_Imagedon10142025_P45_DAPI&V5/DCN_927nm_1024_z250nm_rectangle00001and00002Concatenated.tif")

    data = load_tiff(tiff_path)
    print("Loaded data:", data.shape, data.dtype)

    # If data is (T, Y, X) that's perfect for napari as a "stack".
    # Napari will give you a slider for axis 0 automatically.
    viewer = napari.Viewer()
    viewer.add_image(
        data,
        name="2P_movie",
        contrast_limits=[np.percentile(data, 1), np.percentile(data, 99)],
        colormap="gray",
    )

    napari.run()

