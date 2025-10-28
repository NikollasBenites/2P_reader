import tifffile as tiff
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure
from pathlib import Path

# Force a backend that actually opens a window on macOS outside Jupyter.
# If this annoys PyCharm later, you can comment it out.
matplotlib.use("TkAgg")


def load_tiff_stack(path):
    """
    Load a TIFF movie (T, Y, X) or stack (Z, Y, X) into RAM.
    Returns (data, meta_dict, description_text)
    """
    with tiff.TiffFile(path) as tif:
        data = tif.asarray()
        # extract tags BEFORE closing file to avoid the colormap warning
        first_page = tif.pages[0]
        meta = {tag.name: tag.value for tag in first_page.tags.values()}
        desc = getattr(first_page, "description", None)

    return data, meta, desc


def percentile_contrast(img2d, low=1, high=99):
    """
    Stretch contrast for display only.
    """
    p_low, p_high = np.percentile(img2d, (low, high))
    if p_high == p_low:
        img_disp = np.zeros_like(img2d, dtype=np.float32)
    else:
        img_disp = (img2d.astype(np.float32) - p_low) / (p_high - p_low)
    return np.clip(img_disp, 0, 1)


def show_frame(frame2d, title="frame"):
    """
    Visualize one 2D frame with contrast stretching.
    """
    img_disp01 = percentile_contrast(frame2d)
    plt.imshow(img_disp01, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_as_tiff(img2d_or_3d, out_path, dtype=np.uint16):
    """
    Save image or stack as TIFF, converting dtype if needed.
    """
    arr = img2d_or_3d
    if arr.dtype != dtype:
        # scale if float 0..1 → uint16 0..65535
        if arr.dtype.kind == "f":
            arr_to_save = np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)
        else:
            arr_to_save = arr.astype(dtype)
    else:
        arr_to_save = arr

    tiff.imwrite(str(out_path), arr_to_save)


def summarize_movie(data, save_dir):
    """
    data: np.ndarray shaped (T, Y, X)
    save_dir: Path where we'll drop preview images
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    n_frames = data.shape[0]
    mid_t = n_frames // 2

    # 1. Show and save middle frame
    mid_frame = data[mid_t, :, :]
    show_frame(mid_frame, title=f"Frame {mid_t}/{n_frames}")

    # also save a PNG so you can open it from Finder
    plt.imsave(
        save_dir / f"frame_{mid_t:04d}.png",
        percentile_contrast(mid_frame),
        cmap="gray",
    )

    # 2. Projections across time
    mean_proj = np.mean(data, axis=0)
    max_proj = np.max(data, axis=0)

    # Display them
    show_frame(mean_proj, title="Mean across time")
    show_frame(max_proj, title="Max across time")

    # Save them
    plt.imsave(
        save_dir / "mean_projection.png",
        percentile_contrast(mean_proj),
        cmap="gray",
    )
    plt.imsave(
        save_dir / "max_projection.png",
        percentile_contrast(max_proj),
        cmap="gray",
    )

    # Also save as TIFFs (16-bit)
    save_as_tiff(mean_proj, save_dir / "mean_projection.tif")
    save_as_tiff(max_proj, save_dir / "max_projection.tif")

    print(f"[OK] Saved previews to {save_dir}")

import matplotlib.animation as animation

def play_movie(data, start=0, stop=None, fps=30):
    stop = stop or data.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(percentile_contrast(data[start]), cmap="gray")
    ax.axis("off")

    def update(i):
        im.set_array(percentile_contrast(data[i]))
        ax.set_title(f"Frame {i+1}/{data.shape[0]}")
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=range(start, stop), interval=1000/fps, blit=True
    )
    plt.show()

#


if __name__ == "__main__":
    # >>>>>>>>>  EDIT THIS  <<<<<<<<<
    tiff_path = Path("/Volumes/VCN_Connectome_DC015901_ReadWrite/ExpansionMicroscopy/2Photon/10212025/Conc_VCNSag_SI2ST2_927_1024_30FPS_z500nm_8bit_removedOutliers.tif")

    data, meta, desc = load_tiff_stack(tiff_path)

    print("Data shape:", data.shape)
    print("Data dtype:", data.dtype)
    print("Min/Max intensity:", data.min(), data.max())

    print("\n--- IMAGEJ DESCRIPTION ---")
    print(desc)

    # If shape is (T,Y,X) we'll call it 'movie'
    # If shape is (Z,Y,X) we'll call it 'zstack'
    if data.ndim == 3:
        axis0 = data.shape[0]
        if "frames" in str(desc).lower() or "frames" in meta:
            print(f"Interpreting axis 0 ({axis0}) as TIME (T).")
        elif "slices" in str(desc).lower():
            print(f"Interpreting axis 0 ({axis0}) as DEPTH (Z).")
        else:
            print(f"Axis 0 is {axis0} planes. Could be time or z; we guess TIME.")

        summarize_movie(data, save_dir=Path("previews"))

    else:
        print("Unexpected dimensionality:", data.ndim)
        # We could add support for (T,Z,Y,X) or (Z,T,Y,X) in the future.
