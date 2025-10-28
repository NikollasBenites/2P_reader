# Python environment dependencies for 2P_reader
# Install with: pip install -r requirements.txt
# Notes:
# - viewer.py uses napari which requires a Qt backend; the [all] extra pulls a working Qt stack.
# - reader.py forces matplotlib's TkAgg backend; ensure system Tk is available on your OS (macOS usually has it via python-tk / tk).

numpy
tifffile
matplotlib
scikit-image
napari[all]
