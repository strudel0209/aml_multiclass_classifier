## Installing PyTorch (smaller CPU-only distribution on Linux)

On Linux, `torch` can be very large because `pip` may pull CUDA-enabled wheels by default. To install the smaller **CPU-only** PyTorch distribution, install dependencies using the PyTorch CPU wheel index:

```bash
pip install --upgrade pip && python -m pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```