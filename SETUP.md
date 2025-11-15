## Setup Instructions

### 1. Activate the Virtual Environment

```bash
cd /Users/maggie-z/Desktop/Github/ECE658_LORA
source venv_sparselora/bin/activate
```

### 2. Verify Installation

```bash
python -c "from spft.api import SPFTConfig, get_spft_model; print('✓ spft.api imported successfully!')"
```

### 3. Use in Jupyter Notebook

To use this environment in your Jupyter notebook:

```bash
# Install ipykernel in the virtual environment
source venv_sparselora/bin/activate
pip install ipykernel

# Register the kernel with Jupyter
python -m ipykernel install --user --name=sparselora --display-name "Python (sparselora)"
```

Then in your Jupyter notebook, select the kernel: **Kernel → Change Kernel → Python (sparselora)**

## macOS-Specific Notes

- **liger-kernel**: This package requires Triton (CUDA-only), so it's been patched to be optional. The code will work but may have reduced performance optimizations.
- **bitsandbytes**: Installed but without GPU support (expected on macOS). This is fine for CPU-only usage.
