

```markdown
# GRPO-SingleLawyer

Implementation and evaluation pipeline for training reasoning models using Group Relative Policy Optimization (GRPO) for legal domain tasks.

---

## System Requirements & Prerequisites

### Hardware
- **GPU:** NVIDIA GPU with high VRAM recommended (e.g., **NVIDIA L40S 48GB**, which was used to train the original baseline model).
- **Disk Space:** Sufficient storage for model weights and cache (persistent storage recommended).

### Software & Drivers
- **Python:** `3.12` (exact version required)
- **CUDA:** `13.0` (exact version required)
- [**uv**](https://github.com/astral-sh/uv) (fast Python package installer and resolver)
- [**Git LFS**](https://git-lfs.com/) (required to download pre-trained model weights)
- A [Weights & Biases (W&B)](https://wandb.ai/) account for experiment tracking
- An [OpenRouter](https://openrouter.ai/) API key (or any OpenAI-compatible API provider)

---

## Installation & Setup

### 1. Install Git LFS & Clone the Repository

Git Large File Storage (LFS) is required to pull the pre-trained model checkpoints:

```bash
# Install Git LFS (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Clone the repository
git clone https://github.com/your-username/GRPO-SingleLawyer.git
cd GRPO-SingleLawyer
```

### 2. Environment Setup with `uv` (Python 3.12)

Ensure Python 3.12 is active, then create a virtual environment and sync the pinned dependencies:

```bash
# Create a Python 3.12 virtual environment using uv
uv venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install exact dependencies
uv pip sync requirements.lock
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add your API credentials and project settings to `.env`:

```ini
OPENROUTER_API_KEY="your-openrouter-api-key-here"
WANDB_API_KEY="your-wandb-api-key-here"
WANDB_PROJECT="GRPOSingleLawyerTest"
```

> **Note:** The codebase uses the standard OpenAI client schema. Other OpenAI-compatible providers should work interchangeably by adjusting the base URL and API key, though OpenRouter is the primary tested provider.

---

## Disk Space & Hugging Face Cache Setup

<details>
<summary><b>Click to expand cache redirection tips (recommended for cloud instances / RunPod)</b></summary>

If your root partition (`/` or container overlay) runs out of space while downloading large model weights, redirect the Hugging Face cache directory to a persistent or larger volume (e.g., `/workspace`):

```bash
# Create custom cache directory
mkdir -p /workspace/.cache/huggingface

# Set environment variable for current session
export HF_HOME=/workspace/.cache/huggingface

# Persist to bash profile
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
```

</details>

---

## Training

Training is configured and executed via `ART_train.py`.

### Hyperparameters & Configuration

You can configure training settings directly in `ART_train.py` or pass them via environment variables:

```python
LEARNING_RATE = 5e-6
GROUP_SIZE    = 8
RUN_NAME      = os.environ.get("RUN_NAME", "GRPO-Lawyer-Experiment-1")
PROJECT       = "GRPO-SingleLawyer"
CONTEXT_LEN   = 12288
```

### Start Training

Run the training script:

```bash
# Run with default settings
python3 ART_train.py

# Or specify a custom run name inline
RUN_NAME="my-custom-grpo-run" python3 ART_train.py
```

---

## Evaluation & Demo Generation

The `evaluate.py` script allows you to evaluate trained checkpoints against a test suite or generate demo data.

### Configuration

Open `evaluate.py` and ensure the target model name/path matches the checkpoint you want to evaluate:

```python
# In evaluate.py:
LEARNING_RATE = 5e-6
GROUP_SIZE = 8
RUN_NAME = os.environ.get("RUN_NAME", "SAME THING HERE")   
PROJECT  = "GRPO-SingleLawyer"
CONTEXT_LEN = 12288 ```

### Run Evaluation

In `main()` inside `evaluate.py`, toggle between running the test set or re-generating demo records, then run:

```bash
python3 evaluate.py
```

---

## Project Structure

```text
GRPO-SingleLawyer/
├── ART_train.py        # Main training script (GRPO implementation)
├── evaluate.py         # Evaluation benchmark and demo data generator
├── requirements.lock   # Pinned dependency lockfile
├── .env                # API keys and environment configuration (create this)
└── README.md
```


