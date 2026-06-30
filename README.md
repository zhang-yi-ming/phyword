
## 📦 Installation

```bash
cd cosmos_mot
conda create -n cosmos_mot python=3.10
conda activate cosmos_mot

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Getting Started

```bash
cd scripts
bash train.sh
```

## Main files
- Main training file: `scripts/train.py`
- If you want to see the logistics of MoT and the VLA wrapper class, refer to `models/cosmos_janus.py`
- For test, refer to `scripts/test_trainingset.sh` on training set, `experiments/test_libero.sh` in LIBERO

