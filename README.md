# GSFM — Graph-Structured Flow Matching for Stochastic Human Motion Prediction

Conditional flow matching on the product manifold `X_G = R^3 x (S^2)^B` of a kinematic
tree. A pose is a root translation plus one unit direction per bone; bone lengths are
taken from the observation and held fixed, so every generated pose has exactly the
observed limb lengths (limb stretching and jitter are zero by construction). The velocity
field is a spatio-temporal transformer that attends over joints within a frame (biased by
skeleton hop distance) and over frames within a joint track (biased by signed time
offset in seconds). It carries no per-joint learned embedding, so it is equivariant to
joint relabelling and evaluates unchanged on a skeleton it was never trained on.

The source distribution `q_0` is a wrapped Gaussian of spread `sigma_d` around the
zero-velocity extrapolation of the observation; the conditional path is the geodesic
from a source draw to the ground-truth future; the target velocity is closed form; the
generation time is logit-normal. Sampling integrates the learned field with a midpoint
solver whose every step is an exponential map, with parallel transport of the midpoint
velocity back to the current point.

## Layout

```
.
├── README.md
├── requirements.txt
├── download_amass.py
├── download_h36m.py
├── data/
└── scripts/
    ├── train.py
    ├── evaluate.py
    ├── configs/
    │   ├── amass_deep_hpg.yaml
    │   ├── amass_deep_s030_hpg.yaml
    │   ├── amass_deep_s050_hpg.yaml
    │   ├── amass_deep_s070_hpg.yaml
    │   ├── amass_tied_hpg.yaml
    │   ├── amass_tied_s050_hpg.yaml
    │   └── amass_tied_s070_hpg.yaml
    ├── data_processing/
    │   ├── amass_preprocess.py
    │   ├── dataset.py
    │   └── skeleton.py
    ├── models/
    │   ├── geometry.py
    │   ├── flow_matching.py
    │   ├── gst_transformer.py
    │   └── embeddings.py
    └── utils/
        ├── metrics.py
        ├── published_metrics.py
        ├── dist.py
        └── logging_utils.py
```

## Install

```bash
conda create -n gsfm python=3.10 -y && conda activate gsfm
pip install -r requirements.txt
```

## Data

AMASS needs the SMPL+H body models and the raw archives from the AMASS project page
(registration required). Preprocessing writes 22-joint positions at 60 fps, split into
train / valid / test by sub-dataset as in BeLFusion and SkeletonDiffusion.

```bash
python download_amass.py --login --extract
python -m scripts.data_processing.amass_preprocess --raw-dir data/amass_raw/extracted --body-models data/body_models --out data/data_3d_amass.npz
```

`download_amass.py --login` asks for the AMASS account credentials interactively; if
the site blocks scripted downloads, save the archive links from the download page to a
text file and run `python download_amass.py --from-urls links.txt --extract` instead.
Place the SMPL+H models under `data/body_models/smplh/{male,female,neutral}/model.npz`.

Human3.6M, 17 joints, native 50 fps, resampled to 60 fps at load time:

```bash
python download_h36m.py --data-dir data
```

## Train

Four GPUs, one node, 94,200 steps (100 epochs):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m scripts.train --config scripts/configs/amass_deep_hpg.yaml
```

Single GPU:

```bash
python -m scripts.train --config scripts/configs/amass_deep_hpg.yaml
```

Any field can be overridden without editing the file:

```bash
python -m scripts.train --config scripts/configs/amass_deep_hpg.yaml --set flow.sigma_dir=0.7 --run-name gsfm_s070
```

| config | model |
|---|---|
| `amass_deep_hpg.yaml` | GSFM-deep: 12 distinct blocks, 30.49 M parameters, `sigma_d` 0.9 |
| `amass_tied_hpg.yaml` | GSFM-tied: one block applied 12 times with an iteration embedding, 4.47 M parameters |
| `amass_deep_s030/s050/s070_hpg.yaml` | the source-spread study on GSFM-deep |
| `amass_tied_s050/s070_hpg.yaml` | the source-spread study on GSFM-tied |

## Evaluate

The protocol is 0.5 s observed and 2.0 s predicted at 60 fps, 50 sampled futures per
observation, midpoint solver with 25 steps (50 function evaluations), root-centred
poses. `--pool published` reproduces the BeLFusion / SkeletonDiffusion AMASS test
segmentation (12,742 windows: prediction windows start at frame 180 of every test
sequence and every 120 frames after it), on which the ZeroVelocity baseline matches the
published row column by column.

```bash
python -m scripts.evaluate --checkpoint runs/amass_deep_hpg/checkpoints/ckpt_last.pt --split test --pool published --num-samples 50 --num-steps 25 --solver midpoint --out runs/amass_deep_hpg/eval_test.json
```

Sharded over N GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m scripts.evaluate --checkpoint runs/amass_deep_hpg/checkpoints/ckpt_last.pt --split test --pool published --num-samples 50 --num-steps 25 --solver midpoint --out runs/amass_deep_hpg/eval_test.json
```

Zero-shot kinematics: the AMASS checkpoint on the 17-joint Human3.6M skeleton, no
retargeting and no fine-tuning:

```bash
python -m scripts.evaluate --checkpoint runs/amass_deep_hpg/checkpoints/ckpt_last.pt --split test --num-samples 50 --num-steps 25 --solver midpoint --data-path data/data_3d_h36m_17.npz --data-source-fps 50 --data-subjects S9 S11 --out runs/amass_deep_hpg/eval_h36m.json
```

The JSON holds two conventions. `conv_ade`, `conv_fde` and `mae_sd` are the published
ADE / FDE / MAE (flattened-pose norm in metres, inter-limb angles in degrees). The
`published` block holds MMADE, MMFDE, APDE, APD and CMD exactly as SkeletonDiffusion's
code defines them; those are the numbers that sit next to published tables. The
unified `ade`, `fde`, `apd` keys are per-joint centimetres and are not comparable with
published columns.

## Checkpoints

A training run writes `runs/<run_name>/checkpoints/ckpt_last.pt` with the model
weights, the EMA weights (used for evaluation), the optimiser state and the resolved
configuration. `evaluate.py` rebuilds the model from that configuration, so a checkpoint
needs nothing but the data file it was trained on.
