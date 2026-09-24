![NCR Logo](./misc/banner.png)

# Skeletons in Flow: Graph-Structured Flow Matching for Human Motion Prediction

by Yixuan Wang, Brandon Fallin, and Warren E. Dixon

## Abstract
> Human motion prediction requires diverse future trajectories that remain consistent with observed motion and the articulated physical structure of the body. Skeletal constraints restrict individual poses, while coordinated motion depends on spatial interactions (between connected joints) and temporal interactions (between  time instants). To facilitate human motion prediction in light of these constraints and interactions, we introduce Graph Structured Flow Matching (GSFM), which transports the complete future skeletal trajectory through a single conditional velocity field. The trajectory produces a spatiotemporal skeleton graph, and spatial and temporal attention couple its evolution according to skeletal relations and physical time offsets. Bone directions lie on unit spheres relative to a root joint, and tangent evolution preserves input bone lengths throughout generation. We train the field through conditional flow matching along geodesic paths connecting random trajectories centered on the last-observed pose to recorded future trajectories. Experiments on the Archive of Motion capture As Surface Shapes (AMASS) dataset evaluate prediction accuracy, diversity calibration, and motion statistics. We demonstrate the contributions of spatial and temporal message passing in the developed architecture through an ablation study.

## Layout

```
.
├── README.md
├── requirements.txt
├── download_amass.py
├── download_h36m.py
├── data/
├── misc/
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

AMASS needs the SMPL+H body models and the raw archives from the AMASS project page (registration required). Preprocessing for AMASS constructs 22-joint positions at 60 fps, split into train / valid / test by sub-dataset as in BeLFusion and SkeletonDiffusion.

```bash
python download_amass.py --login --extract
python -m scripts.data_processing.amass_preprocess --raw-dir data/amass_raw/extracted --body-models data/body_models --out data/data_3d_amass.npz
```

The script `download_amass.py --login` asks the user to input their AMASS account credentials. If the site blocks the download using the script, save the archive links from the download page to a `.txt` file and run `python download_amass.py --from-urls links.txt --extract` instead.

Place the SMPL+H models under `data/body_models/smplh/{male,female,neutral}/model.npz`.

Preprocessing for Human3.6M constructs 17-joint positions, at 50 fps, which is resampled to 60 fps:

```bash
python download_h36m.py --data-dir data
```

## Training

To train using 4 GPUs and 94,200 steps (100 epochs), run:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m scripts.train --config scripts/configs/amass_deep_hpg.yaml
```

To train using a single GPU, run:

```bash
python -m scripts.train --config scripts/configs/amass_deep_hpg.yaml
```

Any field in the config can be overridden without directly editing the .yaml file:

```bash
python -m scripts.train --config scripts/configs/amass_deep_hpg.yaml --set flow.sigma_dir=0.7 --run-name gsfm_s070
```

The model configurations are as follows:

| config | model |
|---|---|
| `amass_deep_hpg.yaml` | GSFM-deep: 12 distinct blocks, 30.49 M parameters, `sigma_d` 0.7 |
| `amass_tied_hpg.yaml` | GSFM-tied: one block applied 12 times, 4.47 M parameters, `sigma_d` 0.7 |
| `amass_deep_s030/s050/s070_hpg.yaml` | source scale study on GSFM-deep |
| `amass_tied_s050/s070_hpg.yaml` | source scale study on GSFM-tied |

## Evaluation

For evaluation, we use 0.5 s of observed data and 2.0 s of predicted data at 60 fps against 50 sampled future trajectories per input observation. To evaluate the velocity field, we use a midpoint solver with 25 steps (50 function evaluations). 

Using the flag `--pool published` applies the BeLFusion / SkeletonDiffusion AMASS test partition (12,742 windows with prediction windows starting at frame 180 of every test sequence and every 120 frames after it). This configuration matches that of the ZeroVelocity baseline. To evaluate results, run:

```bash
python -m scripts.evaluate --checkpoint runs/amass_deep_hpg/checkpoints/ckpt_last.pt --split test --pool published --num-samples 50 --num-steps 25 --solver midpoint --out runs/amass_deep_hpg/eval_test.json
```

Distributed over N GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m scripts.evaluate --checkpoint runs/amass_deep_hpg/checkpoints/ckpt_last.pt --split test --pool published --num-samples 50 --num-steps 25 --solver midpoint --out runs/amass_deep_hpg/eval_test.json
```

For zero-shot transfer, we apply the AMASS checkpoint on the 17-joint Human3.6M skeleton without retargeting or fine-tuning:

```bash
python -m scripts.evaluate --checkpoint runs/amass_deep_hpg/checkpoints/ckpt_last.pt --split test --num-samples 50 --num-steps 25 --solver midpoint --data-path data/data_3d_h36m_17.npz --data-source-fps 50 --data-subjects S9 S11 --out runs/amass_deep_hpg/eval_h36m.json
```

We use `conv_ade`, `conv_fde` and `mae_sd` to denote ADE / FDE / MAE (flattened-pose norm in meters, inter-limb angles in degrees). The `published` block contains MMADE, MMFDE, APDE, APD and CMD as they are defined in SkeletonDiffusion. The `ade`, `fde`, `apd` keys are measured in centimeters and are not comparable with existing results. 

## Checkpoints

A single training run writes `runs/<run_name>/checkpoints/ckpt_last.pt` which contains the model weights, the EMA weights (used for evaluation), the optimizer state and the resolved configuration. The file `evaluate.py` rebuilds the model from that configuration.
