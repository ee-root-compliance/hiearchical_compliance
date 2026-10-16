[← README](../README.md) · [简体中文](data_zh.md)

# Data preparation

If you already have complete G1 NPZ motions, skip to step 3 and use their directory as input.

## 1. Install the retargeter (once)

Download SMPL-X motions from [AMASS](https://amass.is.tue.mpg.de/) and obtain the SMPL-X body models following the [official GMR instructions](https://github.com/YanjieZe/GMR). The model directory should contain `smplx/SMPLX_NEUTRAL.npz` and the other required gender models.

Install official GMR in a separate environment:

```bash
conda create -n ceer-retarget python=3.10 -y
conda activate ceer-retarget
git clone https://github.com/YanjieZe/GMR.git /path/to/GMR
export GMR_ROOT=/path/to/GMR
git -C "$GMR_ROOT" checkout 12a1d14b049e358dc182de9c679b30b96a215191
pip install -e "$GMR_ROOT"
pip install numpy==2.2.6 scipy==1.15.3 mujoco==3.4.0 mink==1.0.0 \
  daqp==0.7.2 qpsolvers==4.8.2 smplx==0.1.28 torch==2.9.1
```

## 2. Generate G1 motions

From this repository's root, in the `ceer-retarget` environment:

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 python scripts/data_process/retarget_amass.py \
  --input /path/to/AMASS_SMPLX \
  --output dataset/g1_npz \
  --body-models /path/to/body_models \
  --allowlist scripts/data_process/allowlist_limmt_no_foot_compliance.json \
  --report dataset/retarget_report.json
```

This entry handles retargeting, 50 Hz interpolation, and end-effector frames. No additional GMR fork or PKL conversion step is needed. Preserve AMASS relative paths and filenames; use new output directories.

## 3. Filter and pack

Switch back to the [training environment](installation.md), named `humanoid-compliance` in the installation guide, and run from this repository's root:

```bash
python scripts/data_process/generate_dataset.py \
  --dataset-root dataset/g1_npz \
  --allowlist scripts/data_process/allowlist_limmt_no_foot_compliance.json \
  --mem-path dataset/limmt_no_foot_compliance_full
```

Default training configurations load `dataset/limmt_no_foot_compliance_full/` directly. Continue with [low-level training](ceer1.md). Check the terminal's `matched ... allowlist segments` output to confirm the required segments were found.

<details>
<summary>Compatibility and validation</summary>

The entry pins official GMR and supplies the required compatibility processing. See [provenance and end-to-end validation](retarget_compatibility.md) for comparison with historical data from raw AMASS.

`--mem-path` is the data output directory. Memory mapping is a storage format, not a training step.

For existing GMR PKL files, `convert_gmr_to_npz.py` performs format conversion without rerunning or altering upstream retargeting. See the [adapter validation](data_conversion_validation.md).

</details>

