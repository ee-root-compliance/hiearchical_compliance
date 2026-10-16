[← README](../README.md)

## Dataset

The motion datasets are not redistributed in this repository. Obtain them from
their official distribution channels, such as [AMASS](https://amass.is.tue.mpg.de/),
and follow their licenses.

### Input format

Each input `.npz` must be a motion clip **already retargeted to the Unitree G1**
and sampled at **50 Hz**. Raw AMASS SMPL files are not accepted, and the
converter does not resample.

A G1-retargeted file contains:

| Key | Shape | Notes |
|---|---|---|
| `joint_names` | (J,) | Must include all 29 G1 joints, in any order |
| `body_names` | (B,) | G1 links plus `head_mimic`, `left_hand_mimic`, `right_hand_mimic` |
| `root_pos` | (T, 3) | World position, metres |
| `root_rot` | (T, 4) | World orientation, quaternion **xyzw** |
| `dof_pos` | (T, J) | Radians, in `joint_names` order |
| `local_body_pos` | (T, B, 3) | Body positions in the root frame |
| `local_body_rot` | (T, B, 4) | Body orientations relative to the root, **xyzw** |
| `mocap_framerate`, `frequency`, or `fps` | scalar | Must be 50 |

The LIMMT keypoint format (`qpos`, `qvel`, `kpt2gv_pose`, `gv2wrd_pose`) is also
accepted, but lacks the head and wrist bodies used by the 3-keypoint tasks.

### Directory layout

Allowlist paths are relative to `--dataset-root`, e.g.
`<dataset-root>/ACCAD/Female1General_c3d/A14_-_stand_to_skip_stageii.npz`.
The root should contain only motion `.npz` files.

### Conversion

Create the dataset used by all default task configs:

```bash
python scripts/data_process/generate_dataset.py \
  --dataset-root /path/to/G1_retargeted_motions \
  --allowlist scripts/data_process/allowlist_limmt_no_foot_compliance.json \
  --mem-path dataset/limmt_no_foot_compliance_full
```

Or use the wrapper:

```bash
DATASET_ROOT=/path/to/G1_retargeted_motions bash generate_limmt_amass_full_dataset.sh
```

Datasets are loaded from `<repository>/dataset/`. To store them elsewhere, set
`MEMPATH=/path/to/datasets`.

