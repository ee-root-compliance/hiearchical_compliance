# GMR → CEER conversion validation

Validated using existing local paired GMR PKL and CEER NPZ files. No training or full-corpus retargeting was run.

| Motion | Frames | Common bodies | Maximum body position difference (m) | Maximum body rotation difference (rad) |
|---|---:|---:|---:|---:|
| ACCAD/Female1General_c3d/A14_-_stand_to_skip_stageii | 256 | 41 | 1.86e-6 | 1.51e-6 |
| KIT/1747/displace_from_left_to_right_28_stageii | 226 | 41 | 1.96e-6 | 1.15e-6 |
| WEIZMANN/70/Slow_StraightShort(4)_stageii | 4706 | 41 | 1.86e-6 | 1.80e-6 |

All three preserved source frame counts at 50 Hz. Joint differences were below 6e-8 rad. The generated data passed the existing packer with **7/7 allowlist segments**, totaling 5,413 packed frames, and loaded through the training `MotionDataset.create_from_path_lazy` interface with all required head/hand bodies.

Additional checks passed: controlled 30→50 Hz resampling (31→51 frames over one second), endpoint preservation, unit quaternions, explicit reversed joint order, and zero-quaternion rejection. The resampling fixture is derived from a real motion but is not a separate native 30 Hz dataset validation.

Reproduce with local paired data:

```bash
MUJOCO_GL=disable python scripts/data_process/validate_gmr_conversion.py \
  --dataset-root /path/to/paired_amass \
  --robot-xml /path/to/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
  --output-dir /path/to/new_validation_directory \
  --count 3
```

The directory must not already exist. The validator writes converted samples, a subset allowlist, a training memmap, and `validation.json`. It samples across the available paired corpus; different local data may select different motions.

These checks establish adapter compatibility with the tested G1 XML and existing motions. They do not establish that a different GMR version reproduces the historical retargeting or all original allowlist boundaries.
