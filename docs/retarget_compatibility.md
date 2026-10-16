# Retargeting compatibility

The public preparation path uses official GMR at revision
`12a1d14b049e358dc182de9c679b30b96a215191` and this repository's
`scripts/data_process/retarget_amass.py`. No GentleHumanoid GMR fork is
installed or imported by that entry point.

## Compatibility profile

The profile `ceer-smplx-10hz-pchip50-v1` implements:

- AMASS SMPL-X human pose resampling at 10 Hz, using SLERP for rotations and
  PCHIP for translation, before evaluating the SMPL-X body model.
- Official GMR IK, with a fresh retargeter and subject-height scaling per clip,
  and explicit `use_velocity_limit=True`.
- Robot motion export at 50 Hz: PCHIP for positions and unwrapped joint angles,
  SLERP for root orientation, with endpoint-inclusive timing and rounded counts.
- FK for all G1 bodies and the head/hand virtual frames used by CEER.
- Clip-wide ground-height alignment and initial root XY centering.

The numerical preprocessing/export behavior is adapted from the
GentleHumanoid GMR fork, `Axellwppr/GMR`, revision
`c8970d755519d9bb6e79786a3fb43649b71198fa`. Attribution is retained here and
in [the MIT notice](licenses/gmr.txt). Users install only official GMR.

## End-to-end validation

Input: the local raw SMPL-X motion `CMU/140/140_02_stageii.npz`,
937 frames at 120 Hz. The compatibility entry produces 79 IK frames at
10 Hz and 391 output frames at 50 Hz.

| Comparison | Maximum joint difference (rad) | Maximum body position difference (m) |
|---|---:|---:|
| Fresh execution of the pinned reference fork | 3.21e-7 | 1.77e-6 |
| Existing local historical G1 NPZ | 9.22e-7 | 1.88e-6 |

Root position differences were below 3.3e-7 m, and root orientation differences
were below 3.7e-7 rad. Bodies are compared by name; quaternion differences use
rotation angles rather than component signs.

The output also passes the existing data packer: 391 source frames become
466 frames after its standard padding. This motion is not in the final
training allowlist, so this end-to-end smoke check runs without that filter.
The separate [adapter validation](data_conversion_validation.md) covers
7/7 real allowlist segments. Neither check launches policy training.

This validates one real raw AMASS motion end to end, not the entire corpus
or every historical fork version. The exact version originally used to
generate all historical files is not established by this test.

Validated environment: Python 3.10, numpy 2.2.6, scipy 1.15.3, mujoco 3.4.0,
mink 1.0.0, daqp 0.7.2, qpsolvers 4.8.2, smplx 0.1.28, torch 2.9.1.
Use a separate environment from Isaac Sim's numpy 1.x training environment.

Reproduce against an existing paired reference:

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 python scripts/data_process/validate_retarget_amass.py \
  --input /path/to/raw/CMU/140/140_02_stageii.npz \
  --reference /path/to/existing_g1/CMU/140/140_02_stageii.npz \
  --body-models /path/to/body_models \
  --output-dir /path/to/new_validation_directory \
  --packer-python /path/to/training_env/bin/python
```

The report records input/reference hashes and maximum errors. Reference-fork
execution is a validation-only activity and is not required for data preparation.
