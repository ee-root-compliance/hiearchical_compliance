[← 中文首页](../README_zh.md)

## 数据集

动作数据集不包含在本仓库中。请从官方渠道（例如 [AMASS](https://amass.is.tue.mpg.de/)）获取数据，并遵守相应许可证。

### 输入格式

每个输入 `.npz` 必须已经 retarget 到 Unitree G1，并以 50 Hz 采样。仓库不接受原始 AMASS SMPL 文件，也不会自动重采样。

一个 G1-retargeted 文件包含：

| Key | Shape | 说明 |
|---|---|---|
| `joint_names` | `(J,)` | 必须包含全部 29 个 G1 joints，顺序不限 |
| `body_names` | `(B,)` | G1 links，以及 `head_mimic`、`left_hand_mimic`、`right_hand_mimic` |
| `root_pos` | `(T, 3)` | 世界坐标位置，单位 m |
| `root_rot` | `(T, 4)` | 世界坐标姿态，四元数顺序为 **xyzw** |
| `dof_pos` | `(T, J)` | 单位 rad，顺序与 `joint_names` 一致 |
| `local_body_pos` | `(T, B, 3)` | root frame 下的 body 位置 |
| `local_body_rot` | `(T, B, 4)` | 相对 root 的 body 姿态，四元数顺序为 **xyzw** |
| `mocap_framerate`、`frequency` 或 `fps` | scalar | 必须为 50 |

LIMMT keypoint 格式（`qpos`、`qvel`、`kpt2gv_pose`、`gv2wrd_pose`）也被支持，但其中不包含 3-keypoint task 使用的 head 和 wrist body。

### 数据转换

```bash
python scripts/data_process/generate_dataset.py \
  --dataset-root /path/to/G1_retargeted_motions \
  --allowlist scripts/data_process/allowlist_limmt_no_foot_compliance.json \
  --mem-path dataset/limmt_no_foot_compliance_full
```

也可以使用 wrapper：

```bash
DATASET_ROOT=/path/to/G1_retargeted_motions bash generate_limmt_amass_full_dataset.sh
```

默认 task 从 `<repository>/dataset/` 加载数据。若要保存到其他位置，设置 `MEMPATH=/path/to/datasets`。

