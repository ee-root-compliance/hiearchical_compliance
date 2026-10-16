[← 中文首页](../README_zh.md) · [English](data.md)

# 数据准备

已有完整 G1 NPZ 数据时，直接执行第 3 步，将输入目录换成现有数据路径。

## 1. 安装重定向工具（首次使用）

从 [AMASS](https://amass.is.tue.mpg.de/) 下载 SMPL-X 格式动作，并按 [GMR 官方说明](https://github.com/YanjieZe/GMR) 获取 SMPL-X 人体模型。模型目录应包含 `smplx/SMPLX_NEUTRAL.npz` 等对应性别的模型文件。

在独立环境中安装官方 GMR：

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

## 2. 生成 G1 动作

在本仓库根目录、`ceer-retarget` 环境中运行：

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 python scripts/data_process/retarget_amass.py \
  --input /path/to/AMASS_SMPLX \
  --output dataset/g1_npz \
  --body-models /path/to/body_models \
  --allowlist scripts/data_process/allowlist_limmt_no_foot_compliance.json \
  --report dataset/retarget_report.json
```

本仓库入口会自动处理重定向、50 Hz 插值和末端坐标补全。无需安装额外的 GMR fork，也无需再运行 PKL 转换脚本。保留 AMASS 子目录和文件名；请使用新的输出目录。

## 3. 筛选并生成训练数据

切回[训练环境](installation_zh.md) `humanoid-compliance`，在本仓库根目录运行：

```bash
python scripts/data_process/generate_dataset.py \
  --dataset-root dataset/g1_npz \
  --allowlist scripts/data_process/allowlist_limmt_no_foot_compliance.json \
  --mem-path dataset/limmt_no_foot_compliance_full
```

完成后，默认训练配置直接读取 `dataset/limmt_no_foot_compliance_full/`。接下来即可[训练底层策略](ceer1_zh.md)。检查终端的 `matched ... allowlist segments` 输出，确认所需片段已匹配。

<details>
<summary>兼容性与验证记录</summary>

入口固定官方 GMR 版本，并包含本项目所需的数据兼容处理；来源说明及原始 AMASS 到历史数据的对比见[兼容性验证](retarget_compatibility.md)。

这里的 `--mem-path` 是数据输出目录，memmap 是高效读取数据的存储格式，不是训练步骤。

已有 GMR PKL 时，可使用 `convert_gmr_to_npz.py` 做格式转换，但它不会重做或改变上游 retarget 流程。[转换验证记录](data_conversion_validation.md)

</details>

