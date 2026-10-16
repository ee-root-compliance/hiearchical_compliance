[← 中文首页](../README_zh.md)

## Dummy teleoperation 和交互式环境

`teleop_dummy_pub.py` 通过 UDP 发布 root、head 和双手 pose。请在两个终端中分别启动 simulator 和 publisher。

### 使用 `eval_manipulation.py` 实时 teleoperation

先启动 policy（`-p` 表示 interactive mode）：

```bash
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --full_collision \
  --num_envs 1 \
  --obs_source udp \
  --objects cfg/objects/table_scene.yaml \
  -p
```

然后在第二个终端启动 dummy publisher：

```bash
python teleop_dummy_pub.py \
  --mode 0 \
  --dst_ip 127.0.0.1 \
  --dst_port 15000 \
  --hz 30
```

Mode 0 的键盘控制：

```text
双手：I/K = x +/−，J/L = y 分开/靠拢，U/O = z +/−
Root：W/S = x +/−，A/D = y +/−，F/H = z +/−，Q/E = yaw +/−
退出：ESC 或 Ctrl+C
```

### 可用的物体环境

交互式运行时通过 `--objects` 传入以下配置之一：

| 场景 | 内容 |
|---|---|
| `cfg/objects/controller_test.yaml` | 最小动态 box 碰撞检查 |
| `cfg/objects/boxes_scene.yaml` | 平台和动态 cylinder |
| `cfg/objects/table_scene.yaml` | 桌子、box 和 sphere |
| `cfg/objects/room_scene.yaml` | 桌子、cylinder 和床 |
| `cfg/objects/wall_scene.yaml` | 前方和侧方墙体障碍物 |

物体只在 interactive mode（`-p`）中加载。

### Dummy publisher 模式

| Mode | 命令 | 作用 |
|---|---|---|
| `0` | `python teleop_dummy_pub.py --mode 0` | 键盘控制固定 pose 和 root motion |
| `1` | `python teleop_dummy_pub.py --mode 1` | VR pose 输入，需要本地 `vr`（OpenVR）模块 |
| `2` | `python teleop_dummy_pub.py --mode 2` | 每五秒随机生成手部 pose，用于 smoke test |
| `3` | `python teleop_dummy_pub.py --mode 3` | 接收 UDP 端口 15001 的 root-state CSV，并转发到 15000 |
| `4` | `python teleop_dummy_pub.py --mode 4 --motion-file /path/to/motion.npz` | 回放动作，可选 foot position |
| `5` | `python teleop_dummy_pub.py --mode 5 --motion-file /path/to/motion.npz` | 回放动作并附加 joint target |

Mode 3 接收的 CSV 格式为 `timestamp,x,y,z,qw,qx,qy,qz`。

### 使用本地 checkpoint

```bash
python scripts/eval_manipulation.py \
  --checkpoint /path/to/checkpoint.pt \
  --config-file /path/to/cfg.yaml \
  --full_collision \
  --obs_source udp \
  --objects cfg/objects/controller_test.yaml \
  -p
```

不要把 teleoperation 与 `--ee-compliance-eval` 或 `--root-compliance-eval` 同时使用。

