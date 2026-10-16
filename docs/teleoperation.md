[← README](../README.md)

## Dummy teleoperation and interactive environments

`teleop_dummy_pub.py` publishes root, head, and hand poses over UDP. Run the
simulator and the publisher in separate terminals.

### Live teleoperation with `eval_manipulation.py`

Start the policy first (`-p` = interactive mode):

```bash
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --full_collision \
  --num_envs 1 \
  --obs_source udp \
  --objects cfg/objects/table_scene.yaml \
  -p
```

Then start the dummy publisher in a second terminal:

```bash
python teleop_dummy_pub.py \
  --mode 0 \
  --dst_ip 127.0.0.1 \
  --dst_port 15000 \
  --hz 30
```

Packet format (`G6D1`): 4-byte magic, `uint32` sequence number, then four
world-frame poses (root, head, left hand, right hand) as `x,y,z, qx,qy,qz,qw`
little-endian `float32`.

The keyboard controls in mode 0 are:

```text
Hands: I/K = x +/-, J/L = y apart/together, U/O = z +/-
Root:  W/S = x +/-, A/D = y +/-, F/H = z +/-, Q/E = yaw +/-
Exit:  ESC or Ctrl+C
```

### Available object environments

Pass one of the following files through `--objects` in interactive play mode:

| Scene | Contents |
|---|---|
| `cfg/objects/controller_test.yaml` | Minimal dynamic-box collision check |
| `cfg/objects/boxes_scene.yaml` | Platform and dynamic cylinder |
| `cfg/objects/table_scene.yaml` | Table, box, and sphere |
| `cfg/objects/room_scene.yaml` | Table, cylinder, and bed |
| `cfg/objects/wall_scene.yaml` | Front and side wall obstacles |

Objects are loaded only in interactive mode (`-p`).

### Dummy publisher modes

| Mode | Command | Purpose |
|---|---|---|
| `0` (default) | `python teleop_dummy_pub.py --mode 0` | Keyboard-controlled fixed pose and root motion |
| `1` | `python teleop_dummy_pub.py --mode 1` | VR pose input; requires a local `vr` (OpenVR) module that is **not** included in this repository |
| `2` | `python teleop_dummy_pub.py --mode 2` | Random hand pose every five seconds for smoke testing |
| `3` | `python teleop_dummy_pub.py --mode 3` | Receives root-state CSV on UDP port `15001` and republishes a pose packet to port `15000` |
| `4` | `python teleop_dummy_pub.py --mode 4 --motion-file /path/to/motion.npz` | Replays a motion with optional foot positions |
| `5` | `python teleop_dummy_pub.py --mode 5 --motion-file /path/to/motion.npz` | Replays a motion and appends joint targets |

Mode 3 expects CSV lines `timestamp,x,y,z,qw,qx,qy,qz` on UDP port `15001`.

### Using a local checkpoint

With a local checkpoint:

```bash
python scripts/eval_manipulation.py \
  --checkpoint /path/to/checkpoint.pt \
  --config-file /path/to/cfg.yaml \
  --full_collision \
  --obs_source udp \
  --objects cfg/objects/controller_test.yaml \
  -p
```

Do not combine teleoperation with `--ee-compliance-eval` or `--root-compliance-eval`.

