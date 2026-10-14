import torch
from typing import List

from active_adaptation.utils.math import wrap_to_pi  # noqa: F401
from active_adaptation.utils.motion import MotionDataset, MotionData

class ProgressiveMultiMotionDataset:
    def __init__(self,
                 mem_paths: List[str],
                 path_weights: List[float],
                 env_size: int,
                 max_step_size: int,
                 refresh_threshold: int = 500 * 20,
                 dataset_extra_keys: List[dict] = [],
                 device: torch.device = torch.device("cpu"),
                 ds_device: torch.device = torch.device("cpu"),
                 fix_ds: int = None,
                 fix_motion_id: int = None,
                 sample_once: bool = True):
        self.device = device
        self.ds_device = ds_device
        self.env_size = env_size
        self.max_step_size = max_step_size
        self.refresh_threshold = int(refresh_threshold)
        self.dataset_extra_keys = dataset_extra_keys

        self.fix_ds = fix_ds
        self.fix_motion_id = fix_motion_id
        self.sample_once = sample_once

        self.datasets = [
            MotionDataset.create_from_path_lazy(p, dataset_extra_keys, device=ds_device)
            for p in mem_paths
        ]
        assert len(self.datasets) == len(path_weights)

        body0, joint0 = self.datasets[0].body_names, self.datasets[0].joint_names
        for ds in self.datasets[1:]:
            assert ds.body_names == body0 and ds.joint_names == joint0
        self.body_names = body0
        self.joint_names = joint0

        w = torch.tensor(path_weights, dtype=torch.double)
        self.probs = (w / w.sum()).float().to(device)
        self.counts = [ds.num_motions for ds in self.datasets]

        self._buf_A = self._allocate_empty_buffer()
        self._len_A = torch.zeros(env_size, dtype=torch.int32, device=device)
        self._info_A = self._allocate_info_buffer()

        if not sample_once:
            self._buf_B = self._allocate_empty_buffer()
            self._len_B = torch.zeros(env_size, dtype=torch.int32, device=device)
            self._info_B = self._allocate_info_buffer()

        self._reset_counters = 0
        self._refreshing = False
        self._remaining_mask: torch.Tensor | None = None

        self._populate_buffer_full(target="A")

        self.joint_pos_limit: torch.Tensor | None = None
        self.joint_vel_limit: torch.Tensor | None = None

    def update(self):
        if not self._refreshing:
            # add reset counter
            self._reset_counters += 1

    def reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = env_ids.to(self.device)

        if not self.sample_once:
            if (not self._refreshing) and (self._reset_counters >= self.refresh_threshold):
                self._begin_refresh()

            if self._refreshing:
                self._copy_B_to_A(env_ids)

        return self._len_A[env_ids]

    def get_slice(self,
                  env_ids: torch.Tensor | None,
                  starts: torch.Tensor,
                  steps: int | torch.Tensor = 1) -> "MotionData":
        if env_ids is not None:
            env_ids = env_ids.to(self.device)
        starts = starts.to(self.device)

        if isinstance(steps, int):
            idx = starts.unsqueeze(1) + torch.arange(steps, device=self.device)
        else:
            idx = starts.unsqueeze(1) + steps.to(device=self.device, dtype=torch.long)

        if env_ids is not None:
            idx = idx.clamp(max=(self._len_A[env_ids] - 1).unsqueeze(1))
            sub = self._buf_A[env_ids.unsqueeze(-1), idx]
        else:
            idx = idx.clamp(max=(self._len_A[:] - 1).unsqueeze(1))
            sub = self._buf_A.gather(1, idx)
        sub = self._to_float(sub, dtype=torch.float32)
        return self._post_process(sub)

    def get_slice_info(self, env_ids: torch.Tensor):
        env_ids = env_ids.to(self.device)
        ret = {}
        for k in self.dataset_extra_keys:
            ret[k['name']] = self._info_A[k['name']][env_ids]
        return ret

    def set_limit(self, joint_pos_limit: torch.Tensor, joint_vel_limit: torch.Tensor, joint_names: List[str]):
        self.joint_pos_limit = torch.zeros(1, len(self.joint_names), 2, device=self.device)
        self.joint_vel_limit = torch.zeros(1, len(self.joint_names), 2, device=self.device)

        self.joint_pos_limit[:, :, 0] = -3.14
        self.joint_pos_limit[:, :, 1] = 3.14
        self.joint_vel_limit[:, :, 0] = -10.0
        self.joint_vel_limit[:, :, 1] = 10.0

        for id_asset, name in enumerate(joint_names):
            if name in self.joint_names:
                id_motion = self.joint_names.index(name)
                self.joint_pos_limit[:, id_motion] = joint_pos_limit[0, id_asset]
            else:
                print(f"[warning] joint {name} not found in motion dataset")

    def _allocate_empty_buffer(self) -> "MotionData":
        tpl = self.datasets[0].data
        mm = {}
        for field in tpl.__dataclass_fields__:
            t = getattr(tpl, field)
            if torch.is_floating_point(t):
                mm[field] = torch.zeros(
                    (self.env_size, self.max_step_size) + t.shape[1:],
                    dtype=torch.float16,
                    device=self.device
                )
            else:
                mm[field] = torch.zeros(
                    (self.env_size, self.max_step_size) + t.shape[1:],
                    dtype=t.dtype,
                    device=self.device
                )
        return MotionData(**mm, batch_size=[self.env_size, self.max_step_size], device=self.device)

    def _allocate_info_buffer(self):
        ret = {}
        for k in self.dataset_extra_keys:
            ret[k['name']] = torch.zeros((self.env_size, k['shape']), dtype=k['dtype'], device=self.device)
        return ret

    def _begin_refresh(self):
        self._populate_buffer_full(target="B")
        self._refreshing = True
        self._remaining_mask = torch.ones(self.env_size, dtype=torch.bool, device=self.device)

    def _copy_B_to_A(self, env_ids: torch.Tensor):
        self._buf_A[env_ids] = self._buf_B[env_ids]
        self._len_A[env_ids] = self._len_B[env_ids]
        for k in self.dataset_extra_keys:
            self._info_A[k['name']][env_ids] = self._info_B[k['name']][env_ids]

        self._remaining_mask[env_ids] = False
        if not self._remaining_mask.any():
            self._refreshing = False
            self._reset_counters = 0

    @torch.no_grad()
    def _populate_buffer_full(self, *, target: str):
        assert target in {"A", "B"}
        if target == "A":
            buf, len_buf, info_buf = self._buf_A, self._len_A, self._info_A
        else:
            buf, len_buf, info_buf = self._buf_B, self._len_B, self._info_B

        path_samples = torch.multinomial(self.probs, self.env_size, replacement=True).to(torch.int32)

        if self.fix_ds is not None:
            path_samples[:] = self.fix_ds

        for pi, ds in enumerate(self.datasets):
            mask = (path_samples == pi)
            if not mask.any():
                continue
            cnt = self.counts[pi]
            mids = (torch.rand(mask.sum(), device=self.ds_device) * cnt).floor().to(torch.int32)
            
            if self.fix_motion_id is not None:
                mids[:] = self.fix_motion_id

            mids_long = mids.to(torch.long)
            local_starts = ds.starts[mids_long]
            local_ends = ds.ends[mids_long] - 1
            steps = torch.arange(self.max_step_size, device=self.ds_device, dtype=torch.long)
            local_idx = local_starts.unsqueeze(1) + steps  # (k, max_step)
            local_idx = local_idx.clamp(max=local_ends.unsqueeze(1))

            buf[mask, :self.max_step_size] = self._to_float(ds.data[local_idx].to(self.device), dtype=torch.float16)
            len_buf[mask] = ds.lengths[mids_long].clamp_max(self.max_step_size).to(self.device)

            for k in self.dataset_extra_keys:
                name = k['name']
                info_buf[name][mask] = ds.info[name][mids_long].to(self.device)

    def _post_process(self, data: "MotionData") -> "MotionData":
        data = self._clamp_joint_pos_vel(data)
        data = self._offset_pos_z(data)
        return data

    def _offset_pos_z(self, data: "MotionData", z_offset: float = 0.035):
        data.root_pos_w[..., 2] += z_offset
        data.body_pos_w[..., 2] += z_offset
        return data

    def _clamp_joint_pos_vel(self, data: "MotionData"):
        if self.joint_pos_limit is None:
            return data
        joint_pos = wrap_to_pi(data.joint_pos)
        data.joint_pos[:] = torch.clamp(joint_pos,
                                        self.joint_pos_limit[:, :, 0],
                                        self.joint_pos_limit[:, :, 1])
        data.joint_vel[:] = torch.clamp(data.joint_vel,
                                        self.joint_vel_limit[:, :, 0],
                                        self.joint_vel_limit[:, :, 1])
        return data

    @staticmethod
    def _to_float(data, dtype=torch.float32):
        for f in data.__dataclass_fields__:
            v = getattr(data, f)
            if torch.is_floating_point(v):
                setattr(data, f, v.to(dtype=dtype))
        return data
