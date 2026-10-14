import unittest

import torch

from active_adaptation.utils.ee_compliance import (
    force_stiffness_consistency_reward,
    sample_mixed_xyz_stiffness,
)


class EEStiffnessRangeV2Test(unittest.TestCase):
    def test_mixed_xyz_stiffness_sampling_uses_all_three_branches(self):
        torch.manual_seed(7)
        sampled = sample_mixed_xyz_stiffness(
            20_000,
            torch.tensor([200.0, 200.0, 200.0]),
            torch.tensor([600.0, 600.0, 600.0]),
            levels=torch.tensor([300.0]),
            levels_prob=0.30,
            anchors=torch.tensor([[200.0, 600.0, 600.0]]),
            anchor_prob=0.20,
        )

        anchor_fraction = (sampled == torch.tensor([200.0, 600.0, 600.0])).all(-1).float().mean()
        level_fraction = (sampled == 300.0).all(-1).float().mean()
        continuous_fraction = 1.0 - anchor_fraction - level_fraction
        self.assertAlmostEqual(anchor_fraction.item(), 0.20, delta=0.02)
        self.assertAlmostEqual(level_fraction.item(), 0.30, delta=0.02)
        self.assertAlmostEqual(continuous_fraction.item(), 0.50, delta=0.02)

    def test_force_stiffness_reward_has_correct_sign_and_inactive_mask(self):
        nominal_b = torch.zeros(2, 2, 3)
        force_b = torch.zeros(2, 2, 3)
        force_b[0, 0, 0] = 10.0
        stiffness = torch.full((2, 1, 3), 200.0)

        actual_b = torch.zeros(2, 2, 3)
        actual_b[0, 0, 0] = 0.05  # F / K = 10 / 200.

        values, active = force_stiffness_consistency_reward(
            actual_b,
            nominal_b,
            force_b,
            stiffness.expand(-1, 2, -1),
            force_scale=15.0,
            sigma=1.0,
            force_deadband=0.5,
        )
        torch.testing.assert_close(values[0], torch.ones(1))
        self.assertTrue(active[0].item())
        self.assertFalse(active[1].item())

        # Deflecting in the opposite direction must be penalized, which catches
        # accidental force/deflection sign reversal.
        actual_b[0, 0, 0] = -0.05
        wrong_values, _ = force_stiffness_consistency_reward(
            actual_b,
            nominal_b,
            force_b,
            stiffness.expand(-1, 2, -1),
            force_scale=15.0,
            sigma=1.0,
            force_deadband=0.5,
        )
        self.assertLess(wrong_values[0].item(), values[0].item())


if __name__ == "__main__":
    unittest.main()
