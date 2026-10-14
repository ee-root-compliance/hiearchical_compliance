import unittest

import torch

from active_adaptation.learning.hierarchical.root_student_force_moe import (
    EXPERT_NAMES,
    compose_decoded_expert_actions,
    inverse_stiffness_gate_weights,
)


class RootStudentForceMoETest(unittest.TestCase):
    def test_inverse_stiffness_gate_hits_all_anchors(self):
        stiffness = torch.tensor([[200.0, 400.0, 600.0]])
        weights = inverse_stiffness_gate_weights(stiffness)

        expected = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
        )
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(stiffness))

    def test_inverse_stiffness_gate_interpolates_in_compliance_space(self):
        stiffness = torch.tensor([[300.0, 500.0, 600.0]])
        weights = inverse_stiffness_gate_weights(stiffness)

        torch.testing.assert_close(weights[0, 0], torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0]))
        torch.testing.assert_close(weights[0, 1], torch.tensor([0.6, 0.4, 0.0]))
        torch.testing.assert_close(weights[0, 2], torch.tensor([1.0, 0.0, 0.0]))

    def test_inverse_stiffness_gate_extrapolates_below_low_anchor(self):
        stiffness = torch.tensor([[100.0, 600.0, 200.0]])
        weights = inverse_stiffness_gate_weights(
            stiffness,
            allow_extrapolation=True,
        )

        torch.testing.assert_close(weights[0, 0], torch.tensor([0.0, -2.0, 3.0]))
        torch.testing.assert_close(weights[0, 1], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(weights[0, 2], torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(stiffness))

    def test_inverse_stiffness_gate_supports_four_anchors(self):
        stiffness = torch.tensor([[100.0, 150.0, 200.0], [300.0, 400.0, 600.0]])
        weights = inverse_stiffness_gate_weights(
            stiffness,
            anchors=(100.0, 200.0, 400.0, 600.0),
        )

        expected = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 2.0 / 3.0, 1.0 / 3.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(stiffness))

    def test_axis_component_composition_uses_matching_expert_per_hand(self):
        raw_actions = {name: torch.zeros(1, 6) for name in EXPERT_NAMES}
        raw_actions["x_200"][:, [0, 3]] = torch.atanh(torch.tensor(0.8))
        raw_actions["y_400"][:, [1, 4]] = torch.atanh(torch.tensor(0.4))
        raw_actions["z_200"][:, [2, 5]] = torch.atanh(torch.tensor(-0.3))
        weights = inverse_stiffness_gate_weights(torch.tensor([[200.0, 400.0, 200.0]]))

        raw_action, decoded_action = compose_decoded_expert_actions(raw_actions, weights)

        expected = torch.tensor([[0.8, 0.4, -0.3, 0.8, 0.4, -0.3]])
        torch.testing.assert_close(decoded_action, expected)
        torch.testing.assert_close(torch.tanh(raw_action), expected)

    def test_full_residual_recovers_single_axis_expert_at_anchor(self):
        raw_actions = {name: torch.zeros(1, 6) for name in EXPERT_NAMES}
        desired = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]])
        raw_actions["x_200"] = torch.atanh(desired)
        weights = inverse_stiffness_gate_weights(torch.tensor([[200.0, 600.0, 600.0]]))

        raw_action, decoded_action = compose_decoded_expert_actions(
            raw_actions,
            weights,
            mode="full_residual",
        )

        torch.testing.assert_close(decoded_action, desired)
        torch.testing.assert_close(torch.tanh(raw_action), desired)

    def test_physical_space_composition_preserves_expert_meter_scale(self):
        raw_actions = {name: torch.zeros(1, 6) for name in EXPERT_NAMES}
        raw_actions["x_200"][:, [0, 3]] = torch.atanh(torch.tensor(0.8))
        weights = inverse_stiffness_gate_weights(
            torch.tensor([[100.0, 600.0, 600.0]]),
            anchors=(100.0, 350.0, 600.0),
        )
        scales = {name: [0.15, 0.15, 0.15] for name in EXPERT_NAMES}
        scales["x_200"] = [0.35, 0.15, 0.15]

        raw_action, decoded_action = compose_decoded_expert_actions(
            raw_actions,
            weights,
            compose_in_physical_space=True,
            expert_pos_scales=scales,
            output_pos_scale=[0.35, 0.35, 0.35],
        )

        expected = torch.tensor([[0.8, 0.0, 0.0, 0.8, 0.0, 0.0]])
        torch.testing.assert_close(decoded_action, expected)
        torch.testing.assert_close(torch.tanh(raw_action), expected)

    def test_four_anchor_composition_supports_axiswise_expert_scales(self):
        expert_names = (
            "baseline_600",
            "x_400",
            "x_200",
            "x_100",
            "y_400",
            "y_200",
            "y_100",
            "z_400",
            "z_200",
            "z_100",
        )
        raw_actions = {name: torch.zeros(1, 6) for name in expert_names}
        raw_actions["x_100"][:, [0, 3]] = torch.atanh(torch.tensor(0.8))
        weights = inverse_stiffness_gate_weights(
            torch.tensor([[100.0, 600.0, 600.0]]),
            anchors=(100.0, 200.0, 400.0, 600.0),
        )
        scales = {name: [0.15, 0.15, 0.15] for name in expert_names}
        scales["x_100"] = [0.35, 0.15, 0.15]

        raw_action, decoded_action = compose_decoded_expert_actions(
            raw_actions,
            weights,
            compose_in_physical_space=True,
            expert_pos_scales=scales,
            output_pos_scale=[0.35, 0.35, 0.35],
            expert_names=expert_names,
            axis_expert_names=(
                ("x_400", "x_200", "x_100"),
                ("y_400", "y_200", "y_100"),
                ("z_400", "z_200", "z_100"),
            ),
        )

        expected = torch.tensor([[0.8, 0.0, 0.0, 0.8, 0.0, 0.0]])
        torch.testing.assert_close(decoded_action, expected)
        torch.testing.assert_close(torch.tanh(raw_action), expected)

    def test_four_anchor_physical_space_scales_baseline_at_600(self):
        expert_names = (
            "baseline_600",
            "x_400",
            "x_200",
            "x_100",
            "y_400",
            "y_200",
            "y_100",
            "z_400",
            "z_200",
            "z_100",
        )
        raw_actions = {name: torch.zeros(1, 6) for name in expert_names}
        raw_actions["baseline_600"][:] = torch.atanh(torch.tensor(0.8))
        weights = inverse_stiffness_gate_weights(
            torch.tensor([[600.0, 600.0, 600.0]]),
            anchors=(100.0, 200.0, 400.0, 600.0),
        )
        scales = {name: [0.15, 0.15, 0.15] for name in expert_names}

        _, decoded_action = compose_decoded_expert_actions(
            raw_actions,
            weights,
            compose_in_physical_space=True,
            expert_pos_scales=scales,
            output_pos_scale=[0.35, 0.35, 0.35],
            expert_names=expert_names,
            axis_expert_names=(
                ("x_400", "x_200", "x_100"),
                ("y_400", "y_200", "y_100"),
                ("z_400", "z_200", "z_100"),
            ),
        )

        # A 600-N/m gate must reproduce the 0.15-m baseline residual after
        # converting it to the outer MoE's 0.35-m action scale.
        expected = torch.full((1, 6), 0.8 * 0.15 / 0.35)
        torch.testing.assert_close(decoded_action, expected)


if __name__ == "__main__":
    unittest.main()
