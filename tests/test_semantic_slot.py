import unittest

import torch

from lib.models.dfstrack.reliability_head import ReliabilityHead
from lib.models.dfstrack.semantic_slot import SemanticSlotTracker


class ReliabilityHeadTest(unittest.TestCase):
    def test_focus_matches_entropy_intuition(self):
        head = ReliabilityHead(dim=8, hidden_dim=16)
        slot_attention = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                ]
            ]
        )

        focus = head.compute_focus(slot_attention)

        self.assertTrue(torch.allclose(focus[0, 0], torch.tensor(1.0), atol=1e-5))
        self.assertTrue(torch.allclose(focus[0, 1], torch.tensor(0.0), atol=1e-5))

    def test_forward_shapes(self):
        head = ReliabilityHead(dim=8, hidden_dim=16)
        hz = torch.randn(2, 3, 8)
        h_tilde = torch.randn(2, 3, 8)
        slot_attention = torch.softmax(torch.randn(2, 3, 5), dim=-1)

        reliability, aux = head(hz=hz, h_tilde=h_tilde, slot_attention=slot_attention)

        self.assertEqual(reliability.shape, (2, 3))
        self.assertEqual(aux["slot_focus"].shape, (2, 3))
        self.assertEqual(aux["slot_similarity"].shape, (2, 3))
        self.assertTrue(((reliability >= 0.0) & (reliability <= 1.0)).all())


class SemanticSlotTrackerTest(unittest.TestCase):
    def test_slot_pipeline_steps_and_assignment_normalization(self):
        tracker = SemanticSlotTracker(
            num_slots=3,
            dim=8,
            num_heads=2,
            hidden_dim=16,
            reliability_hidden_dim=12,
        )

        text_tokens = torch.randn(2, 5, 8)
        template_tokens = torch.randn(2, 6, 8)
        search_tokens = torch.randn(2, 7, 8)
        slot_candidate = torch.randn(2, 3, 8)
        slot_attention = torch.softmax(torch.randn(2, 3, 7), dim=-1)
        text_mask = torch.tensor(
            [
                [False, False, False, False, True],
                [False, False, False, False, False],
            ]
        )

        slot_prior, init_aux = tracker.initialize_slots(
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        template_slots, template_aux = tracker.constrain_slots(
            slot_prior=slot_prior,
            template_tokens=template_tokens,
        )
        corrected_slots, correction_aux = tracker.correct_slots(
            template_slots=template_slots,
            slot_candidate=slot_candidate,
            slot_attention=slot_attention,
        )
        modulated_search, modulation_aux = tracker.modulate_search(
            search_tokens=search_tokens,
            corrected_slots=corrected_slots,
        )

        self.assertEqual(slot_prior.shape, (2, 3, 8))
        self.assertEqual(template_slots.shape, (2, 3, 8))
        self.assertEqual(corrected_slots.shape, (2, 3, 8))
        self.assertEqual(modulated_search.shape, (2, 7, 8))
        self.assertEqual(init_aux["text_slot_attention"].shape, (2, 3, 5))
        self.assertEqual(template_aux["template_slot_attention"].shape, (2, 3, 6))
        self.assertEqual(correction_aux["slot_state"].shape, (2, 3, 8))
        self.assertEqual(correction_aux["slot_candidate"].shape, (2, 3, 8))
        self.assertEqual(correction_aux["slot_attention"].shape, (2, 3, 7))
        self.assertEqual(correction_aux["slot_assignment"].shape, (2, 7, 3))
        self.assertEqual(correction_aux["slot_reliability"].shape, (2, 3))
        self.assertEqual(modulation_aux["search_slot_attention"].shape, (2, 7, 3))
        self.assertTrue(
            torch.allclose(
                correction_aux["slot_assignment"].sum(dim=-1),
                torch.ones(2, 7),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
