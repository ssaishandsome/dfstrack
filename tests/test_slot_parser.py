import unittest

import torch

from lib.models.dfstrack.slot_parser import SlotParser


class SlotParserTest(unittest.TestCase):
    def test_shapes_and_normalization(self):
        batch_size, num_tokens, dim, num_slots = 2, 5, 8, 3
        parser = SlotParser(num_slots=num_slots, dim=dim, temperature=0.7)
        tokens = torch.randn(batch_size, num_tokens, dim)

        slots, assignment = parser(tokens)

        self.assertEqual(slots.shape, (batch_size, num_slots, dim))
        self.assertEqual(assignment.shape, (batch_size, num_tokens, num_slots))
        self.assertTrue(
            torch.allclose(
                assignment.sum(dim=-1),
                torch.ones(batch_size, num_tokens),
                atol=1e-5,
            )
        )

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            SlotParser(num_slots=3, dim=8, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
