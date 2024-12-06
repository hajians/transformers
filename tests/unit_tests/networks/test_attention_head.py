import unittest

import numpy as np
import torch

from transformers_tutorial.networks.attention_head import AttentionHead


class TestAttentionHead(unittest.TestCase):

    def setUp(self):
        # Initialize an AttentionHead instance
        self.emb_dim = 4
        self.hiddin_dim = 4
        self.attention_head = AttentionHead(self.emb_dim, self.hiddin_dim)

        # Sample inputs
        self.batch_size = 2
        self.seq_len = 3
        self.hidden_state = torch.rand(self.batch_size, self.seq_len, self.emb_dim)
        self.attention_mask = torch.tensor(
            [[1, 1, 1], [1, 1, 0]]
        )  # Shape: (batch_size, seq_len)

    def test_compute_weights(self):
        # Generate query and key tensors
        q = torch.rand(self.batch_size, self.seq_len, self.hiddin_dim)
        k = torch.rand(self.batch_size, self.seq_len, self.hiddin_dim)

        # Call compute_weights
        weights = AttentionHead.compute_weights(
            q, k, attention_mask=self.attention_mask
        )

        # Assert the shape
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))

        # Assert weight for padding
        np.testing.assert_almost_equal(weights[1][:, -1].detach().numpy(), [0, 0, 1])
        np.testing.assert_almost_equal(weights[1][-1, :].detach().numpy(), [0, 0, 1])

        # Assert softmax probabilities sum to 1 along the last dimension
        sums = torch.sum(weights, dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_scaled_dot_product(self):
        # Generate query, key, and value tensors
        q = torch.rand(self.batch_size, self.seq_len, self.hiddin_dim)
        k = torch.rand(self.batch_size, self.seq_len, self.hiddin_dim)
        v = torch.rand(self.batch_size, self.seq_len, self.hiddin_dim)

        # Call scaled_dot_product
        output = AttentionHead.scaled_dot_product(
            q, k, v, attention_mask=self.attention_mask
        )

        # Assert the shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hiddin_dim))

    def test_forward(self):
        # Call forward method
        output = self.attention_head(
            self.hidden_state, attention_masks=self.attention_mask
        )

        # Assert the shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hiddin_dim))

        # Ensure output is a valid tensor
        self.assertTrue(torch.is_tensor(output))
