import unittest

import numpy as np
import torch
from parameterized import parameterized

from transformers_tutorial.networks.attention_head import (
    AttentionHead,
    MultiHeadAttention,
)

TEST_TENSOR = torch.tensor([[1, 0, 0], [0, 1, 0], [4, 4, 4]]).unsqueeze(0).float()


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

    @parameterized.expand(
        [
            (
                "without_attention",
                {
                    "q": TEST_TENSOR,
                    "v": TEST_TENSOR,
                    "k": TEST_TENSOR,
                    "attention_mask": None,
                },
                torch.softmax(
                    torch.Tensor([[1, 0, 4], [0, 1, 4], [4, 4, 48]]) / np.sqrt(3),
                    dim=-1,
                ).unsqueeze(0),
            ),
            (
                "with_attention",
                {
                    "q": TEST_TENSOR,
                    "v": TEST_TENSOR,
                    "k": TEST_TENSOR,
                    "attention_mask": torch.tensor([[1, 1, 0]]),
                },
                torch.softmax(
                    torch.Tensor([[1, 0, -1e5], [0, 1, -1e5], [-1e5, -1e5, 48]])
                    / np.sqrt(3),
                    dim=-1,
                ).unsqueeze(0),
            ),
        ]
    )
    def test_another_compute_weights(self, name, input_, expected):
        # Given
        q, k, v = input_["q"], input_["k"], input_["v"]
        attention_mask = input_["attention_mask"]

        # When
        output = AttentionHead.compute_weights(q, k, attention_mask=attention_mask)

        # Then
        np.testing.assert_almost_equal(
            output.detach().numpy(),
            expected.numpy(),
            decimal=4,
        )

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


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        # Define parameters for the MultiHeadAttention
        self.emb_dim = 32
        self.hidden_dim = 16
        self.n_heads = 4
        self.seq_len = 8
        self.batch_size = 2

        # Initialize the MultiHeadAttention module
        self.mha = MultiHeadAttention(self.emb_dim, self.hidden_dim, self.n_heads)

        # Sample inputs
        self.hidden_state = torch.rand((self.batch_size, self.seq_len, self.emb_dim))
        self.attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32
        )  # Shape: (batch_size, seq_len)

    def test_output_shape(self):
        # Test that output has the correct shape
        output = self.mha(self.hidden_state)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))

    def test_num_params(self):
        params = [_[0] for _ in self.mha.named_parameters()]

        n_modules_per_head = 2 * 3  # (bias + weight) times (q, k, v)
        n_dense_modules = 2  # (bias + weight)

        self.assertEqual(
            len(params), self.mha.n_heads * n_modules_per_head + n_dense_modules
        )
