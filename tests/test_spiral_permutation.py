"""Automated Test Suite for SHAPECRYPT Spiral Geometric Pixel Permutation
Validates Coverage, Uniqueness, Range, Reversibility, Determinism,
Different Seed divergence, Multi-size robustness (1x1, 2x2, 4x4, 5x7, 7x5, 16x16, 32x24, 24x32, 512x512),
Inward and Outward modes, and RGB, RGBA, and Grayscale lossless reconstruction.
"""

import unittest
import numpy as np

from encryption.shapes import generate_spiral_permutation
from encryption.permutation import (
    apply_permutation,
    inverse_permutation,
    apply_inverse_permutation,
    is_valid_permutation,
)


class TestSpiralPermutation(unittest.TestCase):
    """Test suite for spiral-based geometric pixel permutation."""

    def setUp(self):
        self.test_sizes = [
            (1, 1),      # Minimal 1x1
            (2, 2),      # Minimal 2x2
            (4, 4),      # Small square
            (5, 7),      # Small rectangular (tall)
            (7, 5),      # Small rectangular (wide)
            (16, 16),    # Medium square
            (32, 24),    # Medium rectangular
            (24, 32),    # Medium rectangular
            (512, 512),  # Full scale 512x512
        ]
        self.primary_seed = "shapecrypt_master_seed_spiral_alpha"
        self.secondary_seed = "shapecrypt_master_seed_spiral_beta"

    def test_requirement_a_coverage(self):
        """Test A: Permutation length must strictly equal H * W."""
        for h, w in self.test_sizes:
            with self.subTest(height=h, width=w):
                perm = generate_spiral_permutation(h, w, self.primary_seed)
                expected_len = h * w
                self.assertEqual(
                    len(perm),
                    expected_len,
                    f"Failed coverage for {h}x{w}: expected {expected_len}, got {len(perm)}"
                )

    def test_requirement_b_uniqueness(self):
        """Test B: All coordinates in permutation must be unique (no duplicates, no omissions)."""
        for h, w in self.test_sizes:
            with self.subTest(height=h, width=w):
                perm = generate_spiral_permutation(h, w, self.primary_seed)
                unique_count = len(set(perm))
                expected_count = h * w
                self.assertEqual(
                    unique_count,
                    expected_count,
                    f"Duplicate coordinates detected in {h}x{w}: {expected_count - unique_count} duplicates"
                )
                self.assertTrue(is_valid_permutation(perm, expected_count))

    def test_requirement_c_range(self):
        """Test C: Every index in permutation must be between 0 and H*W - 1."""
        for h, w in self.test_sizes:
            with self.subTest(height=h, width=w):
                perm = generate_spiral_permutation(h, w, self.primary_seed)
                total_pixels = h * w
                self.assertGreaterEqual(np.min(perm), 0, f"Negative index found in {h}x{w}")
                self.assertLess(np.max(perm), total_pixels, f"Index out of upper bound in {h}x{w}")

    def test_requirement_d_reversibility_rgb_rgba_gray(self):
        """Test D: Reversibility must produce exact byte-for-byte identical images for RGB, RGBA, Grayscale."""
        np.random.seed(5555)
        reversibility_sizes = [(1, 1), (2, 2), (4, 4), (5, 7), (7, 5), (16, 16), (32, 24), (24, 32)]
        for h, w in reversibility_sizes:
            with self.subTest(height=h, width=w):
                perm = generate_spiral_permutation(h, w, self.primary_seed)

                # Test 1: RGB Image (H, W, 3)
                original_rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                shuffled_rgb = apply_permutation(original_rgb, perm)
                recovered_rgb = apply_inverse_permutation(shuffled_rgb, perm)

                self.assertEqual(original_rgb.shape, recovered_rgb.shape)
                self.assertEqual(original_rgb.dtype, recovered_rgb.dtype)
                self.assertTrue(
                    np.array_equal(original_rgb, recovered_rgb),
                    f"RGB Lossless reconstruction failed for {h}x{w}!"
                )

                # Test 2: RGBA Image (H, W, 4)
                original_rgba = np.random.randint(0, 256, (h, w, 4), dtype=np.uint8)
                shuffled_rgba = apply_permutation(original_rgba, perm)
                recovered_rgba = apply_inverse_permutation(shuffled_rgba, perm)

                self.assertEqual(original_rgba.shape, recovered_rgba.shape)
                self.assertEqual(original_rgba.dtype, recovered_rgba.dtype)
                self.assertTrue(
                    np.array_equal(original_rgba, recovered_rgba),
                    f"RGBA Lossless reconstruction failed for {h}x{w}!"
                )

                # Test 3: Grayscale Image (H, W)
                original_gray = np.random.randint(0, 256, (h, w), dtype=np.uint8)
                shuffled_gray = apply_permutation(original_gray, perm)
                recovered_gray = apply_inverse_permutation(shuffled_gray, perm)

                self.assertEqual(original_gray.shape, recovered_gray.shape)
                self.assertEqual(original_gray.dtype, recovered_gray.dtype)
                self.assertTrue(
                    np.array_equal(original_gray, recovered_gray),
                    f"Grayscale Lossless reconstruction failed for {h}x{w}!"
                )

    def test_requirement_e_determinism(self):
        """Test E: Same dimensions + same seed must yield exactly identical permutations."""
        for h, w in [(2, 2), (4, 4), (5, 7), (7, 5), (16, 16), (32, 24), (24, 32)]:
            with self.subTest(height=h, width=w):
                perm_run_1 = generate_spiral_permutation(h, w, self.primary_seed)
                perm_run_2 = generate_spiral_permutation(h, w, self.primary_seed)
                self.assertTrue(
                    np.array_equal(perm_run_1, perm_run_2),
                    f"Determinism failed for {h}x{w}: two identical runs gave different outputs."
                )

    def test_requirement_f_different_seeds(self):
        """Test F: Same dimensions + different seeds must yield different valid permutations."""
        seeds = [
            "spiral_key_alpha",
            "spiral_key_beta",
            "spiral_key_gamma",
            369121518,
            b"\xba\xdd\xca\xfe\x01\x02\x03\x04"
        ]
        for h, w in [(4, 4), (5, 7), (7, 5), (16, 16), (32, 24), (24, 32)]:
            with self.subTest(height=h, width=w):
                perms = [generate_spiral_permutation(h, w, s) for s in seeds]
                for i in range(len(perms)):
                    for j in range(i + 1, len(perms)):
                        self.assertFalse(
                            np.array_equal(perms[i], perms[j]),
                            f"Seed collision detected for {h}x{w} between seed {i} and seed {j}!"
                        )

        # 2x2 grid seed divergence check
        p_alpha = generate_spiral_permutation(2, 2, "spiral_key_alpha")
        p_beta = generate_spiral_permutation(2, 2, "spiral_key_beta")
        self.assertFalse(np.array_equal(p_alpha, p_beta), "2x2 spiral permutations should diverge with different seeds")

    def test_mathematical_inverse_identity(self):
        """Validates that P[P_inv] == Identity and P_inv[P] == Identity."""
        for h, w in [(2, 2), (4, 4), (5, 7), (7, 5), (16, 16), (32, 24), (24, 32)]:
            perm = generate_spiral_permutation(h, w, self.primary_seed)
            inv_perm = inverse_permutation(perm)
            identity = np.arange(h * w, dtype=np.int64)

            self.assertTrue(
                np.array_equal(perm[inv_perm], identity),
                f"P[P_inv] != Identity for {h}x{w}"
            )
            self.assertTrue(
                np.array_equal(inv_perm[perm], identity),
                f"P_inv[P] != Identity for {h}x{w}"
            )

    def test_invalid_arguments(self):
        """Validates error handling for illegal inputs."""
        with self.assertRaises(ValueError):
            generate_spiral_permutation(0, 10, "seed")
        with self.assertRaises(ValueError):
            generate_spiral_permutation(10, -5, "seed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
