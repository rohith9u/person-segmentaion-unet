"""Automated tests for U-Net architecture, forward/backward passes, and skip connections."""

import unittest
import torch
import torch.nn as nn

from src.models.unet import DoubleConv, UNet


class TestUNetArchitecture(unittest.TestCase):
    """Test suite validating U-Net architectural specifications."""

    def setUp(self):
        self.model = UNet(in_channels=3, out_channels=1)

    def test_01_output_shape_batch(self):
        """Verify U-Net accepts [B, 3, 256, 256] and outputs [B, 1, 256, 256]."""
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 256, 256)
            out = self.model(x)
            self.assertEqual(
                out.shape,
                (batch_size, 1, 256, 256),
                f"Expected shape {(batch_size, 1, 256, 256)}, got {out.shape}",
            )

    def test_02_forward_backward_pass_finite_loss(self):
        """Verify forward and backward passes execute cleanly with finite BCE loss and valid gradients."""
        x = torch.randn(2, 3, 256, 256)
        target = torch.randint(0, 2, (2, 1, 256, 256)).float()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        logits = self.model(x)
        loss = criterion(logits, target)

        self.assertTrue(torch.isfinite(loss), f"Loss is not finite: {loss.item()}")
        loss.backward()

        # Check gradients exist on all trainable parameters
        grad_norms = []
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient!")
            self.assertTrue(torch.all(torch.isfinite(param.grad)), f"Parameter {name} gradient contains inf/nan!")
            grad_norms.append(param.grad.norm().item())

        self.assertGreater(sum(grad_norms), 0.0, "Total gradient norm is zero.")

    def test_03_skip_connections_and_channels(self):
        """Verify all intermediate encoder, bottleneck, and decoder tensor dimensions."""
        x = torch.randn(1, 3, 256, 256)

        # Encoder
        x1 = self.model.enc1(x)
        self.assertEqual(x1.shape, (1, 32, 256, 256))
        p1 = self.model.pool1(x1)
        self.assertEqual(p1.shape, (1, 32, 128, 128))

        x2 = self.model.enc2(p1)
        self.assertEqual(x2.shape, (1, 64, 128, 128))
        p2 = self.model.pool2(x2)
        self.assertEqual(p2.shape, (1, 64, 64, 64))

        x3 = self.model.enc3(p2)
        self.assertEqual(x3.shape, (1, 128, 64, 64))
        p3 = self.model.pool3(x3)
        self.assertEqual(p3.shape, (1, 128, 32, 32))

        x4 = self.model.enc4(p3)
        self.assertEqual(x4.shape, (1, 256, 32, 32))
        p4 = self.model.pool4(x4)
        self.assertEqual(p4.shape, (1, 256, 16, 16))

        # Bottleneck
        b = self.model.bottleneck(p4)
        self.assertEqual(b.shape, (1, 512, 16, 16))

        # Decoder Step 1
        u1 = self.model.up1(b)
        self.assertEqual(u1.shape, (1, 256, 32, 32))
        cat1 = torch.cat([u1, x4], dim=1)
        self.assertEqual(cat1.shape, (1, 512, 32, 32))
        d1 = self.model.dec1(cat1)
        self.assertEqual(d1.shape, (1, 256, 32, 32))

        # Decoder Step 2
        u2 = self.model.up2(d1)
        self.assertEqual(u2.shape, (1, 128, 64, 64))
        cat2 = torch.cat([u2, x3], dim=1)
        self.assertEqual(cat2.shape, (1, 256, 64, 64))
        d2 = self.model.dec2(cat2)
        self.assertEqual(d2.shape, (1, 128, 64, 64))

        # Decoder Step 3
        u3 = self.model.up3(d2)
        self.assertEqual(u3.shape, (1, 64, 128, 128))
        cat3 = torch.cat([u3, x2], dim=1)
        self.assertEqual(cat3.shape, (1, 128, 128, 128))
        d3 = self.model.dec3(cat3)
        self.assertEqual(d3.shape, (1, 64, 128, 128))

        # Decoder Step 4
        u4 = self.model.up4(d3)
        self.assertEqual(u4.shape, (1, 32, 256, 256))
        cat4 = torch.cat([u4, x1], dim=1)
        self.assertEqual(cat4.shape, (1, 64, 256, 256))
        d4 = self.model.dec4(cat4)
        self.assertEqual(d4.shape, (1, 32, 256, 256))

        # Final Conv
        logits = self.model.final_conv(d4)
        self.assertEqual(logits.shape, (1, 1, 256, 256))

    def test_04_parameter_count(self):
        """Verify model parameter count is within expected U-Net range (~7.8M parameters)."""
        param_count = self.model.count_parameters()
        self.assertGreater(param_count, 7_000_000)
        self.assertLess(param_count, 9_000_000)


if __name__ == "__main__":
    unittest.main()
