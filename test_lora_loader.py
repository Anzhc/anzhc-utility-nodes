import importlib
from pathlib import Path
import sys
import types
import unittest

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
PACKAGE = "anzhc_lora_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
loader = importlib.import_module(PACKAGE + ".lora_loader_node")
svc = importlib.import_module(PACKAGE + ".svc_utils")
adapters = loader.comfy.weight_adapter


def apply_patch(value, base, strength=1.0, offset=None):
    return loader.comfy.lora.calculate_weight([(strength, value, 1.0, offset, None)], base.clone(), "weight")


class SvcLoraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def lora(self, rank=2, alpha=None, shape=(8, 6), dtype=torch.float32):
        up = torch.randn(shape[0], rank, dtype=dtype)
        down = torch.randn(rank, *shape[1:], dtype=dtype)
        if len(shape) > 2:
            up = up.reshape(shape[0], rank, *([1] * (len(shape) - 2)))
        return adapters.LoRAAdapter(set(), (up, down, alpha, None, None, None))

    def check_dense_merge(self, patches, strengths, base, key="weight"):
        offset = key[1] if isinstance(key, tuple) else None
        local_base = base.narrow(*offset) if offset is not None else base
        deltas = [apply_patch(patch, local_base, strength) - local_base if patch is not None else torch.zeros_like(local_base)
                  for patch, strength in zip(patches, strengths)]
        expected = base.clone()
        expected_target = expected.narrow(*offset) if offset is not None else expected
        expected_target += svc.merge_svc_delta_tensors(deltas)
        merged = loader._merge_svc_patch_values(key, patches, strengths, base)
        torch.testing.assert_close(apply_patch(merged, base, offset=offset), expected, atol=2e-5, rtol=2e-5)
        return merged

    def test_equivalent_factorizations_do_not_cancel(self):
        first = self.lora(alpha=1.0)
        up, down = first.weights[:2]
        second = adapters.LoRAAdapter(set(), (-up, -down, 1.0, None, None, None))
        base = torch.zeros(8, 6)
        merged = loader._merge_svc_patch_values("weight", [first, second], [0.7, 0.7], base)
        torch.testing.assert_close(apply_patch(merged, base), apply_patch(first, base, 0.7), atol=2e-6, rtol=2e-6)

    def test_orthogonal_updates_preserve_both_strengths(self):
        first = adapters.LoRAAdapter(set(), (torch.tensor([[1.], [0.]]), torch.tensor([[1., 0.]]), None, None, None, None))
        second = adapters.LoRAAdapter(set(), (torch.tensor([[0.], [1.]]), torch.tensor([[0., 1.]]), None, None, None, None))
        base = torch.zeros(2, 2)
        merged = loader._merge_svc_patch_values("weight", [first, second], [0.6, 0.8], base)
        torch.testing.assert_close(apply_patch(merged, base), torch.diag(torch.tensor([0.6, 0.8])))

    def test_mixed_ranks_alpha_and_signed_strengths(self):
        self.check_dense_merge([self.lora(2, 0.5), self.lora(3, 6.0)], [0.8, -0.4], torch.zeros(8, 6))

    def test_missing_and_disabled_patches_preserve_unique_update(self):
        first = self.lora(alpha=1.0)
        base = torch.zeros(8, 6)
        merged = self.check_dense_merge([first, None, self.lora(3)], [0.7, 0.0, 0.0], base)
        torch.testing.assert_close(apply_patch(merged, base), apply_patch(first, base, 0.7), atol=2e-6, rtol=2e-6)

    def test_bfloat16_weights_are_calibrated_in_float32(self):
        merged = self.check_dense_merge([self.lora(dtype=torch.bfloat16), self.lora(dtype=torch.bfloat16)], [0.3, 1.2], torch.zeros(8, 6))
        self.assertEqual(merged.weights[0].dtype, torch.float32)

    def test_convolution_and_fused_projection_offsets(self):
        shape = (8, 3, 3, 3)
        self.check_dense_merge([self.lora(shape=shape), self.lora(3, shape=shape)], [0.7, 0.3], torch.zeros(shape))
        self.check_dense_merge([self.lora(), self.lora(3)], [0.7, 0.3], torch.zeros(24, 6), ("weight", (0, 8, 8)))

    def test_loha_lokr_and_mixed_diff_use_actual_deltas(self):
        loha = adapters.LoHaAdapter(set(), (torch.randn(8, 2), torch.randn(2, 6), 1.0,
                                            torch.randn(8, 2), torch.randn(2, 6), None, None, None))
        lokr = adapters.LoKrAdapter(set(), (torch.randn(2, 2), torch.randn(4, 3), None, None, None, None, None, None, None))
        base = torch.zeros(8, 6)
        self.check_dense_merge([loha, lokr], [0.7, 0.4], base)
        self.check_dense_merge([self.lora(), ("diff", (torch.randn(8, 6),))], [0.7, -0.4], base)

    def test_dora_uses_base_weight_and_own_scale(self):
        first = self.lora(alpha=1.0)
        first.weights = (*first.weights[:4], torch.rand(8, 1) + 1.0, None)
        self.check_dense_merge([first, self.lora(3)], [0.6, 0.4], torch.randn(8, 6))

    def test_reshape_adapter_preserves_weight_padding(self):
        patches = [self.lora(shape=(8, 6)), self.lora(3, shape=(8, 6))]
        for patch in patches:
            patch.weights = (*patch.weights[:5], (8, 6))
        base = torch.zeros(6, 4)
        merged = loader._merge_svc_patch_values("weight", patches, [0.6, 0.4], base)
        expected = svc.merge_svc_delta_tensors([apply_patch(patch, base, strength) for patch, strength in zip(patches, [0.6, 0.4])])
        torch.testing.assert_close(apply_patch(merged, base), expected)

    def test_model_clip_and_group_strengths(self):
        key = "weight"
        patches = [self.lora(alpha=1.0), self.lora(3, 6.0)]
        entries = []
        for patch, strength_model, strength_clip in zip(patches, [0.8, -0.4], [0.2, 0.9]):
            groups = {group: {} for group in loader._BLOCK_WEIGHT_KEYS}
            groups["others"][key] = patch
            entries.append(loader._LoadedLora("test", "test", {"others": 0.5}, {key: patch}, groups,
                                              strength_model, strength_clip, True, True))
        base = torch.zeros(8, 6)
        for attr in ("strength_model", "strength_clip"):
            merged = loader._merge_svc_lora_patches_for_state_dict(entries, {key: base}, attr)[key]
            deltas = [apply_patch(entry.loaded_patches[key], base, getattr(entry, attr) * 0.5) for entry in entries]
            torch.testing.assert_close(apply_patch(merged, base), svc.merge_svc_delta_tensors(deltas), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
