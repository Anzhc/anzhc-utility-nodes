import importlib
from dataclasses import replace
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
model_patcher = importlib.import_module("comfy.model_patcher")


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

    def test_average_matches_normal_partial_application_for_model_and_clip(self):
        entries = []
        for index, (strength_model, strength_clip) in enumerate(((1.0, 0.6), (-0.3, 0.0), (0.7, 1.2), (0.5, 0.4))):
            patch = self.lora(rank=index + 1, alpha=2.0)
            groups = {group: {} for group in loader._BLOCK_WEIGHT_KEYS}
            groups["others"]["weight"] = patch
            entries.append(loader._LoadedLora(str(index), str(index), {"others": 0.8}, {"weight": patch}, groups,
                                              strength_model, strength_clip, True, index < 3))

        for method, marked in (("average", 3), ("none", 3), ("average", 1), ("average", 0)):
            with self.subTest(method=method, marked=marked):
                selected = [replace(entry, merge=index < marked) for index, entry in enumerate(entries)]
                node = loader.AnzhcLoraLoader()
                node._collect_loras = lambda model, clip, kwargs: (selected, [entry.name for entry in selected])
                model = model_patcher.ModelPatcher(torch.nn.Linear(6, 8, bias=False),
                                                   torch.device("cpu"), torch.device("cpu"))
                clip = model.clone()
                out_model, out_clip, names = node.load_loras(model=model, clip=clip, merge_method=method)
                self.assertEqual(names, "0, 1, 2, 3")
                for output, attr in ((out_model, "strength_model"), (out_clip, "strength_clip")):
                    expected_patches = []
                    for entry in selected:
                        strength = getattr(entry, attr) * 0.8
                        if method == "average" and entry.merge:
                            strength /= marked
                        if strength != 0.0:
                            expected_patches.append((strength, entry.loaded_patches["weight"], 1.0, None, None))
                    base = model.model.weight.detach()
                    expected = loader.comfy.lora.calculate_weight(expected_patches, base.clone(), "weight")
                    actual = loader.comfy.lora.calculate_weight(output.patches["weight"], base.clone(), "weight")
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(len(output.patches["weight"]), len(expected_patches))
                self.assertEqual([entry.strength_model for entry in selected], [1.0, -0.3, 0.7, 0.5])

    def check_dense_merge(self, patches, strengths, base, key="weight"):
        offset = key[1] if isinstance(key, tuple) else None
        local_base = base.narrow(*offset) if offset is not None else base
        deltas = [apply_patch(patch, local_base, strength) - local_base if patch is not None else torch.zeros_like(local_base)
                  for patch, strength in zip(patches, strengths)]
        for patch, strength, delta in zip(patches, strengths, deltas):
            native_patches = [(strength, patch, 1.0, offset, None)] if patch is not None else []
            norm = loader._svc_patches_squared_norm(native_patches, base, "weight")
            torch.testing.assert_close(torch.tensor(norm), delta.square().sum(), atol=2e-4, rtol=2e-4)
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

    def test_post_merge_scaling_preserves_layer_and_direction_ratios(self):
        entries = []
        for a, b in (([1.0, 0.0], [0.1, 0.0]), ([1.0, 1.0], [0.0, 2.0])):
            patches = {"a.weight": ("diff", (torch.tensor(a).diag(),)),
                       "b.weight": ("diff", (torch.tensor(b).diag(),))}
            groups = {group: {} for group in loader._BLOCK_WEIGHT_KEYS}
            groups["others"] = patches
            entries.append(loader._LoadedLora("test", "test", {}, patches, groups, 1.0, 1.0, True, True))

        base = {key: torch.zeros(2, 2) for key in ("a.weight", "b.weight")}
        merged = loader._merge_svc_lora_patches_for_state_dict(entries, base, "strength_model")
        a = apply_patch(merged["a.weight"], base["a.weight"])
        b = apply_patch(merged["b.weight"], base["b.weight"])
        scale = (2.2525 / 6.01)**0.5
        torch.testing.assert_close(a, torch.eye(2) * scale)
        torch.testing.assert_close(b, torch.tensor([0.1, 2.0]).diag() * scale)
        torch.testing.assert_close((a.square().sum() + b.square().sum()).sqrt(), torch.tensor(2.2525**0.5))

    def test_post_merge_scaling_of_low_rank_patches_uses_weighted_input_magnitude(self):
        entries = []
        for index, strength in enumerate((0.3, -0.5, 0.0)):
            up = torch.zeros(8, 1)
            down = torch.zeros(1, 8)
            up[index, 0] = down[0, index] = 1.0
            patch = adapters.LoRAAdapter(set(), (up, down, 2.0, None, None, None))
            groups = {group: {} for group in loader._BLOCK_WEIGHT_KEYS}
            groups["others"]["weight"] = patch
            entries.append(loader._LoadedLora("test", "test", {"others": 0.5}, {"weight": patch},
                                              groups, strength, 0.0, True, True))

        merged = loader._merge_svc_lora_patches_for_state_dict(entries, {"weight": torch.zeros(8, 8)}, "strength_model")
        delta = apply_patch(merged["weight"], torch.zeros(8, 8))
        torch.testing.assert_close(delta.norm(), torch.tensor((0.3**2 + 0.5**2)**0.5 / 3))
        expected = torch.tensor([0.3, -0.5] + [0.0] * 6).diag() / 3
        torch.testing.assert_close(delta, expected)

    def test_post_merge_scaling_matches_average_for_unequal_aligned_loras(self):
        entries = []
        for strength in (0.01, 1.0):
            patches = {"weight": ("diff", (torch.eye(2),))}
            groups = {group: {} for group in loader._BLOCK_WEIGHT_KEYS}
            groups["others"] = patches
            entries.append(loader._LoadedLora("test", "test", {}, patches, groups, strength, 0.0, True, True))
        merged = loader._merge_svc_lora_patches_for_state_dict(entries, {"weight": torch.zeros(2, 2)}, "strength_model")
        torch.testing.assert_close(apply_patch(merged["weight"], torch.zeros(2, 2)), torch.eye(2) * 0.505)

    def test_svc_strength_matches_average_with_overlapping_fused_offsets(self):
        entries = []
        reference_patches = []
        base = torch.zeros(12, 6)
        for index, strength in enumerate((0.8, -0.3, 1.2)):
            offset = (0, 4, 4) if index == 2 else None
            key = ("weight", offset) if offset is not None else "weight"
            patch = self.lora(rank=index + 1, alpha=2.0, shape=(4 if offset else 12, 6))
            groups = {group: {} for group in loader._BLOCK_WEIGHT_KEYS}
            groups["others"][key] = patch
            entries.append(loader._LoadedLora("test", "test", {}, {key: patch}, groups, strength, 0.0, True, True))
            reference_patches.append((strength / 3, patch, 1.0, offset, None))
        reference = loader.comfy.lora.calculate_weight(reference_patches, base.clone(), "weight")
        norm = loader._svc_patches_squared_norm(reference_patches, base, "weight")
        torch.testing.assert_close(torch.tensor(norm), reference.square().sum())
        merged = loader._merge_svc_lora_patches_for_state_dict(entries, {"weight": base}, "strength_model")
        merged_patches = [(1.0, patch, 1.0, key[1] if isinstance(key, tuple) else None, None)
                          for key, patch in merged.items()]
        actual = loader.comfy.lora.calculate_weight(merged_patches, base.clone(), "weight")
        torch.testing.assert_close(actual.norm(), reference.norm())

    def test_svc_strength_matches_average_for_dora_and_mixed_adapters(self):
        first = self.lora(alpha=1.0)
        first.weights = (*first.weights[:4], torch.rand(8, 1) + 1.0, None)
        base = torch.randn(8, 6)
        for second in (self.lora(3), ("diff", (torch.randn(8, 6),))):
            entries = []
            reference_patches = []
            for patch, strength in ((first, 0.7), (second, -0.3)):
                groups = {group: {} for group in loader._BLOCK_WEIGHT_KEYS}
                groups["others"]["weight"] = patch
                entries.append(loader._LoadedLora("test", "test", {}, {"weight": patch}, groups, strength, 0.0, True, True))
                reference_patches.append((strength / 2, patch, 1.0, None, None))
            reference = loader.comfy.lora.calculate_weight(reference_patches, base.clone(), "weight") - base
            merged = loader._merge_svc_lora_patches_for_state_dict(entries, {"weight": base}, "strength_model")
            actual = apply_patch(merged["weight"], base) - base
            torch.testing.assert_close(actual.norm(), reference.norm())

    def test_zero_and_opposing_updates_remain_zero(self):
        for delta in (torch.zeros(3, 4), torch.ones(3, 4)):
            merged = svc.merge_svc_delta_tensors([delta, -delta])
            torch.testing.assert_close(merged, torch.zeros_like(delta))
            entries = []
            for value in (delta, -delta):
                patches = {"weight": ("diff", (value,))}
                groups = {group: {} for group in loader._BLOCK_WEIGHT_KEYS}
                groups["others"] = patches
                entries.append(loader._LoadedLora("test", "test", {}, patches, groups, 1.0, 0.0, True, True))
            merged = loader._merge_svc_lora_patches_for_state_dict(entries, {"weight": torch.zeros_like(delta)}, "strength_model")
            torch.testing.assert_close(apply_patch(merged["weight"], torch.zeros_like(delta)), torch.zeros_like(delta))

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
            expected = svc.merge_svc_delta_tensors(deltas)
            expected *= torch.stack(deltas).mean(0).norm() / expected.norm()
            torch.testing.assert_close(apply_patch(merged, base), expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
