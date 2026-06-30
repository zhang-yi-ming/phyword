import sys
import types
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


janus_module = types.ModuleType("janus")
janus_diffusion_module = types.ModuleType("janus.diffusion")


class DummyActionEmbedder(torch.nn.Module):
    pass


class DummyFinalLayer(torch.nn.Module):
    pass


janus_diffusion_module.ActionEmbedder = DummyActionEmbedder
janus_diffusion_module.FinalLayer = DummyFinalLayer
janus_module.diffusion = janus_diffusion_module
sys.modules.setdefault("janus", janus_module)
sys.modules.setdefault("janus.diffusion", janus_diffusion_module)

cosmos_janus_module = types.ModuleType("models.cosmos_janus")


class DummySlimLlamaMLP(torch.nn.Module):
    pass


cosmos_janus_module.SlimLlamaMLP = DummySlimLlamaMLP
sys.modules.setdefault("models.cosmos_janus", cosmos_janus_module)

from models.cosmos_janus_cot import MoTAttentionWrapper3  # noqa: E402


class ValueTokenMaskTest(unittest.TestCase):
    def _wrapper(self, video_to_value=False, nonvalue_to_value=False, action_self_causal=False):
        wrapper = MoTAttentionWrapper3.__new__(MoTAttentionWrapper3)
        wrapper.cosmos_self_only_bridge = False
        wrapper.decosmos = False
        wrapper.bridge_action_self_causal_override = bool(action_self_causal)
        wrapper.value_token_mask_nonvalue_to_value = bool(nonvalue_to_value)
        wrapper.value_token_mask_video_to_value = bool(video_to_value or nonvalue_to_value)
        return wrapper

    def test_full_bridge_value_slice_mask_modes(self):
        device = torch.device("cpu")
        value_tokens = 2 * 3
        S_v = 5 * value_tokens
        S_l = 2
        S_a = 3
        value_start = S_v - value_tokens
        value_end = S_v

        baseline = self._wrapper()._build_bridge_mask(S_v, S_l, S_a, device, value_token_count=0)[0, 0]
        disabled = self._wrapper()._build_bridge_mask(
            S_v, S_l, S_a, device, value_token_count=value_tokens
        )[0, 0]
        self.assertTrue(torch.equal(disabled, baseline))

        video_masked = self._wrapper(video_to_value=True)._build_bridge_mask(
            S_v, S_l, S_a, device, value_token_count=value_tokens
        )[0, 0]
        self.assertFalse(video_masked[:value_start, value_start:value_end].any())
        self.assertTrue(video_masked[value_start:value_end, :value_start].all())
        self.assertTrue(video_masked[value_start:value_end, value_start:value_end].all())
        self.assertTrue(video_masked[S_v:S_v + S_l, value_start:value_end].all())
        self.assertTrue(video_masked[S_v + S_l:, value_start:value_end].all())

        nonvalue_masked = self._wrapper(nonvalue_to_value=True)._build_bridge_mask(
            S_v, S_l, S_a, device, value_token_count=value_tokens
        )[0, 0]
        self.assertFalse(nonvalue_masked[:value_start, value_start:value_end].any())
        self.assertFalse(nonvalue_masked[S_v:S_v + S_l, value_start:value_end].any())
        self.assertFalse(nonvalue_masked[S_v + S_l:, value_start:value_end].any())
        self.assertTrue(nonvalue_masked[value_start:value_end, :value_start].all())
        self.assertTrue(nonvalue_masked[value_start:value_end, value_start:value_end].all())

    def test_cached_masks_only_apply_strong_mode(self):
        device = torch.device("cpu")
        value_tokens = 6
        S_v = 30
        S_l = 2
        S_a = 3
        value_start = S_v - value_tokens
        value_end = S_v

        video_only = self._wrapper(video_to_value=True)
        action_mask = video_only._build_action_only_mask(
            S_v, S_l, S_a, device, value_token_count=value_tokens
        )[0, 0]
        latent_mask = video_only._build_latent_only_mask(
            S_v, S_l, device, value_token_count=value_tokens
        )[0, 0]
        self.assertTrue(action_mask[:, value_start:value_end].all())
        self.assertTrue(latent_mask[:, value_start:value_end].all())

        strong = self._wrapper(nonvalue_to_value=True)
        action_mask = strong._build_action_only_mask(
            S_v, S_l, S_a, device, value_token_count=value_tokens
        )[0, 0]
        latent_mask = strong._build_latent_only_mask(
            S_v, S_l, device, value_token_count=value_tokens
        )[0, 0]
        self.assertFalse(action_mask[:, value_start:value_end].any())
        self.assertFalse(latent_mask[:, value_start:value_end].any())

    def test_full_bridge_masks_action_value_token_for_nonvalue_queries(self):
        device = torch.device("cpu")
        S_v = 4
        S_l = 2
        S_a = 4
        action_start = S_v + S_l
        value_col = S_v + S_l + S_a - 1

        for action_self_causal in (False, True):
            wrapper = self._wrapper(action_self_causal=action_self_causal)
            baseline = wrapper._build_bridge_mask(S_v, S_l, S_a, device)[0, 0]
            mask = wrapper._build_bridge_mask(
                S_v,
                S_l,
                S_a,
                device,
                action_value_token_count=1,
            )[0, 0]
            self.assertTrue(torch.equal(mask[:action_start, value_col], baseline[:action_start, value_col]))
            self.assertFalse(mask[action_start:value_col, value_col].any())
            self.assertTrue(mask[value_col, :value_col].all())
            self.assertTrue(mask[value_col, value_col])

    def test_action_only_masks_action_value_token_for_nonvalue_queries(self):
        device = torch.device("cpu")
        S_v = 4
        S_l = 2
        S_a = 4
        value_col = S_v + S_l + S_a - 1
        value_query_row = S_l + S_a - 1

        for action_self_causal in (False, True):
            wrapper = self._wrapper(action_self_causal=action_self_causal)
            baseline = wrapper._build_action_only_mask(S_v, S_l, S_a, device)[0, 0]
            mask = wrapper._build_action_only_mask(
                S_v,
                S_l,
                S_a,
                device,
                action_value_token_count=1,
            )[0, 0]
            self.assertTrue(torch.equal(mask[:S_l, value_col], baseline[:S_l, value_col]))
            self.assertFalse(mask[S_l:value_query_row, value_col].any())
            self.assertTrue(mask[value_query_row, :].all())


if __name__ == "__main__":
    unittest.main()
