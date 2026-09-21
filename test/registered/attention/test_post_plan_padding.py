"""Post-plan eager padding must preserve the semantic plan and real outputs."""

import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAFlashMLAMetadata,
    DSAMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")


def make_batch(bs=2, tokens=2, device="cpu", mode=ForwardMode.DECODE):
    return ForwardBatch(
        forward_mode=mode,
        batch_size=bs,
        input_ids=torch.zeros(tokens, dtype=torch.int64, device=device),
        positions=torch.zeros(tokens, dtype=torch.int64, device=device),
        req_pool_indices=torch.arange(bs, device=device),
        seq_lens=torch.full((bs,), 16, dtype=torch.int32, device=device),
        out_cache_loc=torch.arange(64, 64 + tokens, device=device),
        seq_lens_sum=16 * bs,
    )


def pad_batch(fb, bs, tokens):
    fb.mark_forward_metadata_ready()
    fb.batch_size = bs
    fb.input_ids = fb.input_ids.new_zeros(tokens)
    fb.positions = fb.positions.new_zeros(tokens)
    fb.out_cache_loc = torch.nn.functional.pad(
        fb.out_cache_loc, (0, tokens - fb.out_cache_loc.numel())
    )


def make_backend(fb, *, flashmla=False):
    """Minimal backend state; production metadata adaptation/forward stay real."""
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.device = fb.input_ids.device
    backend.dsa_decode_impl = "flashmla_kv" if flashmla else "triton"
    backend.dsa_index_kpool = 1
    backend.dsa_index_topk = 2048
    backend.hisparse_coordinator = None
    backend.use_fused_topk = True
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend._arange_buf = torch.arange(128, dtype=torch.int32, device=backend.device)
    backend.flashmla_kv_num_q_heads = 64
    backend.real_page_size = 64
    backend.kv_cache_dim = 576
    backend.dsa_kv_cache_store_fp8 = False
    backend.use_mha = False
    tokens = fb.input_ids.numel()
    lengths = torch.full((tokens,), 16, dtype=torch.int32, device=backend.device)
    backend.forward_metadata = DSAMetadata(
        page_size=64,
        cache_seqlens_int32=fb.seq_lens,
        max_seq_len_q=tokens // fb.batch_size,
        max_seq_len_k=16,
        cu_seqlens_q=torch.arange(
            0,
            tokens + 1,
            tokens // fb.batch_size,
            dtype=torch.int32,
            device=backend.device,
        ),
        cu_seqlens_k=compute_cu_seqlens(fb.seq_lens),
        page_table_1=torch.arange(64, 80, dtype=torch.int32, device=backend.device)
        .expand(tokens, -1)
        .clone(),
        real_page_table=torch.ones(
            (fb.batch_size, 1), dtype=torch.int32, device=backend.device
        ),
        dsa_cache_seqlens_int32=lengths,
        dsa_cu_seqlens_q=backend.get_device_int32_arange(tokens + 1),
        dsa_cu_seqlens_k=compute_cu_seqlens(lengths),
        dsa_extend_seq_lens_list=[tokens // fb.batch_size] * fb.batch_size,
        dsa_seqlens_expanded=lengths.clone(),
        # Sentinels stand for the true-B indexer schedule and its context lengths.
        paged_mqa_schedule_metadata=torch.tensor([123], device=backend.device),
        paged_mqa_ctx_lens_2d=fb.seq_lens.view(-1, 1),
        flashmla_metadata=(
            backend._compute_flashmla_metadata(lengths, 1) if flashmla else None
        ),
    )
    return backend


class TestPostPlanPadding(CustomTestCase):
    def test_unknown_backend_rejects_drift_before_model(self):
        for mode, bs, tokens in (
            (ForwardMode.DECODE, 4, 4),
            (ForwardMode.DRAFT_EXTEND_V2, 2, 8),
        ):
            with self.subTest(mode=mode):
                fb = make_batch(mode=mode)
                pad_batch(fb, bs, tokens)
                model = SimpleNamespace(forward=Mock())
                runner = EagerRunner.__new__(EagerRunner)
                runner.enable_pdmux = False
                backend = AttentionBackend()
                runner.model_runner = SimpleNamespace(
                    model=model,
                    attn_backend=backend,
                    device_timer=None,
                    _pp_kwargs=lambda _: {},
                    _extend_forward_kwargs=lambda *_: {},
                    prefill_cuda_graph_runner=None,
                )
                runner.load_batch = lambda batch, _: batch
                with forward_context(ForwardContext(attn_backend=backend)):
                    with self.assertRaisesRegex(
                        RuntimeError, "planned .*tokens=2.*final"
                    ):
                        runner.execute(fb)
                model.forward.assert_not_called()

    def test_matched_plan_and_empty_batch_need_no_adapter(self):
        for bs in (0, 2):
            fb = make_batch(bs=bs, tokens=bs)
            fb.mark_forward_metadata_ready()
            with AttentionBackend().use_forward_metadata_after_padding(fb):
                pass

    def test_wrapper_checks_all_children_and_restores_on_failure(self):
        calls = []

        class Adapter(AttentionBackend):
            @contextmanager
            def _use_padded_forward_metadata(self, fb):
                calls.append("enter")
                try:
                    yield
                finally:
                    calls.append("exit")

        wrapper = AttentionBackend()
        wrapper.attn_backend_list = [Adapter(), AttentionBackend()]
        fb = make_batch()
        pad_batch(fb, 4, 4)
        with self.assertRaises(RuntimeError):
            with wrapper.use_forward_metadata_after_padding(fb):
                self.fail("an unsupported child must fail before the body")
        self.assertEqual(calls, ["enter", "exit"])

    def test_dsa_preserves_indexer_plan_and_restores_after_exception(self):
        for mode, real_tokens, final_bs, final_tokens in (
            (ForwardMode.DECODE, 2, 4, 4),
            (ForwardMode.DECODE, 2, 4, 2),
            (ForwardMode.DRAFT_EXTEND_V2, 6, 2, 8),
            (ForwardMode.DRAFT_EXTEND_V2, 6, 4, 8),
        ):
            with self.subTest(mode=mode, final_bs=final_bs, final_tokens=final_tokens):
                fb = make_batch(tokens=real_tokens, mode=mode)
                backend = make_backend(fb)
                original = backend.forward_metadata
                pad_batch(fb, final_bs, final_tokens)
                with self.assertRaisesRegex(ValueError, "body failed"):
                    with backend.use_forward_metadata_after_padding(fb):
                        view = backend._get_attention_metadata()
                        indexer = backend.get_indexer_metadata(0, fb)
                        self.assertIs(indexer.attn_metadata, original)
                        self.assertIs(
                            indexer.paged_mqa_schedule_metadata,
                            original.paged_mqa_schedule_metadata,
                        )
                        self.assertEqual(indexer.paged_mqa_ctx_lens_2d.shape, (2, 1))
                        self.assertEqual(view.page_table_1.shape[0], final_tokens)
                        self.assertEqual(
                            view.dsa_cu_seqlens_q.numel(), final_tokens + 1
                        )
                        self.assertEqual(
                            view.dsa_cache_seqlens_int32.tolist(),
                            [16] * real_tokens + [0] * (final_tokens - real_tokens),
                        )
                        self.assertEqual(
                            view.dsa_cu_seqlens_k[-1].item(), 16 * real_tokens
                        )
                        raise ValueError("body failed")
                self.assertIs(backend._get_attention_metadata(), original)
                self.assertEqual(original.dsa_cache_seqlens_int32.numel(), real_tokens)

    def test_missing_real_rows_are_not_hidden_by_padding(self):
        for field in (
            "dsa_seqlens_expanded",
            "dsa_cache_seqlens_int32",
            "page_table_1",
        ):
            with self.subTest(field=field):
                fb = make_batch()
                backend = make_backend(fb)
                backend.forward_metadata = replace(
                    backend.forward_metadata,
                    **{field: getattr(backend.forward_metadata, field)[:1]},
                )
                pad_batch(fb, 4, 4)
                with self.assertRaisesRegex(RuntimeError, "real attention extent"):
                    with backend.use_forward_metadata_after_padding(fb):
                        self.fail("incomplete real metadata must not launch")

    def test_flashmla_schedule_is_rebuilt_only_for_attention(self):
        fb = make_batch()
        backend = make_backend(fb)
        backend.dsa_decode_impl = "flashmla_kv"
        old = DSAFlashMLAMetadata(torch.zeros(1), torch.zeros(3, dtype=torch.int32))
        backend.forward_metadata = replace(
            backend.forward_metadata, flashmla_metadata=old
        )
        rebuilt = DSAFlashMLAMetadata(torch.ones(1), torch.zeros(5, dtype=torch.int32))
        pad_batch(fb, 4, 4)
        with patch.object(
            backend, "_compute_flashmla_metadata", return_value=rebuilt
        ) as build:
            with backend.use_forward_metadata_after_padding(fb):
                self.assertIs(
                    backend._get_attention_metadata().flashmla_metadata, rebuilt
                )
                self.assertIs(backend.forward_metadata.flashmla_metadata, old)
            build.assert_called_once()
            self.assertEqual(build.call_args.args[0].tolist(), [16, 16, 0, 0])

    def test_eager_uses_active_step_backend_and_equivalent_replan_bypasses_adapter(
        self,
    ):
        for equivalent in (False, True):
            with self.subTest(equivalent=equivalent):
                fb = make_batch()
                backend = make_backend(fb)
                pad_batch(fb, 4, 4)
                fb.forward_metadata_replan_equivalent = equivalent
                outer = backend if equivalent else AttentionBackend()
                outer.init_forward_metadata = Mock()

                def body(*args, **kwargs):
                    expected = 2 if equivalent else 4
                    self.assertEqual(
                        backend._get_attention_metadata().dsa_cache_seqlens_int32.numel(),
                        expected,
                    )

                runner = EagerRunner.__new__(EagerRunner)
                runner.enable_pdmux = False
                runner.load_batch = lambda batch, _: batch
                runner.model_runner = SimpleNamespace(
                    model=SimpleNamespace(forward=body),
                    attn_backend=outer,
                    device_timer=None,
                    _pp_kwargs=lambda _: {},
                )
                with forward_context(ForwardContext(attn_backend=backend)):
                    runner.execute(fb)
                self.assertEqual(
                    outer.init_forward_metadata.call_count, int(equivalent)
                )
                self.assertEqual(
                    backend._get_attention_metadata().dsa_cache_seqlens_int32.numel(), 2
                )

    def test_existing_padding_capacity_does_not_define_real_rows(self):
        fb = make_batch()
        backend = make_backend(fb)
        # An oversized allocation is not evidence of additional real requests.
        backend.forward_metadata = replace(
            backend.forward_metadata,
            dsa_cache_seqlens_int32=torch.tensor([16, 16, 99, 99], dtype=torch.int32),
        )
        pad_batch(fb, 4, 4)
        with backend.use_forward_metadata_after_padding(fb):
            self.assertEqual(
                backend._get_attention_metadata().dsa_cache_seqlens_int32.tolist(),
                [16, 16, 0, 0],
            )

    def test_missing_plan_and_short_flashmla_offsets_fail_early(self):
        for missing in (True, False):
            fb = make_batch()
            backend = make_backend(fb)
            backend.dsa_decode_impl = "flashmla_kv"
            backend.forward_metadata = (
                None
                if missing
                else replace(
                    backend.forward_metadata,
                    flashmla_metadata=DSAFlashMLAMetadata(
                        torch.zeros(1), torch.zeros(2)
                    ),
                )
            )
            pad_batch(fb, 4, 4)
            with self.assertRaisesRegex(RuntimeError, "missing|real attention extent"):
                with backend.use_forward_metadata_after_padding(fb):
                    self.fail("incomplete metadata must fail before the body")

    def test_trtllm_existing_decode_replan_remains_scoped(self):
        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        fb = make_batch()
        pad_batch(fb, 4, 4)
        backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
        backend._kv_shard_pool = None
        original = SimpleNamespace(block_kv_indices=torch.zeros(2, 1), seq_lens_k=None)
        backend.forward_decode_metadata = original
        backend._decode_kernel_loc = None

        def init(batch):
            backend.forward_decode_metadata = SimpleNamespace(
                block_kv_indices=torch.zeros(4, 1)
            )
            batch.decode_trtllm_mla_metadata = backend.forward_decode_metadata

        backend.init_forward_metadata = Mock(side_effect=init)
        with self.assertRaisesRegex(ValueError, "body failed"):
            with backend.use_forward_metadata_after_padding(fb):
                self.assertEqual(
                    fb.decode_trtllm_mla_metadata.block_kv_indices.shape[0], 4
                )
                raise ValueError("body failed")
        backend.init_forward_metadata.assert_called_once_with(fb)
        self.assertIs(backend.forward_decode_metadata, original)
        self.assertIsNone(fb.decode_trtllm_mla_metadata)

    def test_shrinking_after_plan_is_rejected(self):
        fb = make_batch()
        fb.mark_forward_metadata_ready()
        fb.input_ids = fb.input_ids[:1]
        with self.assertRaisesRegex(RuntimeError, "Invalid attention pre-plan"):
            with AttentionBackend().use_forward_metadata_after_padding(fb):
                self.fail("shrinking cannot be treated as padding")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDSAPaddingKernel(CustomTestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_parallel

        self.enterContext(get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0))

    def test_real_outputs_and_kv_writes_match_unpadded_forward(self):
        for mode, real_tokens, final_bs, num_tokens in (
            (ForwardMode.DECODE, 2, 4, 4),
            (ForwardMode.DRAFT_EXTEND_V2, 6, 2, 8),
        ):
            for fused in (False, True):
                with self.subTest(mode=mode, fused=fused):
                    self._check_kernel(mode, real_tokens, final_bs, num_tokens, fused)

    def _check_kernel(self, mode, real_tokens, final_bs, num_tokens, fused):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        torch.manual_seed(42)
        fb = make_batch(device="cuda", tokens=real_tokens, mode=mode)
        backend = make_backend(fb, flashmla=True)
        backend.use_fused_topk = fused
        pool = MLATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            use_dsa=True,
        )
        backend.token_to_kv_pool = pool
        cache = pool.get_key_buffer(0)
        cache.normal_()
        initial = cache.clone()
        layer = SimpleNamespace(
            is_cross_attention=False,
            tp_q_head_num=64,
            v_head_dim=512,
            head_dim=576,
            scaling=576**-0.5,
            layer_id=0,
        )
        q = torch.randn((num_tokens, 64, 576), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((num_tokens, 1, 512), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((num_tokens, 1, 64), device="cuda", dtype=torch.bfloat16)
        indices = torch.full((real_tokens, 2048), -1, device="cuda", dtype=torch.int32)
        indices[:, :16] = torch.arange(16, device="cuda", dtype=torch.int32) + (
            64 if fused else 0
        )
        forward = backend.forward_decode if mode.is_decode() else backend.forward_extend
        reference = forward(
            q[:real_tokens],
            k[:real_tokens],
            k[:real_tokens],
            layer,
            fb,
            k_rope=rope[:real_tokens],
            topk_indices=indices,
        )
        reference_cache = cache.clone()
        cache.copy_(initial)
        original = backend.forward_metadata
        fb.mark_forward_metadata_ready()
        fb.batch_size = final_bs
        # Use the actual producer of token/cache-location padding.
        fb._pad_inputs_to_size(
            SimpleNamespace(attn_backend=backend), num_tokens, final_bs
        )
        # The old call reaches FlashMLA with only R+1 split offsets for M queries.
        error = AssertionError if mode.is_decode() and not fused else RuntimeError
        message = "" if error is AssertionError else "num_splits must have shape"
        with self.assertRaisesRegex(error, message) as failure:
            forward(q, k, k, layer, fb, k_rope=rope, topk_indices=indices)
        print(
            f"old path ({mode.name}, fused={fused}): {error.__name__}: {failure.exception}"
        )
        cache.copy_(initial)
        with backend.use_forward_metadata_after_padding(fb):
            output = forward(q, k, k, layer, fb, k_rope=rope, topk_indices=indices)
            self.assertEqual(
                backend._get_attention_metadata().flashmla_metadata.num_splits.numel(),
                num_tokens + 1,
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(output[:real_tokens], reference, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output.shape[0], num_tokens)
        # Padding uses reserved slot 0; real KV slots must match exactly.
        torch.testing.assert_close(cache[1:], reference_cache[1:], rtol=0, atol=0)
        self.assertIs(backend.forward_metadata, original)


if __name__ == "__main__":
    unittest.main()
