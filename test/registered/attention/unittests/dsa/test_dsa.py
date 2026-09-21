import unittest
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
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    DSA_DECODE_IMPL_VARIANTS,
    DSA_PAGE_SIZE,
    DSA_PREFILL_IMPL_VARIANTS,
    DSAAttentionCase,
    make_dsa_dense_fallback_cases,
    make_dsa_sparse_cases,
    run_dsa_attention_case,
    run_dsa_sparse_attention_case,
    run_dsa_sparse_cuda_graph_decode_impl_variant_case,
    run_dsa_sparse_decode_impl_variant_case,
    run_dsa_sparse_fp8_decode_case,
    run_dsa_sparse_fp8_prefill_case,
    run_dsa_sparse_prefill_impl_variant_case,
    run_dsa_sparse_speculative_forward_mode_case,
    run_dsa_sparse_tilelang_decode_case,
    run_dsa_sparse_tilelang_prefill_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dsa_sparse_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_dsa_eagle_draft_cuda_graph_runner_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=34, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=34, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSAAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dsa_dense_fallback_cases("dsa")
    SPARSE_CASES = make_dsa_sparse_cases("dsa")
    # PCG/BCG split-op extend coverage is *not* added here — DSA's
    # MHA_ONE_SHOT dense fallback passes K as concatenated prefix+extend
    # (length = sum(seq_lens)) to `module.attn`, but
    # `unified_attention_with_output` (`radix_attention.py:170-208`) slices
    # K to `forward_batch.global_num_token_non_padded_cpu` (= live extend-token
    # count), under the per-token K convention used by Triton/FlashInfer/
    # FA. The K-slice removes the prefix portion, so DSA's dense fallback
    # output diverges by ~50% mismatch under piecewise CG. See
    # dsa/README.md "Production-Unsupported" for the path forward.

    def test_mha_one_shot_dense_fallback_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                # GB300 (SM10.x) kernel requires 128-dim query/value;
                # use head_dim=128 rather than the generic DEFAULT_HEAD_DIM=16.
                run_dsa_attention_case(self, case, head_dim=128)

    def test_sparse_topk_cases(self):
        for case in self.SPARSE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_sparse_attention_case(self, case)

    # Non-trailing index layouts. The reference gathers Q/K via
    # `fixture.topk_rows`, so any valid permutation of keys in
    # `[0, key_count)` produces a matching reference. These layouts
    # exercise the kernel's non-contiguous gather path (production
    # top-k by attention score is not naturally trailing for long
    # prefixes). Use long-prefix decode where `key_count > index_topk`
    # so the pattern actually subsamples (with key_count <= topk,
    # strided/head_tail collapse back to the trailing case).
    NON_TRAILING_INDEX_CASES = (
        (
            DSAAttentionCase(
                name="dsa_sparse_decode_strided_index_long_prefix",
                backend="dsa",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=1,
                page_size=DSA_PAGE_SIZE,
                prefix_lens=(2048,),
            ),
            "strided",
        ),
        (
            DSAAttentionCase(
                name="dsa_sparse_decode_head_tail_index_long_prefix",
                backend="dsa",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=1,
                page_size=DSA_PAGE_SIZE,
                prefix_lens=(2048,),
            ),
            "head_tail",
        ),
    )

    def test_sparse_non_trailing_index_cases(self):
        for case, pattern in self.NON_TRAILING_INDEX_CASES:
            with self.subTest(case=case.name, backend=case.backend, pattern=pattern):
                run_dsa_sparse_attention_case(self, case, index_pattern=pattern)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    # shuffled_pages is the default for all DSA tests via
    # build_dsa_attention_fixture / build_dsa_sparse_attention_fixture;
    # this method opts into the more aggressive interleaved_pages +
    # non_monotonic_extend layouts on representative dense fallback and
    # sparse top-k cases.
    LAYOUT_DENSE_CASES = (
        DSAAttentionCase(
            name="layout_dsa_dense_fallback_two_request",
            backend="dsa",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(0, 32),
            extend_lens=(32, 16),
        ),
    )
    LAYOUT_SPARSE_CASES = (
        DSAAttentionCase(
            name="layout_dsa_sparse_decode_long_prefix",
            backend="dsa",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(2048,),
        ),
    )

    def test_layout_robustness_dense_cases(self):
        for case in self.LAYOUT_DENSE_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                with self.subTest(case=case.name, layout=layout):
                    run_dsa_attention_case(self, case, head_dim=128, loc_layout=layout)

    def test_layout_robustness_sparse_cases(self):
        for case in self.LAYOUT_SPARSE_CASES:
            for layout in ("interleaved_pages",):
                with self.subTest(case=case.name, layout=layout):
                    run_dsa_sparse_attention_case(self, case, loc_layout=layout)

    # CG decode replay via the sparse `flashmla_kv` path (cached MLA latent
    # KV, written by `_populate_dsa_sparse_prefix_kv` at fixture build).
    # Unlike the MHA_ONE_SHOT dense fallback (where K is passed inline as
    # prefix+extend and `unified_attention_with_output` slicing breaks
    # piecewise CG), sparse decode reads cached K and is CG-compatible.
    CUDA_GRAPH_DECODE_CASES = (
        DSAAttentionCase(
            name="runner_cuda_graph_dsa_sparse_decode_flashmla_kv",
            backend="dsa",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(127, 128),
        ),
    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_sparse_cuda_graph_decode_case(self, case)

    # DSA implementation-variant matrix. DSA exposes multiple kernel
    # implementations (`flashmla_sparse`, `flashmla_kv`, `fa3`, `tilelang`,
    # `trtllm`, `aiter`) selected by `--dsa-prefill-backend` /
    # `--dsa-decode-backend`. Each variant maps to a distinct kernel path
    # in `dsa_backend.py`; `dsa_impl_capability` gates per hardware/SDK so
    # impls unavailable on the test box (e.g., `trtllm` requires SM100+,
    # `aiter` requires HIP) emit a clean `skipTest` with a reason rather
    # than spuriously failing.
    PREFILL_IMPL_CASE = DSAAttentionCase(
        name="dsa_sparse_prefill_impl_variant",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        # Long prefix keeps the backend on the MLA path (above the
        # MHA_ONE_SHOT short-sequence threshold) so the impl override
        # actually routes through `dsa_prefill_impl`.
        prefix_lens=(2048,),
        extend_lens=(1,),
    )
    DECODE_IMPL_CASE = DSAAttentionCase(
        name="dsa_sparse_decode_impl_variant",
        backend="dsa",
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(128,),
    )

    def test_sparse_prefill_impl_variants(self):
        for impl in DSA_PREFILL_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                run_dsa_sparse_prefill_impl_variant_case(
                    self, self.PREFILL_IMPL_CASE, impl
                )

    def test_sparse_decode_impl_variants(self):
        for impl in DSA_DECODE_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                run_dsa_sparse_decode_impl_variant_case(
                    self, self.DECODE_IMPL_CASE, impl
                )

    # Speculative forward-mode coverage. TARGET_VERIFY and
    # DRAFT_EXTEND_V2 both route through the `dsa_decode_impl`
    # dispatcher (the same kernel selection as plain DECODE) but
    # produce different `seqlens_expanded` and `cu_seqlens_q` from
    # `dsa_backend.py:469-529`. `DSAMockModelRunner.__init__` derives
    # `speculative_num_draft_tokens` from `case.extend_lens` so deep_gemm
    # JIT-compiles with a non-zero aligned batch size.
    SPECULATIVE_FORWARD_MODE_CASES = (
        DSAAttentionCase(
            name="dsa_sparse_target_verify",
            backend="dsa",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(128,),
            extend_lens=(3,),
        ),
        DSAAttentionCase(
            name="dsa_sparse_draft_extend_v2",
            backend="dsa",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(128,),
            extend_lens=(3,),
        ),
    )

    def test_sparse_speculative_forward_mode_cases(self):
        for case in self.SPECULATIVE_FORWARD_MODE_CASES:
            with self.subTest(case=case.name, mode=case.forward_mode.name):
                run_dsa_sparse_speculative_forward_mode_case(self, case)

    # FP8 KV cache (`dsa_kv_cache_store_fp8=True`) — the production
    # deployment dtype. Switches `DSATokenToKVPool` to packed
    # FP8-nope/BF16-rope storage at 656 bytes/token; `set_mla_kv_buffer`
    # routes through `quantize_k_cache_separate` and the kernel reads
    # FP8 directly. The reference stays on BF16 K (independent of the
    # cache bytes), and `DSA_SPARSE_FP8_ATOL=0.2` absorbs FP8 quant
    # noise — same separation principle as the DSV4 SWA fixture so a
    # silent pack/write bug cannot corrupt both paths identically.
    #
    # FP8 + `flashmla_sparse` prefill + EXTEND + non-empty prefix is the
    # only combo that hits `TopkTransformMethod.RAGGED`
    # (`get_topk_transform_method`), which exercises
    # `dequantize_k_cache_paged` and the `topk_indices_offset` shift —
    # paths that the BF16 default suite never reaches.
    FP8_PREFILL_RAGGED_CASE = DSAAttentionCase(
        name="dsa_sparse_fp8_prefill_ragged_topk",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        # Long prefix → above MHA threshold, RAGGED topk transform
        prefix_lens=(2048,),
        extend_lens=(1,),
    )
    FP8_PREFILL_PAGED_CASE = DSAAttentionCase(
        name="dsa_sparse_fp8_prefill_paged_topk",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(2048,),
        extend_lens=(1,),
    )
    FP8_DECODE_CASE = DSAAttentionCase(
        name="dsa_sparse_fp8_decode",
        backend="dsa",
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(128,),
    )

    def test_sparse_fp8_prefill_cases(self):
        for impl in DSA_PREFILL_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                # Each impl that isn't in `DSA_FP8_COMPATIBLE_PREFILL_IMPLS`
                # emits skipTest from the helper with the reason. The
                # `flashmla_sparse` impl hits the RAGGED-topk path; the
                # others stay on PAGED.
                case = (
                    self.FP8_PREFILL_RAGGED_CASE
                    if impl == "flashmla_sparse"
                    else self.FP8_PREFILL_PAGED_CASE
                )
                run_dsa_sparse_fp8_prefill_case(self, case, dsa_prefill_backend=impl)

    def test_sparse_fp8_decode_cases(self):
        for impl in DSA_DECODE_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                run_dsa_sparse_fp8_decode_case(
                    self, self.FP8_DECODE_CASE, dsa_decode_backend=impl
                )

    # Tilelang sparse cases — dedicated topk=2048 fixture.
    # `tilelang_sparse_fwd` asserts `topk == 2048` at
    # `dsa/tilelang_kernel.py:1345`, so this fixture variant carries a
    # 2048-wide trailing-topk row builder. Prefix length must be >= 2048
    # to produce a real (non-padded) topk row.
    TILELANG_PREFILL_CASE = DSAAttentionCase(
        name="dsa_sparse_tilelang_prefill",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(4096,),
        extend_lens=(1,),
    )
    TILELANG_DECODE_CASE = DSAAttentionCase(
        name="dsa_sparse_tilelang_decode",
        backend="dsa",
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(4096,),
    )

    def test_sparse_tilelang_prefill_case(self):
        run_dsa_sparse_tilelang_prefill_case(self, self.TILELANG_PREFILL_CASE)

    def test_sparse_tilelang_decode_case(self):
        run_dsa_sparse_tilelang_decode_case(self, self.TILELANG_DECODE_CASE)

    # EAGLE production draft CUDA-graph runner integration. Wires DSA
    # through `speculative_draft_runner.py`'s shared
    # `EagleDraftCudaGraphRunnerAdapter` (same lifecycle as DSV4 /
    # dense / MLA). DSA's chain-only constraint comes from the
    # synthesized topk_indices path — tree draft needs parent-indices
    # plumbing through that synthesis; deferred.
    EAGLE_DRAFT_CASES = (
        DSAAttentionCase(
            name="runner_eagle_draft_decode_cuda_graph_dsa_chain",
            backend="dsa",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(128, 192),
        ),
    )

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        for case in self.EAGLE_DRAFT_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_eagle_draft_cuda_graph_runner_case(self, case)

    # CG decode replay with FP8 KV cache. Captures and replays through
    # `flashmla_kv` (the only FP8-compatible decode kernel). The
    # `_clone_dsa_sparse_cache` hook is reused as-is — it snapshots the
    # raw uint8 K buffer bytes, which round-trip correctly across
    # capture/replay regardless of bf16 vs FP8 packing.
    def test_sparse_fp8_cuda_graph_decode_case(self):
        from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
            run_dsa_sparse_cuda_graph_decode_case,
        )

        run_dsa_sparse_cuda_graph_decode_case(
            self,
            self.FP8_DECODE_CASE,
            dsa_decode_backend="flashmla_kv",
            fp8_kv_cache=True,
        )

    # CG decode replay parametrized over `dsa_decode_backend` impl. The
    # `flashmla_kv` baseline is already covered by
    # `test_runner_mode_cuda_graph_decode_cases`; this method extends the
    # CG matrix to every supported decode impl (`flashmla_sparse` /
    # `flashmla_kv` / `fa3` on H200, with `tilelang` / `trtllm` / `aiter`
    # skip-gated). Each impl re-builds the fixture with the impl forced
    # so the captured graph uses that specific kernel.
    def test_sparse_cuda_graph_decode_impl_variants(self):
        for impl in DSA_DECODE_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                run_dsa_sparse_cuda_graph_decode_impl_variant_case(
                    self, self.CUDA_GRAPH_DECODE_CASES[0], impl
                )


def _make_padding_batch(bs=2, tokens=2, device="cpu", mode=ForwardMode.DECODE):
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


def _pad_planned_batch(fb, bs, tokens):
    fb.mark_forward_metadata_ready()
    fb.batch_size = bs
    fb.input_ids = fb.input_ids.new_zeros(tokens)
    fb.positions = fb.positions.new_zeros(tokens)
    fb.out_cache_loc = torch.nn.functional.pad(
        fb.out_cache_loc, (0, tokens - fb.out_cache_loc.numel())
    )


def _make_padding_backend(fb, *, flashmla=False):
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
                fb = _make_padding_batch(mode=mode)
                _pad_planned_batch(fb, bs, tokens)
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

    def test_dsa_preserves_indexer_plan_and_restores_after_exception(self):
        for mode, real_tokens, final_bs, final_tokens in (
            (ForwardMode.DECODE, 2, 4, 4),
            (ForwardMode.DECODE, 2, 4, 2),
            (ForwardMode.DRAFT_EXTEND_V2, 6, 2, 8),
            (ForwardMode.DRAFT_EXTEND_V2, 6, 4, 8),
        ):
            with self.subTest(mode=mode, final_bs=final_bs, final_tokens=final_tokens):
                fb = _make_padding_batch(tokens=real_tokens, mode=mode)
                backend = _make_padding_backend(fb)
                original = backend.forward_metadata
                _pad_planned_batch(fb, final_bs, final_tokens)
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
                fb = _make_padding_batch()
                backend = _make_padding_backend(fb)
                backend.forward_metadata = replace(
                    backend.forward_metadata,
                    **{field: getattr(backend.forward_metadata, field)[:1]},
                )
                _pad_planned_batch(fb, 4, 4)
                with self.assertRaisesRegex(RuntimeError, "real attention extent"):
                    with backend.use_forward_metadata_after_padding(fb):
                        self.fail("incomplete real metadata must not launch")

    def test_flashmla_schedule_is_rebuilt_only_for_attention(self):
        fb = _make_padding_batch()
        backend = _make_padding_backend(fb)
        backend.dsa_decode_impl = "flashmla_kv"
        old = DSAFlashMLAMetadata(torch.zeros(1), torch.zeros(3, dtype=torch.int32))
        backend.forward_metadata = replace(
            backend.forward_metadata, flashmla_metadata=old
        )
        rebuilt = DSAFlashMLAMetadata(torch.ones(1), torch.zeros(5, dtype=torch.int32))
        _pad_planned_batch(fb, 4, 4)
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
                fb = _make_padding_batch()
                backend = _make_padding_backend(fb)
                _pad_planned_batch(fb, 4, 4)
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
        fb = _make_padding_batch()
        backend = _make_padding_backend(fb)
        # An oversized allocation is not evidence of additional real requests.
        backend.forward_metadata = replace(
            backend.forward_metadata,
            dsa_cache_seqlens_int32=torch.tensor([16, 16, 99, 99], dtype=torch.int32),
        )
        _pad_planned_batch(fb, 4, 4)
        with backend.use_forward_metadata_after_padding(fb):
            self.assertEqual(
                backend._get_attention_metadata().dsa_cache_seqlens_int32.tolist(),
                [16, 16, 0, 0],
            )

    def test_missing_plan_and_short_flashmla_offsets_fail_early(self):
        for missing in (True, False):
            fb = _make_padding_batch()
            backend = _make_padding_backend(fb)
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
            _pad_planned_batch(fb, 4, 4)
            with self.assertRaisesRegex(RuntimeError, "missing|real attention extent"):
                with backend.use_forward_metadata_after_padding(fb):
                    self.fail("incomplete metadata must fail before the body")

    def test_trtllm_existing_decode_replan_remains_scoped(self):
        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        fb = _make_padding_batch()
        _pad_planned_batch(fb, 4, 4)
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
        fb = _make_padding_batch(device="cuda", tokens=real_tokens, mode=mode)
        backend = _make_padding_backend(fb, flashmla=True)
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
