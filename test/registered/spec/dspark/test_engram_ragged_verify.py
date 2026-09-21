"""Engram and low-ratio request mapping for compact DSpark verification.

Exercise the CPU fallback, CUDA hash kernel, and replay with changing request
boundaries. The oracle builds each request's n-grams independently in Python.
"""

import unittest
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices
from sglang.srt.layers.engram import EngramHasher, EngramLayout
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Imports require the GPU wheels even when testing the CPU fallback.
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class _Tokenizer:
    def __init__(self):
        self.backend_tokenizer = Tokenizer(
            WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0")
        )

    def __len__(self):
        return 64


def _hasher(device):
    layout = EngramLayout(
        max_ngram_size=4,
        layer_ids=(1, 14),
        num_embeddings=(41, 71),
        primes=(((11,), (13,), (17,)), ((19,), (23,), (29,))),
        n_heads=1,
        head_dim=8,
    )
    hasher = EngramHasher(layout, _Tokenizer(), 0, 64).to(device)
    hasher.init_history(8, device)
    hasher.history[:8] = torch.arange(1, 25, device=device).reshape(8, 3)
    return hasher


def _batch(lens, width, device, *, bucket=None, slots=None, ragged=True, cap=False):
    bucket = sum(lens) if bucket is None else bucket
    slots = [6, 1, 4][: len(lens)] if slots is None else slots
    layout = None
    if ragged:
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor(lens, dtype=torch.int32, device=device),
            graph_num_tokens=bucket,
        )
        if cap:
            layout = layout.padded_to_bucket(padded_bs=len(slots), cap=width)
    sequences = [[30 + 7 * r + i for i in range(n)] for r, n in enumerate(lens)]
    prefixes = [0, 1, 9][: len(lens)]
    ids = torch.zeros(bucket, dtype=torch.int64, device=device)
    positions = torch.zeros_like(ids)
    total = sum(lens)
    ids[:total] = torch.tensor([t for seq in sequences for t in seq], device=device)
    positions[:total] = torch.tensor(
        [p + i for p, n in zip(prefixes, lens) for i in range(n)], device=device
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=torch.tensor(slots, dtype=torch.int64, device=device),
        positions=positions,
        spec_info=SimpleNamespace(draft_token_num=width, ragged_verify_layout=layout),
        out_cache_loc=None,
        engram_history=None,
    )
    return ids, batch, sequences, prefixes


def _reference(hasher, sequences, prefixes, slots):
    history = hasher.history.cpu().tolist()
    token_map = hasher.token_map.cpu().tolist()
    multipliers = hasher.multipliers.cpu().tolist()
    primes = hasher.primes.cpu().tolist()
    offsets = hasher.offsets.cpu().tolist()
    result = []
    for sequence, prefix, slot in zip(sequences, prefixes, slots):
        context = history[slot] + sequence
        for t in range(len(sequence)):
            compressed = [
                hasher.pad_id
                if prefix + t < s
                else token_map[context[hasher.max_ngram_size - 1 + t - s]]
                for s in range(hasher.max_ngram_size)
            ]
            layers = []
            for layer, mult in enumerate(multipliers):
                rolling = compressed[0] * mult[0]
                hashes = []
                for s in range(1, hasher.max_ngram_size):
                    rolling ^= compressed[s] * mult[s]
                    heads = primes[layer][s - 1]
                    for h, p in enumerate(heads):
                        hashes.append(
                            rolling % p + offsets[layer][(s - 1) * len(heads) + h]
                        )
                layers.append(hashes)
            result.append(layers)
    return torch.tensor(result, dtype=torch.int64, device=hasher.history.device)


class TestEngramRaggedVerify(CustomTestCase):
    def _check(self, hasher, ids, batch, sequences, prefixes, output=None):
        before = hasher.history.clone()
        slots = batch.req_pool_indices.cpu().tolist()
        expected = _reference(hasher, sequences, prefixes, slots)
        if output is None:
            output = (
                hasher(ids, batch),
                token_req_indices(batch, num_tokens=ids.numel()),
            )
        hashes, request_ids = output
        torch.testing.assert_close(hashes[: len(expected)], expected, rtol=0, atol=0)
        expected_req = torch.tensor(
            [slot for slot, seq in zip(slots, sequences) for _ in seq],
            device=ids.device,
        )
        torch.testing.assert_close(request_ids[: len(expected_req)], expected_req)
        self.assertEqual(hashes.shape, (ids.numel(), 2, 3))
        self.assertEqual(request_ids.shape, ids.shape)
        torch.testing.assert_close(hasher.history, before, rtol=0, atol=0)
        torch.testing.assert_close(token_req_indices(batch), request_ids)

    def test_uniform_and_ragged_hashes(self):
        for device in ("cpu", "cuda"):
            hasher = _hasher(device)
            for width, lens, ragged in (
                (4, [4, 4, 4], False),
                (6, [6, 6, 6], False),
                (6, [6, 6, 6], True),
                (4, [4, 3, 1], True),
                (6, [4, 2, 4], True),
                (6, [6, 4, 2], True),
                (6, [2, 6, 1], True),
                (6, [1, 1, 1], True),
            ):
                with self.subTest(device=device, lens=lens, ragged=ragged):
                    self._check(hasher, *_batch(lens, width, device, ragged=ragged))

    def test_bucket_padding(self):
        for device in ("cpu", "cuda"):
            hasher = _hasher(device)
            for lens, bucket, slots, cap in (
                ([4, 2, 4], 12, [6, 1, 4], False),
                ([6, 4, 2], 24, [6, 1, 4, 0], False),
                ([6, 6], 12, [6, 1, 0, 0, 0], False),
                ([1, 1, 1], 24, [6, 1, 4, 0], True),
                ([4, 2, 4], 18, [6, 1, 4], True),
            ):
                with self.subTest(device=device, lens=lens, bucket=bucket, cap=cap):
                    self._check(
                        hasher,
                        *_batch(lens, 6, device, bucket=bucket, slots=slots, cap=cap),
                    )

    def test_only_accepted_tokens_commit_history(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                hasher = _hasher(device)
                ids, batch, sequences, prefixes = _batch([4, 2, 4], 6, device)
                self._check(hasher, ids, batch, sequences, prefixes)
                expected = hasher.history.clone()
                commits = [1, 2, 4]
                verify_ids = torch.full((3, 6), 63, dtype=torch.int64, device=device)
                for r, (slot, seq, accepted) in enumerate(
                    zip([6, 1, 4], sequences, commits)
                ):
                    verify_ids[r, : len(seq)] = torch.tensor(seq, device=device)
                    window = expected[slot].tolist() + seq[:accepted]
                    expected[slot] = torch.tensor(window[-3:], device=device)
                hasher.commit_after_verify(
                    verify_ids,
                    batch.req_pool_indices,
                    torch.tensor(commits, dtype=torch.int32, device=device),
                )
                torch.testing.assert_close(hasher.history, expected, rtol=0, atol=0)

    def test_cuda_graph_replay_updates_request_boundaries(self):
        hasher = _hasher("cuda")
        ids, batch, _, _ = _batch([4, 4, 4], 6, "cuda", slots=[6, 1, 4, 0], cap=True)
        captured = batch.spec_info.ragged_verify_layout
        history = hasher.history.clone()

        def forward():
            return hasher(ids, batch), token_req_indices(batch, num_tokens=ids.numel())

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()

        for lens, slots in (
            ([4, 2, 4], [4, 6, 1, 0]),
            ([1, 5], [1, 6, 0, 0]),
            ([6, 6], [6, 4, 0, 0]),
            ([2, 1, 1], [1, 4, 6, 0]),
        ):
            with self.subTest(lens=lens):
                live_ids, live, sequences, prefixes = _batch(
                    lens, 6, "cuda", bucket=12, slots=slots, cap=True
                )
                layout = live.spec_info.ragged_verify_layout
                ids.copy_(live_ids)
                batch.positions.copy_(live.positions)
                batch.req_pool_indices.copy_(live.req_pool_indices)
                # Match DecodeCudaGraphRunner._stage_ragged_verify_layout:
                # only lens and qo_indptr are refreshed, not extend_start_loc.
                captured.verify_lens.copy_(layout.verify_lens)
                captured.qo_indptr_device.copy_(layout.qo_indptr_device)
                graph.replay()
                torch.cuda.synchronize()
                self._check(hasher, ids, batch, sequences, prefixes, output=output)
                torch.testing.assert_close(hasher.history, history, rtol=0, atol=0)

    def test_cuda_graph_variants_share_staged_layout(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        # Candidate attention captures multiple graphs for one token bucket.
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.max_bs = 4
        runner.captured_req_width = 6
        runner.capture_num_tokens = [6, 12, 18, 24]
        runner.device = "cuda"
        runner._captured_ragged_layouts = {}
        hasher = _hasher("cuda")
        graphs = []
        for _ in range(2):
            ids, batch, _, _ = _batch([4, 4, 4], 6, "cuda", slots=[6, 1, 4, 0])
            batch.spec_info.ragged_verify_layout = runner._capture_ragged_verify_layout(
                12
            )

            def forward():
                return hasher(ids, batch), token_req_indices(batch, num_tokens=12)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = forward()
            graphs.append((graph, ids, batch, output))

        live_ids, live, sequences, prefixes = _batch(
            [4, 2, 4], 6, "cuda", bucket=12, slots=[4, 6, 1, 0]
        )
        runner._stage_ragged_verify_layout(live.spec_info.ragged_verify_layout, 12)
        for variant, (graph, ids, batch, output) in enumerate(graphs):
            with self.subTest(variant=variant):
                ids.copy_(live_ids)
                batch.positions.copy_(live.positions)
                batch.req_pool_indices.copy_(live.req_pool_indices)
                graph.replay()
                torch.cuda.synchronize()
                self._check(hasher, ids, batch, sequences, prefixes, output=output)


if __name__ == "__main__":
    unittest.main()
