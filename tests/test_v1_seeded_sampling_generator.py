# SPDX-License-Identifier: Apache-2.0
"""Tests for v1 seeded sampling generator handling."""

import mlx.core as mx
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.sampler import Sampler

from vllm_metal.v1.model_runner import (
    MetalModelRunner,
    RequestState,
    _create_request_generator,
)


def _constant_logits_model(vocab_size: int):
    def _model(input_ids: mx.array, cache=None) -> mx.array:  # noqa: ANN001
        batch_size = int(input_ids.shape[0])
        base = mx.arange(vocab_size, dtype=mx.float32)[None, None, :]
        return mx.broadcast_to(base, (batch_size, 1, vocab_size))

    return _model


def test_seeded_sampling_generator_advances_across_decode_steps() -> None:
    """Seeded sampling should reuse (and advance) a per-request generator."""
    vocab_size = 32
    runner = MetalModelRunner.__new__(MetalModelRunner)
    runner.device = torch.device("cpu")
    runner.model_args = {"vocab_size": vocab_size}
    runner._sampler = Sampler()
    runner._rust_state_manager = None
    runner.model = _constant_logits_model(vocab_size)

    sp = SamplingParams(temperature=1.0, seed=123)
    generator = _create_request_generator(runner.device, sp)
    assert generator is not None

    state = RequestState(
        token_ids=[1],
        cache=[],
        sampling_params=sp,
        generator=generator,
    )

    before = generator.get_state()
    runner._batched_decode([("r1", state)])
    after_first = generator.get_state()
    runner._batched_decode([("r1", state)])
    after_second = generator.get_state()

    assert not torch.equal(after_first, before)
    assert not torch.equal(after_second, after_first)
