# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slowfast_nfnets.slowfast_nfnet."""

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import numpy as np

from slowfast_nfnets import slowfast_nfnet


rng = np.random.default_rng(12345)
_RAND_SPEC = rng.uniform(-1., 1., size=(2, 400, 128, 1)).astype(np.float32)


class SlowfastNfnetTest(parameterized.TestCase):

  def test_slow_nfnet_f0_runs(self):
    def forward(inputs):
      model = slowfast_nfnet.NFNet(variant='F0-slow')
      return model(inputs, is_training=True)

    init_fn, apply_fn = hk.transform_with_state(forward)
    key = jax.random.PRNGKey(42)
    params, state = init_fn(key, _RAND_SPEC)
    out, _ = jax.jit(apply_fn)(params, state, key, _RAND_SPEC)
    self.assertListEqual(list(out.shape), [2, 3072])

  def test_fast_nfnet_f0_runs(self):
    def forward(inputs):
      model = slowfast_nfnet.NFNet(variant='F0-fast')
      return model(inputs, is_training=True)

    init_fn, apply_fn = hk.transform_with_state(forward)
    key = jax.random.PRNGKey(42)
    params, state = init_fn(key, _RAND_SPEC)
    out, _ = jax.jit(apply_fn)(params, state, key, _RAND_SPEC)
    self.assertListEqual(list(out.shape), [2, 384])

  def test_slowfast_nfnet_f0_runs(self):
    def forward(inputs):
      model = slowfast_nfnet.SlowFastNFNet()
      return model(inputs, is_training=True)

    init_fn, apply_fn = hk.transform_with_state(forward)
    key = jax.random.PRNGKey(42)
    params, state = init_fn(key, _RAND_SPEC)
    out, _ = jax.jit(apply_fn)(params, state, key, _RAND_SPEC)
    self.assertListEqual(list(out.shape), [2, 3456])


if __name__ == '__main__':
  absltest.main()
