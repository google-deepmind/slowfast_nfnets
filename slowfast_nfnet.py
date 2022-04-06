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
"""Slowfast Norm-Free Nets (https://arxiv.org/abs/2111.12124).

This class of network contains a fast and a slow branch, both based on the NFNet
backbone (https://arxiv.org/abs/2102.06171). The is also a fushion layer mixing
fast features to the slow branch.

The NFNet backbone follows the original implementation:

  https://github.com/deepmind/deepmind-research/blob/master/nfnets/nfnet.py

with changes to allow for customized kernel and stride patterns.
"""

from typing import Text, Optional, Sequence, Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from slowfast_nfnets import base


class NFNet(hk.Module):
  """Normalizer-Free networks, The Next Generation."""

  variant_dict = base.slowfast_nfnet_params

  def __init__(self,
               variant: Text = 'F0',
               width: float = 1.0,
               se_ratio: float = 0.5,
               alpha: float = 0.2,
               stochdepth_rate: float = 0.1,
               drop_rate: Optional[float] = None,
               activation: Text = 'gelu',
               # Multiplier for the final conv channel count
               final_conv_mult: int = 2,
               final_conv_ch: Optional[int] = None,
               use_two_convs: bool = True,
               name: Optional[Text] = 'NFNet'):
    super().__init__(name=name)
    self.variant = variant
    self.width = width
    self.se_ratio = se_ratio
    # Get variant info
    block_params = self.variant_dict[self.variant]
    self.width_pattern = block_params['width']
    self.depth_pattern = block_params['depth']
    self.bneck_pattern = block_params['expansion']
    self.group_pattern = block_params['group_width']
    self.big_pattern = block_params['big_width']
    stem_kernel_pattern = block_params['stem_kernel_pattern']
    stem_stride_pattern = block_params['stem_stride_pattern']
    kernel_pattern = block_params['kernel_pattern']
    stride_pattern = block_params['stride_pattern']
    self.activation = base.nonlinearities[activation]
    if drop_rate is None:
      self.drop_rate = block_params['drop_rate']
    else:
      self.drop_rate = drop_rate
    self.which_conv = base.WSConv2D
    # Stem
    ch = self.width_pattern[0] // 2
    self.stem = hk.Sequential([
        self.which_conv(ch // 8, kernel_shape=stem_kernel_pattern[0],
                        stride=stem_stride_pattern[0], padding='SAME',
                        name='stem_conv0'),
        self.activation,
        self.which_conv(ch // 4, kernel_shape=stem_kernel_pattern[1],
                        stride=stem_stride_pattern[1], padding='SAME',
                        name='stem_conv1'),
        self.activation,
        self.which_conv(ch // 2, kernel_shape=stem_kernel_pattern[2],
                        stride=stem_stride_pattern[2], padding='SAME',
                        name='stem_conv2'),
        self.activation,
        self.which_conv(ch, kernel_shape=stem_kernel_pattern[3],
                        stride=stem_stride_pattern[3], padding='SAME',
                        name='stem_conv3'),
    ])

    # Body
    self.blocks = []
    expected_std = 1.0
    num_blocks = sum(self.depth_pattern)
    index = 0  # Overall block index
    block_args = zip(self.width_pattern, self.depth_pattern, self.bneck_pattern,
                     self.group_pattern, self.big_pattern, kernel_pattern,
                     stride_pattern)
    for (block_width, stage_depth, expand_ratio,
         group_size, big_width, kernel, stride) in block_args:
      for block_index in range(stage_depth):
        # Scalar pre-multiplier so each block sees an N(0,1) input at init
        beta = 1./ expected_std
        # Block stochastic depth drop-rate
        block_stochdepth_rate = stochdepth_rate * index / num_blocks
        out_ch = (int(block_width * self.width))
        self.blocks += [NFBlock(ch,
                                out_ch,
                                expansion=expand_ratio,
                                se_ratio=se_ratio,
                                group_size=group_size,
                                kernel_shape=kernel,
                                stride=stride if block_index == 0 else (1, 1),
                                beta=beta,
                                alpha=alpha,
                                activation=self.activation,
                                which_conv=self.which_conv,
                                stochdepth_rate=block_stochdepth_rate,
                                big_width=big_width,
                                use_two_convs=use_two_convs,
                                )]
        ch = out_ch
        index += 1
         # Reset expected std but still give it 1 block of growth
        if block_index == 0:
          expected_std = 1.0
        expected_std = (expected_std **2 + alpha**2)**0.5

    # Head
    if final_conv_mult is None:
      if final_conv_ch is None:
        raise ValueError('Must provide one of final_conv_mult or final_conv_ch')
      ch = final_conv_ch
    else:
      ch = int(final_conv_mult * ch)
    self.final_conv = self.which_conv(ch, kernel_shape=1,
                                      padding='SAME', name='final_conv')

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    """Return the output of the final layer without any [log-]softmax."""
    # Stem
    out = self.stem(x)
    # Blocks
    for block in self.blocks:
      out, _ = block(out, is_training=is_training)
    # Final-conv->activation, pool, dropout, classify
    out = self.activation(self.final_conv(out))
    out = jnp.mean(out, [1, 2])
    return out


class NFBlock(hk.Module):
  """Normalizer-Free Net Block."""

  def __init__(self,
               in_ch: int,
               out_ch: int,
               expansion: float = 0.5,
               se_ratio: float = 0.5,
               kernel_shape: Sequence[int] = (3, 1),
               second_conv_kernel_shape: Sequence[int] = (1, 3),
               group_size: int = 128,
               stride: Sequence[int] = (1, 1),
               beta: float = 1.0,
               alpha: float = 0.2,
               which_conv: Any = base.WSConv2D,
               activation: Any = jax.nn.gelu,
               big_width: bool = True,
               use_two_convs: bool = True,
               stochdepth_rate: Optional[float] = None,
               name: Optional[Text] = None):
    super().__init__(name=name)
    self.in_ch, self.out_ch = in_ch, out_ch
    self.expansion = expansion
    self.se_ratio = se_ratio
    self.kernel_shape = kernel_shape
    self.activation = activation
    self.beta, self.alpha = beta, alpha
    # Mimic resnet style bigwidth scaling?
    width = int((self.out_ch if big_width else self.in_ch) * expansion)
    # Round expanded with based on group count
    self.groups = width // group_size
    self.width = group_size * self.groups
    self.stride = stride
    self.use_two_convs = use_two_convs
    # Conv 0 (typically expansion conv)
    self.conv0 = which_conv(self.width, kernel_shape=1, padding='SAME',
                            name='conv0')
    # Grouped NxN conv
    self.conv1 = which_conv(self.width, kernel_shape=kernel_shape,
                            stride=stride, padding='SAME',
                            feature_group_count=self.groups, name='conv1')
    if self.use_two_convs:
      self.conv1b = which_conv(self.width,
                               kernel_shape=second_conv_kernel_shape,
                               stride=1, padding='SAME',
                               feature_group_count=self.groups, name='conv1b')
    # Conv 2, typically projection conv
    self.conv2 = which_conv(self.out_ch, kernel_shape=1, padding='SAME',
                            name='conv2')
    # Use shortcut conv on channel change or downsample.
    self.use_projection = np.prod(stride) > 1 or self.in_ch != self.out_ch
    if self.use_projection:
      self.conv_shortcut = which_conv(self.out_ch, kernel_shape=1,
                                      padding='SAME', name='conv_shortcut')
    # Squeeze + Excite Module
    self.se = base.SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)

    # Are we using stochastic depth?
    self._has_stochdepth = (stochdepth_rate is not None and
                            stochdepth_rate > 0. and stochdepth_rate < 1.0)
    if self._has_stochdepth:
      self.stoch_depth = base.StochDepth(stochdepth_rate)

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    out = self.activation(x) * self.beta
    if np.prod(self.stride) > 1:  # Average-pool downsample.
      pool_size = [1] + list(self.stride) + [1]
      shortcut = hk.avg_pool(out, window_shape=pool_size,
                             strides=pool_size, padding='SAME')
      if self.use_projection:
        shortcut = self.conv_shortcut(shortcut)
    elif self.use_projection:
      shortcut = self.conv_shortcut(out)
    else:
      shortcut = x
    out = self.conv0(out)
    out = self.conv1(self.activation(out))
    if self.use_two_convs:
      out = self.conv1b(self.activation(out))
    out = self.conv2(self.activation(out))
    out = (self.se(out) * 2) * out  # Multiply by 2 for rescaling
    # Get average residual standard deviation for reporting metrics.
    res_avg_var = jnp.mean(jnp.var(out, axis=[0, 1, 2]))
    # Apply stochdepth if applicable.
    if self._has_stochdepth:
      out = self.stoch_depth(out, is_training)
    # SkipInit Gain
    out = out * hk.get_parameter('skip_gain', (), out.dtype, init=jnp.zeros)
    return out * self.alpha + shortcut, res_avg_var


class FuseFast2Slow(hk.Module):
  """Fuses the information from the Fast pathway to the Slow pathway."""

  def __init__(self,
               channels: int,
               kernel_size: Sequence[int] = (7, 1),
               stride: Sequence[int] = (4, 1),
               which_conv: Any = base.WSConv2D,
               activation: Any = jax.nn.gelu,
               name: Optional[Text] = None):
    super().__init__(name=name)
    self._activation = activation
    self._which_conv = which_conv
    self._conv_f2s = which_conv(channels, kernel_shape=kernel_size,
                                stride=stride, padding='SAME',
                                with_bias=True, name='conv_f2s')

  def __call__(self, x_f: chex.Array, x_s: chex.Array,
               is_training: bool) -> chex.Array:
    fuse = self._conv_f2s(x_f)
    fuse = self._activation(fuse)
    x_s_fuse = jnp.concatenate([x_s, fuse], axis=-1)
    return x_s_fuse


class SlowFastNFNet(hk.Module):
  """Slow fast NFNet-F0."""

  def __init__(self,
               variant: Text = 'F0',
               time_ratio: int = 4,
               activation: Any = jax.nn.gelu,
               fusion_conv_channel_ratio: int = 2,
               name: Optional[Text] = None):
    super().__init__(name=name)
    self._time_ratio = time_ratio
    self._activation = activation
    self._fast_net = NFNet(variant=f'{variant}-fast')
    self._slow_net = NFNet(variant=f'{variant}-slow')
    self._depth_pattern = self._slow_net.depth_pattern
    self._fast2slow = []
    for channels in self._fast_net.width_pattern[0] * np.array([1, 4, 8, 16]):
      self._fast2slow.append(
          FuseFast2Slow(
              channels=channels * fusion_conv_channel_ratio))

  def __call__(self, x: chex.Array, is_training: bool) -> chex.Array:
    h_s = hk.max_pool(
        x, window_shape=(1, self._time_ratio, 1, 1),
        strides=(1, self._time_ratio, 1, 1),
        padding='SAME')
    h_f = x

    h_s = self._slow_net.stem(h_s)
    h_f = self._fast_net.stem(h_f)

    depth = 0
    for block_group, num_blocks in enumerate(self._depth_pattern):
      f2s_block = self._fast2slow[block_group]
      h_s = f2s_block(h_f, h_s, is_training)
      for _ in range(num_blocks):
        fast_block = self._fast_net.blocks[depth]
        slow_block = self._slow_net.blocks[depth]
        h_f, _ = fast_block(h_f, is_training=is_training)
        h_s, _ = slow_block(h_s, is_training=is_training)
        depth += 1

    assert depth == len(self._fast_net.blocks)

    h_f = self._activation(self._fast_net.final_conv(h_f))
    h_s = self._activation(self._slow_net.final_conv(h_s))
    h_f = jnp.mean(h_f, [1, 2])
    h_s = jnp.mean(h_s, [1, 2])
    out = jnp.concatenate([h_f, h_s], axis=-1)

    return out
