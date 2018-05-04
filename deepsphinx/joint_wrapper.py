# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np

from tensorflow import float32
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import tf_logging
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import gen_array_ops 
from tensorflow.python.ops import numerics
from tensorflow.python.ops.logging_ops import Print
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import AttentionMechanism


__all__ = [
    "JointWrapper",
    "JointWrapperState"
]

print_all = False

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access

def _max_segment_length_mask(max_segment_length, length):
  mask_high_len = math_ops.range(length)+1
  mask_low_len = gen_math_ops.minimum(length-mask_high_len+max_segment_length, length)
  mask_high_bool = array_ops.sequence_mask(mask_high_len, maxlen=length)
  mask_low_bool = array_ops.reverse(array_ops.sequence_mask(mask_low_len, maxlen=length), axis=[-1])
  return gen_math_ops.logical_and(mask_high_bool, mask_low_bool)

def my_tf_round(x, decimals = 0):
  multiplier = constant_op.constant(10**decimals, dtype=x.dtype)
  return math_ops.round(x * multiplier) / multiplier

def _mask_log_prob_matrix(log_prob_matrix, 
                          max_segment_length, 
                          length, 
                          log_prob_mask_value):
  batch_size = array_ops.shape(log_prob_matrix)[0]
  log_prob_matrix_mask = gen_array_ops.tile(array_ops.expand_dims(
      _max_segment_length_mask(max_segment_length, length), 0),
    [batch_size, 1, 1])
  score_mask_values = float('-inf') * array_ops.ones_like(log_prob_matrix)
  masked_log_prob_matrix = array_ops.where(log_prob_matrix_mask, 
    log_prob_matrix, score_mask_values)
  return math_ops.log(clip_ops.clip_by_value(nn_ops.softmax(masked_log_prob_matrix), 1e-16, 1.0))

def _mask_prob_matrix2(prob_matrix, 
                       max_segment_length, 
                       length):
  batch_size = array_ops.shape(prob_matrix)[0]
  prob_matrix_mask = gen_array_ops.tile(array_ops.expand_dims(
      _max_segment_length_mask(max_segment_length, length), 0),
    [batch_size, 1, 1])
  score_mask_values = 1e-10 * array_ops.ones_like(prob_matrix)
  masked_prob_matrix = array_ops.where(array_ops.transpose(prob_matrix_mask, perm=[0,2,1]), 
    prob_matrix, score_mask_values)
  return nn_ops.softmax(math_ops.log(clip_ops.clip_by_value(masked_prob_matrix, 1e-10, 1.0)))

class JointWrapperState(
    collections.namedtuple("JointWrapperState",
                           ("cell_state", "attention", "time", "alignments", "previous_output",
                            "beam_probs", "beam_probs_history", "beam_attention", 
                            "beam_alignments", "beam_previous_alignments", "beam_previous_outputs", 
                            "beam_states"))): 
                            # "beam_outputs"))):  
                            # For JointWrapperState - Karan
  """`namedtuple` storing the state of a `JointWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `JointWrapperState`.

    Returns:
      A new `JointWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    return super(JointWrapperState, self)._replace(**kwargs)


def hardmax(logits, name=None):
  """Returns batched one-hot vectors.

  The depth index containing the `1` is that of the maximum logit value.

  Args:
    logits: A batch tensor of logit values.
    name: Name to use when creating ops.
  Returns:
    A batched one-hot tensor.
  """
  with ops.name_scope(name, "Hardmax", [logits]):
    logits = ops.convert_to_tensor(logits, name="logits")
    if logits.get_shape()[-1].value is not None:
      depth = logits.get_shape()[-1].value
    else:
      depth = array_ops.shape(logits)[-1]
    return array_ops.one_hot(
        math_ops.argmax(logits, -1), depth, dtype=logits.dtype)


def _compute_attention(attention_mechanism, cell_output, previous_alignments,
                       attention_layer, expansion=1):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments = attention_mechanism(
      cell_output, previous_alignments, expansion)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = array_ops.expand_dims(alignments, 1)
  # Context is the inner product of alignments and values along the
  # memory time dimension.previous_alignments
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  context = math_ops.matmul(expanded_alignments, attention_mechanism.values) 
  context = array_ops.squeeze(context, [1])

  if attention_layer is not None:
    attention = attention_layer(array_ops.concat([cell_output, context], 1))
  else:
    attention = context

  return attention, alignments


class JointWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               max_segment_length,
               attention_layer_size=None,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               name=None):
    """Construct the `JointWrapper`.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `JointWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = JointWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: A list of `AttentionMechanism` instances or a single
        instance.
      attention_layer_size: A list of Python integers or a single Python
        integer, the depth of the attention (output) layer(s). If None
        (default), use the context as attention at each time step. Otherwise,
        feed the context and cell output into the attention layer to generate
        attention at each time step. If attention_mechanism is a list,
        attention_layer_size must be a list of the same length.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      name: Name to use when creating ops.

    Raises:
      TypeError: `attention_layer_size` is not None and (`attention_mechanism`
        is a list but `attention_layer_size` is not; or vice versa).
      ValueError: if `attention_layer_size` is not None, `attention_mechanism`
        is a list, and its length does not match that of `attention_layer_size`.
    """
    super(JointWrapper, self).__init__(name=name)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if isinstance(attention_mechanism, (list, tuple)):
      raise TypeError(
              "Only a singla attention_mechanism must be given to JointWrapper."
              "AttentionMechanism, saw %s given."
              % len(attention_mechanism))
    else:
      if not isinstance(attention_mechanism, AttentionMechanism):
        raise TypeError(
            "attention_mechanism must be an AttentionMechanism"
            "instance, saw type: %s"
            % type(attention_mechanism).__name__)

    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None:
      self._attention_layer = layers_core.Dense(attention_layer_size, name="attention_layer", 
        use_bias=False)
      self._attention_layer_size = attention_layer_size
    else:
      self._attention_layer = None
      self._attention_layer_size = attention_mechanism.values.get_shape()[-1].value                  

    self._cell = cell
    self._attention_mechanism = attention_mechanism
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._max_segment_length = max_segment_length
    with ops.name_scope(name, "JointWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            final_state_tensor.shape[0].value
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing JointWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            self._batch_size_checks(state_batch_size, error_message)):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  def _batch_size_checks(self, batch_size, error_message):
    return [check_ops.assert_equal(batch_size,
                                   self._attention_mechanism.batch_size,
                                   message=error_message)]

  @property
  def output_size(self):
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    """The `state_size` property of `JointWrapper`.

    Returns:
      An `JointWrapperState` tuple containing shapes used by this object.
    """    
    alignments_size = self._attention_mechanism.alignments_size
    state_size = JointWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        alignments=alignments_size,
        previous_output=self._cell.output_size,
        beam_probs=alignments_size*tensor_shape.TensorShape([]), 
        beam_probs_history=None,
        beam_attention=(alignments_size, self._attention_layer_size), 
        beam_alignments=(alignments_size, alignments_size),
        beam_previous_alignments=(alignments_size, alignments_size),
        beam_previous_outputs=(alignments_size*alignments_size, self._cell.output_size),
        beam_states=nest.map_structure(lambda x: (alignments_size, x), self._cell.state_size)) 
    return state_size

  def zero_state(self, batch_size, dtype):
    """Return an initial (zero) state tuple for this `JointWrapper`.

    **NOTE** Please see the initializer documentation for details of how
    to call `zero_state` if using an `JointWrapper` with a
    `BeamSearchDecoder`.

    Args:
      batch_size: `0D` integer tensor: the batch size.
      dtype: The internal state data type.

    Returns:
      An `JointWrapperState` tuple containing zeroed out tensors and,
      possibly, empty `TensorArray` objects.

    Raises:
      ValueError: (or, possibly at runtime, InvalidArgument), if
        `batch_size` does not match the output size of the encoder passed
        to the wrapper object at initialization time.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of JointWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          self._batch_size_checks(batch_size, error_message)):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
    alignments_size = self._attention_mechanism.alignments_size
    z_state =  JointWrapperState(
      cell_state=cell_state,
      time=array_ops.zeros([], dtype=dtypes.int32),
      attention=_zero_state_tensors(self._attention_layer_size, batch_size,dtype),
      alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
      previous_output=array_ops.ones((batch_size, self._cell.output_size), dtype), #UNUSED
      beam_probs=_zero_state_tensors(alignments_size, batch_size, dtype),
      beam_probs_history=tensor_array_ops.TensorArray(dtype, size=0, dynamic_size=True),
      beam_attention=gen_array_ops.reshape(_zero_state_tensors(self._attention_layer_size, batch_size*alignments_size, dtype),
        (batch_size, alignments_size, -1)), # Added nn_ops.softmax - 01/05/2018 Removed - 02/05/2018
      beam_alignments=gen_array_ops.reshape(self._attention_mechanism.initial_alignments(batch_size*alignments_size, dtype),
        (batch_size, alignments_size, -1)), 
      beam_previous_alignments=gen_array_ops.reshape(self._attention_mechanism.initial_alignments(batch_size*alignments_size, dtype),
        (batch_size, alignments_size, -1)),
      beam_previous_outputs=array_ops.ones((batch_size, alignments_size*alignments_size, self._cell.output_size)),
      beam_states=nest.map_structure(lambda x: gen_array_ops.reshape(x, (batch_size, alignments_size, -1)), 
        self._cell.zero_state(batch_size*alignments_size, dtype)))
    return z_state

  def call(self, inputs, state):
    if not isinstance(state, JointWrapperState):
      raise TypeError("Expected state to be instance of JointWrapperState. "
                      "Received type %s instead."  % type(state))
    
    # inputs: y_{i-1}, state.attention: c_i, cell_state: s_{i-1}, cell_output: y_i, next_cell_state: s_i
    if print_all: inputs = Print(inputs, [state.time, math_ops.argmax(inputs, axis=-1)], message="inputs (argmax of one hot):", summarize=1000)
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
    if print_all: cell_output = Print(cell_output, [state.time, cell_output], message="cell_output:", summarize=1000)

    # Sizes for beam: batch_size, inputs_size, alignment_size, attention_size
    # beam_inputs_size: vocabulary_size, beam_alignment_size: T, beam_attention_size: dimension(c_i)
    beam_batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0] 
    beam_inputs_size = inputs.shape[1].value or array_ops.shape(inputs)[1] 
    beam_alignments_size = state.alignments.shape[-1].value or array_ops.shape(state.alignments)[-1] 
    beam_attention_size = state.attention.shape[-1].value or array_ops.shape(state.attention)[-1]

    # Calculating the needed reshaped tensors
    # beam_inputs: y_{i-1} x T, beam_inputs_big: y_{i-1} x T^2
    beam_inputs = gen_array_ops.reshape(gen_array_ops.tile(inputs, [1, beam_alignments_size]), 
      (beam_batch_size*beam_alignments_size, beam_inputs_size))
    if print_all: beam_inputs = Print(beam_inputs, [state.time, beam_inputs], message="beam_inputs:", summarize=1000)
    # beam_inputs beam_inputs, [beam_inputs], message="beam_inputs", summarize=10000)
    beam_inputs_big = gen_array_ops.reshape(gen_array_ops.tile(inputs, [1, beam_alignments_size*beam_alignments_size]), 
      (beam_batch_size*beam_alignments_size*beam_alignments_size, beam_inputs_size))
    if print_all: beam_inputs_big = Print(beam_inputs_big, [state.time, beam_inputs_big], message="beam_inputs_big:", summarize=1000)
    # beam_attention: c_{i,1:T}, beam_attention_big: c_{i,1:T} x T   
    beam_attention = gen_array_ops.reshape(state.beam_attention, 
      (beam_batch_size*beam_alignments_size, beam_attention_size)) # [h1, h2, ...]
    if print_all: beam_attention = Print(beam_attention, [state.time, beam_attention], message="beam_attention:", summarize=1000)
    beam_attention_big = gen_array_ops.reshape(gen_array_ops.tile(state.beam_attention, [1, 1, beam_alignments_size]), 
      (beam_batch_size*beam_alignments_size*beam_alignments_size, beam_attention_size)) # [[1, 1, ...], [2, 2, ...], ...]
    if print_all: beam_attention_big = Print(beam_attention_big, [state.time, beam_attention_big], message="beam_attention_big:", summarize=1000)
    # beam_cell_state: s_{i-1} x T, beam_cell_state_big: s_{i-1,1:T} x T
    beam_cell_state = nest.map_structure(lambda x: gen_array_ops.reshape(gen_array_ops.tile(x, [1, beam_alignments_size]), 
      (beam_batch_size*beam_alignments_size, -1)), state.cell_state)  # <<<<<< BAD IDEA ??? >>>>>>>
    beam_cell_state_big = nest.map_structure(lambda x: gen_array_ops.reshape(gen_array_ops.tile(x, [1, beam_alignments_size, 1]), 
      (beam_batch_size*beam_alignments_size*beam_alignments_size, -1)), state.beam_states) # [[1, 2, ...], [1, 2, ...], ...]
    
    # Calculating the next cell states and outputs
    # beam_cell_output: y_{i,1:T}, beam_next_cell_state, new_beam_states: s_{i,1:T}
    beam_inputs_processed = self._cell_input_fn(beam_inputs, beam_attention) # [h1, h2, ...]
    if print_all: beam_inputs_processed = Print(beam_inputs_processed, [state.time, beam_inputs_processed], message="beam_inputs_processed:", summarize=1000)
    beam_cell_output, beam_next_cell_state = self._cell(beam_inputs_processed, beam_cell_state) # [h1, h2, ...]  
    if print_all: beam_cell_output = Print(beam_cell_output, [state.time, beam_cell_output], message="beam_cell_output:", summarize=1000) 
    new_beam_states = nest.map_structure(lambda x: gen_array_ops.reshape(x, (beam_batch_size, beam_alignments_size, -1)), 
      beam_next_cell_state) # [h1, h2, ...]

    # Calculating the outputs 
    # beam_cell_output_big: y_{i,1:T,1:T}
    beam_inputs_processed_big = self._cell_input_fn(beam_inputs_big, beam_attention_big) # [[1, 1, ...], [2, 2, ...], ...]
    if print_all: beam_inputs_processed_big = Print(beam_inputs_processed_big, [state.time, beam_inputs_processed_big], message="beam_inputs_processed_big:", summarize=1000)
    beam_cell_output_big, _ = self._cell(beam_inputs_processed_big, beam_cell_state_big) # [[1, 2, ...], [1, 2, ...], ...]
    beam_cell_output_big = nn_ops.softmax(beam_cell_output_big)
            # Added softmax above instead of using it on state.beam_previous_outputs - 01/05/2018
    if print_all: beam_cell_output_big = Print(beam_cell_output_big, [state.time, beam_cell_output_big], message="beam_cell_output_big", summarize=1000)
    new_beam_previous_outputs = gen_array_ops.reshape(beam_cell_output_big, 
      (beam_batch_size, beam_alignments_size*beam_alignments_size, self._cell.output_size))
    if print_all: new_beam_previous_outputs = Print(new_beam_previous_outputs, [state.time, new_beam_previous_outputs], message="new_beam_previous_outputs:", summarize=1000)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying JointWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    with ops.control_dependencies(
        self._batch_size_checks(cell_batch_size, error_message)):
      cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    beam_cell_batch_size = (
        beam_cell_output.shape[0].value or array_ops.shape(beam_cell_output)[0])//(beam_alignments_size)
    beam_error_message = (
        "(BEAM Part) When applying JointWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    with ops.control_dependencies(
        self._batch_size_checks(beam_cell_batch_size, beam_error_message)):
      beam_cell_output = array_ops.identity(
          beam_cell_output, name="beam_checked_cell_output")

    beam_cell_batch_size_big = (
        beam_cell_output_big.shape[0].value or array_ops.shape(beam_cell_output_big)[0])//(beam_alignments_size*beam_alignments_size)
    beam_error_message_big = (
        "(BEAM^2 Part) When applying JointWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    with ops.control_dependencies(
        self._batch_size_checks(beam_cell_batch_size_big, beam_error_message_big)):
      beam_cell_output_big = array_ops.identity(
          beam_cell_output_big, name="beam_checked_cell_output_big")

    # previous_alignments, beam_previous_alignments: NOT USED LATER
    previous_alignments = [state.alignments]
    beam_previous_alignments = [gen_array_ops.reshape(state.beam_previous_alignments, 
      (beam_batch_size*beam_alignments_size, beam_alignments_size))]

    new_attention, new_alignments = _compute_attention(
        self._attention_mechanism, cell_output, previous_alignments,
        self._attention_layer if self._attention_layer else None)
    if print_all: new_attention = Print(new_attention, [state.time, new_attention], message="new_attention:", summarize=1000)
    if print_all: new_alignments = Print(new_alignments, [state.time, new_alignments], message="new_alignments:", summarize=1000)

    # Compute new beam alignments
    # beam_alignments: \alpha_{i+1,1:T}
    _, beam_alignments = _compute_attention(
        self._attention_mechanism, beam_cell_output, beam_previous_alignments,
        self._attention_layer if self._attention_layer else None, expansion=beam_alignments_size)
    if print_all: beam_alignments = Print(beam_alignments, [state.time, beam_alignments], message="beam_alignments", summarize=1000)
    new_beam_alignments = gen_array_ops.reshape(beam_alignments, 
      (beam_batch_size, beam_alignments_size, beam_alignments_size)) # [[1, 2, ...], [1, 2, ...], ...]
    if print_all: new_beam_alignments = Print(new_beam_alignments, [state.time, new_beam_alignments], message="new_beam_alignments:", summarize=1000)

    # Compute new beam attention
    # beam_attention: c_{i+1,1:T}
    memory_size = self._attention_mechanism.values.shape[-1].value or array_ops.shape(self._attention_mechanism.values.shape)[-1]
    new_beam_attention = self._attention_layer(array_ops.concat([
        # beam_cell_output,
        gen_array_ops.reshape(gen_array_ops.tile(cell_output, [1, beam_alignments_size]), 
          (beam_batch_size*beam_alignments_size, beam_inputs_size)),
        gen_array_ops.reshape(self._attention_mechanism.values, 
          (beam_batch_size*beam_alignments_size, memory_size))
      ], 1)) 
    new_beam_attention = gen_array_ops.reshape(new_beam_attention, 
      (beam_batch_size, beam_alignments_size, beam_attention_size))
    if print_all: new_beam_attention = Print(new_beam_attention, [state.time, new_beam_attention], message="new_beam_attention:", summarize=1000)
    new_beam_attention = numerics.verify_tensor_all_finite(new_beam_attention, msg="NEW_BEAM_ATTENTION PROBLEM!!!")

    # Compute new beam probs
    # beam_output_indices: extracted the index out of inputs and for positions of target 
    # probabilities from beam_cell_output_big
    beam_output_indices = gen_array_ops.reshape(gen_array_ops.tile(array_ops.expand_dims(math_ops.argmax(inputs, axis=1), 1), 
      [1, beam_alignments_size*beam_alignments_size]), 
      (beam_batch_size*beam_alignments_size*beam_alignments_size, 1)) 
    beam_output_indices = array_ops.concat([array_ops.expand_dims(math_ops.range(math_ops.cast(
      beam_batch_size*beam_alignments_size*beam_alignments_size, dtype=beam_output_indices.dtype), 
      dtype=beam_output_indices.dtype), 1), beam_output_indices], axis=1)
    if print_all: beam_output_indices = Print(beam_output_indices, [state.time, beam_output_indices], message="beam_output_indices:", summarize=1000)
    # beam_output_probs: extracted (gathered) the target probabilities from state.beam_previous_outputs
    # state.beam_previous_outputs: y_{i-1,1:T,1:T}
    beam_output_probs = gen_array_ops.reshape(array_ops.gather_nd(gen_array_ops.reshape( 
          # Removed softmax betweeen gather_nd and reshape and put it on self._cell(...) directly - 01/05/2018
          state.beam_previous_outputs, 
          (beam_batch_size*beam_alignments_size*beam_alignments_size, self._cell.output_size)), 
        beam_output_indices), 
      (beam_batch_size, beam_alignments_size, beam_alignments_size)) # [[1, 2, ...], [1, 2, ...], ...]
    beam_output_probs = nn_ops.softmax(math_ops.exp(beam_output_probs)) # Added softmax and exp above to normalize y over k - 03/05/2018 
    if print_all: beam_output_probs = Print(beam_output_probs, [state.time, beam_output_probs], message="beam_output_probs:", summarize=1000)
    # beam_alignment_probs: \alpha_{i-1,1:T,1:T}
    # beam_alignment_probs = _mask_prob_matrix2(state.beam_previous_alignments,         # HERE IS THE MASKING FOR MAX SEGMENT LENGTH CONTRAINT
      # self._max_segment_length,
      # beam_alignments_size)
    beam_alignment_probs = state.beam_previous_alignments
    # beam_alignment_probs = array_ops.transpose(state.beam_previous_alignments, perm=[0,2,1]) # [[1, 2, ...], [1, 2, ...], ...]      
    beam_alignment_probs = array_ops.transpose(beam_alignment_probs, perm=[0,2,1]) # [[1, 2, ...], [1, 2, ...], ...]
    # beam_added_probs: \alpha_{i-1,1:T,1:T} *elementwise multiplied* y_{i-1,1:T,1:T, TARGET} (addition in log domain)
    if print_all: beam_alignment_probs = Print(beam_alignment_probs, [state.time, beam_alignment_probs], message="beam_alignment_probs:", summarize=1000)
    beam_added_probs = math_ops.log(clip_ops.clip_by_value(beam_alignment_probs, 
      1e-16, 1.0)) + math_ops.log(clip_ops.clip_by_value(beam_output_probs, 
      1e-16, 1.0)) # [[1, 2, ...], [1, 2, ...], ...] 
    if print_all: beam_added_probs = Print(beam_added_probs, [state.time, beam_added_probs], message="beam_added_probs", summarize=1000)
    # new_beam_probs: beam_added_probs *matrix multplied* \beta_{i-1} (tiling and log(sum(exp(...))) in log domain)
    new_beam_probs = math_ops.reduce_logsumexp(beam_added_probs + 
      gen_array_ops.reshape(gen_array_ops.tile(state.beam_probs, 
        [1, beam_alignments_size]), (beam_batch_size, beam_alignments_size, beam_alignments_size)), axis=2)
    if print_all: new_beam_probs = Print(new_beam_probs, [state.time, new_beam_probs], message="new_beam_probs:", summarize=1000)

    if print_all: 
      state_dot_time = Print(state.time, [state.time, state.time], message="state.time:", summarize=1000)
    else:
      state_dot_time = state.time

    next_state = JointWrapperState(
        time=state_dot_time + 1,
        cell_state=next_cell_state,
        attention=new_attention,
        alignments=new_alignments,
        previous_output=cell_output,
        beam_probs=new_beam_probs,
        beam_probs_history=state.beam_probs_history.write(state_dot_time, new_beam_probs),
        beam_attention=new_beam_attention,
        beam_alignments=new_beam_alignments,
        beam_previous_alignments=state.beam_alignments,
        beam_previous_outputs=new_beam_previous_outputs,
        beam_states=new_beam_states)

    if self._output_attention:
      return new_attention, next_state
    else:
      return cell_output, next_state