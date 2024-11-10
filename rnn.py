"""RNN helpers for TensorFlow models.


@@bidirectional_dynamic_rnn
@@dynamic_rnn
@@raw_rnn
@@static_rnn
@@static_state_saving_rnn
@@static_bidirectional_rnn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def _concat(prefix, suffix, static=False):
    """Concat that enables int, list, tuple, Tensor or TensorShape values.

    This function takes a size specification, which can be an integer, a
    TensorShape, a Tensor, a list, or a tuple, and converts it into a concatenated
    Tensor (if static = False) or a list of integers (if static = True).

    Args:
        prefix: The prefix; usually the batch size (and/or time step size).
          (int, list, tuple, TensorShape, or Tensor.)
        suffix: TensorShape, int, list, tuple, or Tensor.
        static: If `True`, return a python list with possibly unknown dimensions. If
          `False`, return a `Tensor` with value `concat(prefix, suffix)`

    Returns:
        shape: the concatenation of prefix and suffix.

    Raises:
        ValueError: if `suffix` is not a int, list, tuple, TensorShape, or Tensor.
        ValueError: if `prefix` is not a int, list, tuple, TensorShape, or Tensor.
    """
    def _shape_list(x):
        if isinstance(x, int):
            return [x]
        elif isinstance(x, (list, tuple)):
            return list(x)
        elif isinstance(x, tf.TensorShape):
            return x.as_list()
        elif isinstance(x, tf.Tensor):
            return tf.get_static_value(x)
        else:
            raise ValueError("Unknown type for %s of type %s" % (x, type(x)))

    if static:
        p_shape = _shape_list(prefix)
        s_shape = _shape_list(suffix)
        if p_shape is None or s_shape is None:
            return tf.TensorShape(None)
        else:
            return tf.TensorShape(p_shape + s_shape)
    else:
        if isinstance(prefix, tf.Tensor):
            p = tf.reshape(prefix, [-1])
        else:
            p = tf.constant(_shape_list(prefix), dtype=tf.int32)
        if isinstance(suffix, tf.Tensor):
            s = tf.reshape(suffix, [-1])
        else:
            s = tf.constant(_shape_list(suffix), dtype=tf.int32)
        return tf.concat([p, s], axis=0)


# def _concat(prefix, suffix, static=False):
#   """Concat that enables int, Tensor or TensorShape values.

#   This function takes a size specification, which can be an integer, a
#   TensorShape, or a Tensor, and converts it into a concatenated Tensor
#   (if static = False) or a list of integers (if static = True).

#   Args:
#     prefix: The prefix; usually the batch size (and/or time step size).
#       (TensorShape, int, or Tensor.)
#     suffix: TensorShape, int, or Tensor.
#     static: If `True`, return a python list with possibly unknown dimensions. If
#       `False`, return a `Tensor` with value `concat(prefix, suffix)`

#   Returns:
#     shape: the concatenation of prefix and suffix.

#   Raises:
#     ValueError: if `suffix` is not a int, TensorShape, or Tensor.
#     ValueError: if `prefix` is not a int, TensorShape, or Tensor.
#   """
#   if isinstance(prefix, tf.Tensor):
#     p = prefix
#     p_static = tf.TensorShape(prefix.shape)
#   else:
#     p = None
#     if isinstance(prefix, int):
#       p_static = tf.TensorShape([prefix])
#     elif isinstance(prefix, tf.TensorShape):
#       p_static = prefix
#     else:
#       raise ValueError("Unknown type for prefix %s" % type(prefix))

#   if isinstance(suffix, tf.Tensor):
#     s = suffix
#     s_static = tf.TensorShape(suffix.shape)
#   else:
#     s = None
#     if isinstance(suffix, int):
#       s_static = tf.TensorShape([suffix])
#     elif isinstance(suffix, tf.TensorShape):
#       s_static = suffix
#     else:
#       raise ValueError("Unknown type for suffix %s" % type(suffix))

#   if static:
#     shape = tf.TensorShape(tf.TensorShape(p_static).as_list() + tf.TensorShape(s_static).as_list())
#   else:
#     p = p if p is not None else tf.constant(p_static.as_list(), dtype=tf.int32)
#     s = s if s is not None else tf.constant(s_static.as_list(), dtype=tf.int32)
#     shape = tf.concat([p, s], 0)
#   return shape

def _like_rnncell(cell):
  return isinstance(cell, tf.nn.rnn_cell.RNNCell)

def _transpose_batch_time(x):
  """Transpose the batch and time dimensions of a Tensor.

  Retains as much of the static shape information as possible.

  Args:
    x: A tensor of rank 2 or higher.

  Returns:
    x transposed along the first two dimensions.

  Raises:
    ValueError: if `x` is rank 1 or lower.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    raise ValueError(
        "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
        (x, x_static_shape))
  x_rank = tf.rank(x)
  x_t = tf.transpose(
      x, tf.concat(
          ([1, 0], tf.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tf.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t

def _best_effort_input_batch_size(flat_input):
  """Get static input batch size if available, with fallback to the dynamic one.

  Args:
    flat_input: An iterable of time major input Tensors of shape [max_time,
      batch_size, ...]. All inputs should have compatible batch sizes.

  Returns:
    The batch size in Python integer if available, or a scalar Tensor otherwise.

  Raises:
    ValueError: if there is any input with an invalid shape.
  """
  for input_ in flat_input:
    shape = input_.shape
    if shape.ndims is None:
      continue
    if shape.ndims < 2:
      raise ValueError(
          "Expected input tensor %s to have rank at least 2" % input_)
    batch_size = shape[1].value
    if batch_size is not None:
      return batch_size
  # Fallback to the dynamic batch size of the first input.
  return tf.shape(flat_input[0])[1]

def _infer_state_dtype(explicit_dtype, state):
  """Infer the dtype of an RNN state.

  Args:
    explicit_dtype: explicitly declared dtype or None.
    state: RNN's hidden state. Must be a Tensor or a nested iterable containing
      Tensors.

  Returns:
    dtype: inferred dtype of hidden state.

  Raises:
    ValueError: if `state` has heterogeneous dtypes or is empty.
  """
  if explicit_dtype is not None:
    return explicit_dtype
  elif tf.nest.is_nested(state):
    inferred_dtypes = [element.dtype for element in tf.nest.flatten(state)]
    if not inferred_dtypes:
      raise ValueError("Unable to infer dtype from empty state.")
    all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
    if not all_same:
      raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
    return inferred_dtypes[0]
  else:
    return state.dtype

def _rnn_step(
    time, sequence_length, min_sequence_length, max_sequence_length,
    zero_output, state, call_cell, state_size, skip_conditionals=False):

  flat_state = tf.nest.flatten(state)
  flat_zero_output = tf.nest.flatten(zero_output)

  def _copy_one_through(output, new_output):
    # If the state contains a scalar value we simply pass it through.
    if output.shape.ndims == 0:
      return new_output
    copy_cond = (time >= sequence_length)
    with tf.colocate_with(new_output):
      return tf.where(copy_cond, output, new_output)

  def _copy_some_through(flat_new_output, flat_new_state):
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    flat_new_output = [
        _copy_one_through(zero_output, new_output)
        for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
    flat_new_state = [
        _copy_one_through(state, new_state)
        for state, new_state in zip(flat_state, flat_new_state)]
    return flat_new_output + flat_new_state

  def _maybe_copy_some_through():
    """Run RNN step.  Pass through either no or some past state."""
    new_output, new_state = call_cell()

    tf.nest.assert_same_structure(state, new_state)

    flat_new_state = tf.nest.flatten(new_state)
    flat_new_output = tf.nest.flatten(new_output)
    return tf.cond(
        # if t < min_seq_len: calculate and return everything
        time < min_sequence_length, lambda: flat_new_output + flat_new_state,
        # else copy some of it through
        lambda: _copy_some_through(flat_new_output, flat_new_state))

  if skip_conditionals:
    # Instead of using conditionals, perform the selective copy at all time
    # steps.  This is faster when max_seq_len is equal to the number of unrolls
    # (which is typical for dynamic_rnn).
    new_output, new_state = call_cell()
    tf.nest.assert_same_structure(state, new_state)
    flat_new_state = tf.nest.flatten(new_state)
    flat_new_output = tf.nest.flatten(new_output)
    final_output_and_state = _copy_some_through(flat_new_output, flat_new_state)
  else:
    empty_update = lambda: flat_zero_output + flat_state
    final_output_and_state = tf.cond(
        # if t >= max_seq_len: copy all state through, output zeros
        time >= max_sequence_length, empty_update,
        # otherwise calculation is required: copy some or all of it through
        _maybe_copy_some_through)

  if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
    raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")
  final_output = final_output_and_state[:len(flat_zero_output)]
  final_state = final_output_and_state[len(flat_zero_output):]

  for output, flat_output in zip(final_output, flat_zero_output):
    output.set_shape(flat_output.get_shape())
  for substate, flat_substate in zip(final_state, flat_state):
    substate.set_shape(flat_substate.get_shape())

  final_output = tf.nest.pack_sequence_as(
      structure=zero_output, flat_sequence=final_output)
  final_state = tf.nest.pack_sequence_as(
      structure=state, flat_sequence=final_state)

  return final_output, final_state

def _reverse_seq(input_seq, lengths):
  """Reverse a list of Tensors up to specified lengths.

  Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, n_features)
               or nested tuples of tensors.
    lengths:   A `Tensor` of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply reverses
               the list.

  Returns:
    time-reversed sequence
  """
  if lengths is None:
    return list(reversed(input_seq))

  flat_input_seq = tuple(tf.nest.flatten(input_) for input_ in input_seq)

  flat_results = [[] for _ in range(len(input_seq))]
  for sequence in zip(*flat_input_seq):
    input_shape = tf.TensorShape(None)
    for input_ in sequence:
      input_shape = input_shape.merge_with(input_.get_shape())
      input_.set_shape(input_shape)

    # Join into (time, batch_size, depth)
    s_joined = tf.stack(sequence)

    # Reverse along dimension 0
    s_reversed = tf.reverse_sequence(s_joined, lengths, seq_axis=0, batch_axis=1)
    # Split again into list
    result = tf.unstack(s_reversed)
    for r, flat_result in zip(result, flat_results):
      r.set_shape(input_shape)
      flat_result.append(r)

  results = [tf.nest.pack_sequence_as(structure=input_, flat_sequence=flat_result)
             for input_, flat_result in zip(input_seq, flat_results)]
  return results

def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):

  if not _like_rnncell(cell_fw):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not _like_rnncell(cell_bw):
    raise TypeError("cell_bw must be an instance of RNNCell")

  with tf.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with tf.variable_scope("fw") as fw_scope:
      output_fw, output_state_fw = dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
          initial_state=initial_state_fw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=fw_scope)

    # Backward direction
    if not time_major:
      time_dim = 1
      batch_dim = 0
    else:
      time_dim = 0
      batch_dim = 1

    def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return tf.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_axis=seq_dim, batch_axis=batch_dim)
      else:
        return tf.reverse(input_, axis=[seq_dim])

    with tf.variable_scope("bw") as bw_scope:
      inputs_reverse = _reverse(
          inputs, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_axis=batch_dim)
      tmp, output_state_bw = dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)

  output_bw = _reverse(
      tmp, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)

  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)

def dynamic_rnn(cell, inputs, att_scores=None, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):

  if not _like_rnncell(cell):
    raise TypeError("cell must be an instance of RNNCell")

  flat_input = tf.nest.flatten(inputs)

  if not time_major:
    # (B,T,D) => (T,B,D)
    flat_input = [tf.convert_to_tensor(input_) for input_ in flat_input]
    flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

  parallel_iterations = parallel_iterations or 32
  if sequence_length is not None:
    sequence_length = tf.to_int32(sequence_length)
    if sequence_length.get_shape().ndims not in (None, 1):
      raise ValueError(
          "sequence_length must be a vector of length batch_size, "
          "but saw shape: %s" % sequence_length.get_shape())
    sequence_length = tf.identity(  # Just to find it in the graph.
        sequence_length, name="sequence_length")

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with tf.variable_scope(scope or "rnn") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)
    batch_size = _best_effort_input_batch_size(flat_input)

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = tf.shape(x)
      packed_shape = tf.stack(shape)
      return tf.Assert(
          tf.reduce_all(tf.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    if sequence_length is not None:
      # Perform some shape validation
      with tf.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = tf.identity(
            sequence_length, name="CheckSeqLen")

    inputs = tf.nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, final_state) = _dynamic_rnn_loop(
        cell,
        inputs,
        state,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        att_scores=att_scores,
        sequence_length=sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (T,B,D) => (B,T,D)
      outputs = tf.nest.map_structure(_transpose_batch_time, outputs)

    return (outputs, final_state)

def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      att_scores=None,
                      sequence_length=None,
                      dtype=None):

  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  state_size = cell.state_size

  flat_input = tf.nest.flatten(inputs)
  flat_output_size = tf.nest.flatten(cell.output_size)

  # Construct an initial output
  input_shape = tf.shape(flat_input[0])
  time_steps = input_shape[0]
  batch_size = _best_effort_input_batch_size(flat_input)

  inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

  const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

  for shape in inputs_got_shape:
    if not shape[2:].is_fully_defined():
      raise ValueError(
          "Input size (depth of inputs) must be accessible via shape inference,"
          " but saw value None.")
    got_time_steps = shape[0].value
    got_batch_size = shape[1].value
    if const_time_steps != got_time_steps:
      raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
      raise ValueError(
          "Batch_size is not the same for all the elements in the input.")

  # Prepare dynamic conditional copying of state & output
  def _create_zero_arrays(size):
    size = _concat(batch_size, size)
    return tf.zeros(
        tf.stack(size), _infer_state_dtype(dtype, state))

  flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
  zero_output = tf.nest.pack_sequence_as(structure=cell.output_size,
                                         flat_sequence=flat_zero_output)

  if sequence_length is not None:
    min_sequence_length = tf.reduce_min(sequence_length)
    max_sequence_length = tf.reduce_max(sequence_length)

  time = tf.constant(0, dtype=tf.int32, name="time")

  with tf.name_scope("dynamic_rnn") as scope:
    base_name = scope

  def _create_ta(name, dtype):
    return tf.TensorArray(dtype=dtype,
                          size=time_steps,
                          tensor_array_name=base_name + name)

  output_ta = tuple(_create_ta("output_%d" % i,
                               _infer_state_dtype(dtype, state))
                    for i in range(len(flat_output_size)))
  input_ta = tuple(_create_ta("input_%d" % i, flat_input[i].dtype)
                   for i in range(len(flat_input)))

  input_ta = tuple(ta.unstack(input_)
                   for ta, input_ in zip(input_ta, flat_input))

  def _time_step(time, output_ta_t, state, att_scores=None):
    """Take a time step of the dynamic RNN.

    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.

    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """

    input_t = tuple(ta.read(time) for ta in input_ta)
    # Restore some shape information
    for input_, shape in zip(input_t, inputs_got_shape):
      input_.set_shape(shape[1:])

    input_t = tf.nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
    if att_scores is not None:
        att_score = att_scores[:, time, :]
        call_cell = lambda: cell(input_t, state, att_score)
    else:
        call_cell = lambda: cell(input_t, state)

    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()

    # Pack state if using state tuples
    output = tf.nest.flatten(output)

    output_ta_t = tuple(
        ta.write(time, out) for ta, out in zip(output_ta_t, output))
    if att_scores is not None:
        return (time + 1, output_ta_t, new_state, att_scores)
    else:
        return (time + 1, output_ta_t, new_state)

  if att_scores is not None:
      _, output_final_ta, final_state, _ = tf.while_loop(
          cond=lambda time, *_: time < time_steps,
          body=_time_step,
          loop_vars=(time, output_ta, state, att_scores),
          parallel_iterations=parallel_iterations,
          swap_memory=swap_memory)
  else:
      _, output_final_ta, final_state = tf.while_loop(
          cond=lambda time, *_: time < time_steps,
          body=_time_step,
          loop_vars=(time, output_ta, state),
          parallel_iterations=parallel_iterations,
          swap_memory=swap_memory)

  # Unpack final output if not using output tuples.
  final_outputs = tuple(ta.stack() for ta in output_final_ta)

  # Restore some shape information
  for output, output_size in zip(final_outputs, flat_output_size):
    shape = _concat(
        [const_time_steps, const_batch_size], output_size, static=True)
    output.set_shape(shape)

  final_outputs = tf.nest.pack_sequence_as(
      structure=cell.output_size, flat_sequence=final_outputs)

  return (final_outputs, final_state)

# 继续根据前面的模式，完成其余的函数适配。

def raw_rnn(cell, loop_fn,
            parallel_iterations=None, swap_memory=False, scope=None):

    if not _like_rnncell(cell):
        raise TypeError("cell必须是RNNCell的实例")
    if not callable(loop_fn):
        raise TypeError("loop_fn必须是可调用的函数")

    parallel_iterations = parallel_iterations or 32

    # 创建一个新的作用域，其中缓存设备要么由父作用域确定，要么设置为与RNN其余部分相同的设备。
    with tf.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        time = tf.constant(0, dtype=tf.int32)
        (elements_finished, next_input, initial_state, emit_structure,
         init_loop_state) = loop_fn(
            time, None, None, None)  # time, cell_output, cell_state, loop_state
        flat_input = tf.nest.flatten(next_input)

        # 如果没有可用的loop_state，为while_loop创建一个替代的loop_state。
        loop_state = init_loop_state if init_loop_state is not None else tf.constant(0, dtype=tf.int32)

        input_shape = [input_.get_shape() for input_ in flat_input]
        static_batch_size = input_shape[0][0]

        for input_shape_i in input_shape:
            # 静态验证批次大小是否一致
            static_batch_size = static_batch_size.merge_with(input_shape_i[0])

        batch_size = static_batch_size.value
        if batch_size is None:
            batch_size = tf.shape(flat_input[0])[0]

        tf.nest.assert_same_structure(initial_state, cell.state_size)
        state = initial_state
        flat_state = tf.nest.flatten(state)
        flat_state = [tf.convert_to_tensor(s) for s in flat_state]
        state = tf.nest.pack_sequence_as(structure=state,
                                         flat_sequence=flat_state)

        if emit_structure is not None:
            flat_emit_structure = tf.nest.flatten(emit_structure)
            flat_emit_size = [emit.shape if emit.shape.is_fully_defined() else
                              tf.shape(emit) for emit in flat_emit_structure]
            flat_emit_dtypes = [emit.dtype for emit in flat_emit_structure]
        else:
            emit_structure = cell.output_size
            flat_emit_size = tf.nest.flatten(emit_structure)
            flat_emit_dtypes = [flat_state[0].dtype] * len(flat_emit_size)

        flat_emit_ta = [
            tf.TensorArray(
                dtype=dtype_i, dynamic_size=True, size=0, name="rnn_output_%d" % i)
            for i, dtype_i in enumerate(flat_emit_dtypes)]
        emit_ta = tf.nest.pack_sequence_as(structure=emit_structure,
                                           flat_sequence=flat_emit_ta)
        flat_zero_emit = [
            tf.zeros(_concat(batch_size, size_i), dtype_i)
            for size_i, dtype_i in zip(flat_emit_size, flat_emit_dtypes)]
        zero_emit = tf.nest.pack_sequence_as(structure=emit_structure,
                                             flat_sequence=flat_zero_emit)

        def condition(unused_time, elements_finished, *_):
            return tf.logical_not(tf.reduce_all(elements_finished))

        def body(time, elements_finished, current_input,
                 emit_ta, state, loop_state):
            """raw_rnn的内部循环体。

            Args:
              time: time标量。
              elements_finished: batch_size的布尔向量。
              current_input: 可能是嵌套的输入张量元组。
              emit_ta: 可能是嵌套的输出TensorArray元组。
              state: 可能是嵌套的状态张量元组。
              loop_state: 可能是嵌套的loop state张量。

            Returns:
              具有与Args相同大小的元组，但值已更新。
            """
            (next_output, cell_state) = cell(current_input, state)

            tf.nest.assert_same_structure(state, cell_state)
            tf.nest.assert_same_structure(cell.output_size, next_output)

            next_time = time + 1
            (next_finished, next_input, next_state, emit_output,
             next_loop_state) = loop_fn(
                next_time, next_output, cell_state, loop_state)

            tf.nest.assert_same_structure(state, next_state)
            tf.nest.assert_same_structure(current_input, next_input)
            tf.nest.assert_same_structure(emit_ta, emit_output)

            # 如果loop_fn为next_loop_state返回None，则重用之前的loop_state。
            loop_state = loop_state if next_loop_state is None else next_loop_state

            def _copy_some_through(current, candidate):
                """通过tf.where复制部分张量。"""
                def copy_fn(cur_i, cand_i):
                    with tf.colocate_with(cand_i):
                        return tf.where(elements_finished, cur_i, cand_i)
                return tf.nest.map_structure(copy_fn, current, candidate)

            emit_output = _copy_some_through(zero_emit, emit_output)
            next_state = _copy_some_through(state, next_state)

            emit_ta = tf.nest.map_structure(
                lambda ta, emit: ta.write(time, emit), emit_ta, emit_output)

            elements_finished = tf.logical_or(elements_finished, next_finished)

            return (next_time, elements_finished, next_input,
                    emit_ta, next_state, loop_state)

        returned = tf.while_loop(
            condition, body, loop_vars=[
                time, elements_finished, next_input,
                emit_ta, state, loop_state],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        (emit_ta, final_state, final_loop_state) = returned[-3:]

        if init_loop_state is None:
            final_loop_state = None

        return (emit_ta, final_state, final_loop_state)

def static_rnn(cell,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):

    if not _like_rnncell(cell):
        raise TypeError("cell必须是RNNCell的实例")
    if not tf.nest.is_nested(inputs):
        raise TypeError("inputs必须是一个序列")
    if not inputs:
        raise ValueError("inputs不能为空")

    outputs = []
    with tf.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # 获取输入序列的第一个元素
        first_input = inputs
        while tf.nest.is_nested(first_input):
            first_input = first_input[0]

        if first_input.get_shape().ndims != 1:

            input_shape = first_input.get_shape().with_rank_at_least(2)
            fixed_batch_size = input_shape[0]

            flat_inputs = tf.nest.flatten(inputs)
            for flat_input in flat_inputs:
                input_shape = flat_input.get_shape().with_rank_at_least(2)
                batch_size, input_size = input_shape[0], input_shape[1:]
                fixed_batch_size.merge_with(batch_size)
                for i, size in enumerate(input_size):
                    if size.value is None:
                        raise ValueError(
                            "输入的尺寸（inputs的第%d维度）必须通过形状推断获得，但发现值为None。" % i)
        else:
            fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = tf.shape(first_input)[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("如果未提供initial_state，必须指定dtype")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:  # 准备变量
            sequence_length = tf.convert_to_tensor(
                sequence_length, name="sequence_length")
            if sequence_length.get_shape().ndims not in (None, 1):
                raise ValueError(
                    "sequence_length必须是长度为batch_size的向量")

            def _create_zero_output(output_size):
                size = _concat(batch_size, output_size)
                output = tf.zeros(
                    tf.stack(size), _infer_state_dtype(dtype, state))
                shape = _concat(fixed_batch_size.value, output_size, static=True)
                output.set_shape(tf.TensorShape(shape))
                return output

            output_size = cell.output_size
            flat_output_size = tf.nest.flatten(output_size)
            flat_zero_output = tuple(
                _create_zero_output(size) for size in flat_output_size)
            zero_output = tf.nest.pack_sequence_as(
                structure=output_size, flat_sequence=flat_zero_output)

            sequence_length = tf.to_int32(sequence_length)
            min_sequence_length = tf.reduce_min(sequence_length)
            max_sequence_length = tf.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0:
                varscope.reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length is not None:
                (output, state) = _rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=cell.state_size)
            else:
                (output, state) = call_cell()

            outputs.append(output)

        return (outputs, state)

def static_state_saving_rnn(cell,
                            inputs,
                            state_saver,
                            state_name,
                            sequence_length=None,
                            scope=None):
    """接受状态保存器进行时间截断RNN计算的RNN。

    Args:
      cell: 一个RNNCell实例。
      inputs: 长度为T的输入列表，每个都是形状为[batch_size, input_size]的Tensor。
      state_saver: 一个具有state和save_state方法的状态保存对象。
      state_name: 字符串或字符串元组。用于state_saver的名称。
                  如果cell返回状态元组（即cell.state_size是一个元组），
                  那么state_name应该是与cell.state_size长度相同的字符串元组。
                  否则，它应该是一个字符串。
      sequence_length: （可选）大小为[batch_size]的int32/int64向量。
        有关sequence_length的更多细节，请参阅rnn()的文档。
      scope: 创建的子图的VariableScope；默认为"rnn"。

    Returns:
      一个（outputs, state）元组，其中：
        outputs是长度为T的输出列表（每个输入一个）
        state是最终状态

    Raises:
      TypeError: 如果cell不是RNNCell的实例。
      ValueError: 如果inputs为None或空列表，或者state_name的元数和类型与cell.state_size不匹配。
    """
    state_size = cell.state_size
    state_is_tuple = tf.nest.is_nested(state_size)
    state_name_tuple = tf.nest.is_nested(state_name)

    if state_is_tuple != state_name_tuple:
        raise ValueError("state_name的类型应该与cell.state_size相同。"
                         "state_name: %s, cell.state_size: %s" % (str(state_name),
                                                                  str(state_size)))

    if state_is_tuple:
        state_name_flat = tf.nest.flatten(state_name)
        state_size_flat = tf.nest.flatten(state_size)

        if len(state_name_flat) != len(state_size_flat):
            raise ValueError("#elems(state_name) != #elems(state_size): %d vs. %d" %
                             (len(state_name_flat), len(state_size_flat)))

        initial_state = tf.nest.pack_sequence_as(
            structure=state_size,
            flat_sequence=[state_saver.state(s) for s in state_name_flat])
    else:
        initial_state = state_saver.state(state_name)

    (outputs, state) = static_rnn(
        cell,
        inputs,
        initial_state=initial_state,
        sequence_length=sequence_length,
        scope=scope)

    if state_is_tuple:
        flat_state = tf.nest.flatten(state)
        state_name = tf.nest.flatten(state_name)
        save_state = [
            state_saver.save_state(name, substate)
            for name, substate in zip(state_name, flat_state)
        ]
    else:
        save_state = [state_saver.save_state(state_name, state)]

    with tf.control_dependencies(save_state):
        last_output = outputs[-1]
        flat_last_output = tf.nest.flatten(last_output)
        flat_last_output = [
            tf.identity(output) for output in flat_last_output
        ]
        outputs[-1] = tf.nest.pack_sequence_as(
            structure=last_output, flat_sequence=flat_last_output)

    return (outputs, state)

def static_bidirectional_rnn(cell_fw,
                             cell_bw,
                             inputs,
                             initial_state_fw=None,
                             initial_state_bw=None,
                             dtype=None,
                             sequence_length=None,
                             scope=None):

    if not _like_rnncell(cell_fw):
        raise TypeError("cell_fw必须是RNNCell的实例")
    if not _like_rnncell(cell_bw):
        raise TypeError("cell_bw必须是RNNCell的实例")
    if not tf.nest.is_nested(inputs):
        raise TypeError("inputs必须是一个序列")
    if not inputs:
        raise ValueError("inputs不能为空")

    with tf.variable_scope(scope or "bidirectional_rnn"):
        # 前向方向
        with tf.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = static_rnn(
                cell_fw,
                inputs,
                initial_state_fw,
                dtype,
                sequence_length,
                scope=fw_scope)

        # 后向方向
        with tf.variable_scope("bw") as bw_scope:
            reversed_inputs = _reverse_seq(inputs, sequence_length)
            tmp, output_state_bw = static_rnn(
                cell_bw,
                reversed_inputs,
                initial_state_bw,
                dtype,
                sequence_length,
                scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)
    # 将每个前向/后向输出连接
    flat_output_fw = tf.nest.flatten(output_fw)
    flat_output_bw = tf.nest.flatten(output_bw)

    flat_outputs = tuple(
        tf.concat([fw, bw], 1)
        for fw, bw in zip(flat_output_fw, flat_output_bw))

    outputs = tf.nest.pack_sequence_as(
        structure=output_fw, flat_sequence=flat_outputs)

    return (outputs, output_state_fw, output_state_bw)