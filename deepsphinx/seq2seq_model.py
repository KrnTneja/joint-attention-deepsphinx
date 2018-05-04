'''Tensorflow model for speech recognition'''
import tensorflow as tf
from deepsphinx.vocab import VOCAB_SIZE, VOCAB_TO_INT, VOCAB_SIZE_INF, VOCAB_TO_INT_INF
from deepsphinx.utils import FLAGS
from deepsphinx.lm import LMCellWrapper
from deepsphinx.attention import BahdanauAttentionCutoff
from deepsphinx.joint_wrapper import JointWrapper



def encoding_layer(
        input_lengths,
        rnn_inputs,
        keep_prob):
    ''' Encoding layer for the model.

    Args:
        input_lengths (Tensor): A tensor of input lenghts of instances in
            batches
        rnn_inputs (Tensor): Inputs

    Returns:
        Encoding output, LSTM state, output length
    '''
    for layer in range(FLAGS.num_conv_layers):
        # print("rnn_inputs.get_shape()", rnn_inputs.get_shape())
        filter = tf.get_variable(
            "conv_filter{}".format(layer + 1),
            shape=[FLAGS.conv_layer_width, rnn_inputs.get_shape()[2], FLAGS.conv_layer_size])
        rnn_inputs = tf.nn.conv1d(rnn_inputs, filter, 1, 'SAME')
    for layer in range(FLAGS.num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(
                FLAGS.rnn_size,
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw,
                output_keep_prob=keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=rnn_inputs.get_shape()[2])

            cell_bw = tf.contrib.rnn.LSTMCell(
                FLAGS.rnn_size,
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw,
                output_keep_prob=keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=rnn_inputs.get_shape()[2])

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                rnn_inputs,
                input_lengths,
                dtype=tf.float32)

            if layer != FLAGS.num_layers - 1:
                rnn_inputs = tf.concat(enc_output,2)
                rnn_inputs = rnn_inputs[:, ::2, :]
                input_lengths = (input_lengths + 1) // 2
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state, input_lengths
    
def get_dec_cell(
        enc_output,
        enc_output_lengths,
        use_lm,
        fst,
        tile_size,
        keep_prob):
    '''Decoding cell for attention based model

    Return:
        `RNNCell` Instance
    '''

    lstm = tf.contrib.rnn.LSTMCell(
        FLAGS.rnn_size,
        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        # name='input_lstm')

    dec_cell_inp = tf.contrib.rnn.DropoutWrapper(
        lstm,
        output_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32)
        # name='input_lstm_dropout_wrapper')
    
    lstm = tf.contrib.rnn.LSTMCell(
        FLAGS.rnn_size,
        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        # name='middle_lstm')
    dec_cell = tf.contrib.rnn.DropoutWrapper(
        lstm,
        output_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32)
        # name='middle_lstm_dropout_wrapper')

    if FLAGS.inflection_task:
        dec_cell_out = tf.contrib.rnn.LSTMCell(
            FLAGS.rnn_size,
            num_proj=VOCAB_SIZE_INF,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            # name='output_lstm')
    else:
        dec_cell_out = tf.contrib.rnn.LSTMCell(
            FLAGS.rnn_size,
            num_proj=VOCAB_SIZE,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            # name='output_lstm')

    dec_cell = tf.contrib.rnn.MultiRNNCell(
        [dec_cell_inp] +
        [dec_cell] * (FLAGS.num_decoding_layers - 2) +
        [dec_cell_out])
        # name='decoder_MultiRNNCell')

    enc_output = tf.contrib.seq2seq.tile_batch(
        enc_output,
        tile_size,
        name='enc_output_tile')

    enc_output_lengths = tf.contrib.seq2seq.tile_batch(
        enc_output_lengths,
        tile_size,
        name='enc_output_lengths_tile')

    attn_mech = BahdanauAttentionCutoff(
        FLAGS.rnn_size,
        enc_output,
        enc_output_lengths,
        normalize=True,
        name='BahdanauAttentionCuttOff')

    if FLAGS.inflection_task:
        if FLAGS.use_joint_prob:
            dec_cell = JointWrapper(
                cell=dec_cell,
                attention_mechanism=attn_mech,
                max_segment_length=FLAGS.max_segment_length,
                attention_layer_size=VOCAB_SIZE_INF,
                output_attention=True,
                name='JointWrapperCustom')
        else: 
            dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=dec_cell,
                attention_mechanism=attn_mech,
                attention_layer_size=VOCAB_SIZE_INF,
                output_attention=True,
                name='AttentionWrapper')
    else:
        if FLAGS.use_joint_prob:
            dec_cell = JointWrapper(
                cell=dec_cell,
                attention_mechanism=attn_mech,
                max_segment_length=FLAGS.max_segment_length,
                attention_layer_size=VOCAB_SIZE,
                output_attention=True,
                name='JointWrapperCustom')
        else: 
            dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=dec_cell,
                attention_mechanism=attn_mech,
                attention_layer_size=VOCAB_SIZE,
                output_attention=True,
                name='AttentionWrapper')

    if use_lm:
        dec_cell = LMCellWrapper(dec_cell, fst, 5)

    return dec_cell


#pylint: disable-msg=too-many-arguments
def training_decoding_layer(
        target_data,
        target_lengths,
        enc_output,
        enc_state,
        enc_output_lengths,
        fst,
        keep_prob):
    ''' Training decoding layer for the model.

    Returns:
        Training logits
    '''
    if FLAGS.inflection_task:
        target_data = tf.concat(
            [tf.fill([FLAGS.batch_size, 1], VOCAB_TO_INT_INF['<s>']),
             target_data[:, :-1]], 1, name='concat_s_tags')   # NOTE: Replacing </s> with <s> - Karan
    else:
        target_data = tf.concat(
            [tf.fill([FLAGS.batch_size, 1], VOCAB_TO_INT['<s>']),
             target_data[:, :-1]], 1, name='concat_s_tags')   # NOTE: Replacing </s> with <s> - Karan

    dec_cell = get_dec_cell(
        enc_output,
        enc_output_lengths,
        FLAGS.use_train_lm,
        fst,
        1,
        keep_prob)
    
    initial_state = dec_cell.zero_state(
        dtype=tf.float32,
        batch_size=FLAGS.batch_size)
    
    if FLAGS.inflection_task:
        target_data = tf.nn.embedding_lookup(
            tf.eye(VOCAB_SIZE_INF),
            target_data, 
            name='target_data_embedding_lookup')
    else:
        target_data = tf.nn.embedding_lookup(
            tf.eye(VOCAB_SIZE),
            target_data, 
            name='target_data_embedding_lookup')

    training_helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=target_data,   # Decides what will be given as next_inputs - Karan 
        sequence_length=target_lengths,
        time_major=False,
        name='TrainingHelper')

    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=dec_cell,
        helper=training_helper,
        initial_state=initial_state) 
        # name='TrainingBasicDecoder')

    training_logits, training_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
        training_decoder,
        output_time_major=False,
        impute_finished=True)
        # name="training_dynamic_decode")

    # training_logits = tf.contrib.framework.nest.map_structure(lambda x: tf.Print(x, [x], message="Training Logits", summarize=1000), training_logits)

    return training_logits, training_final_state

def inference_decoding_layer(
        enc_output,
        enc_state,
        enc_output_lengths,
        fst,
        keep_prob):
    ''' Inference decoding layer for the model.

    Returns:
        Predictions
    '''

    if FLAGS.inflection_task:
        start_tokens = tf.fill(
            [FLAGS.batch_size],
            VOCAB_TO_INT_INF['<s>'],
            name='start_tokens')
    else:
        start_tokens = tf.fill(
            [FLAGS.batch_size],
            VOCAB_TO_INT['<s>'],
            name='start_tokens')

    if FLAGS.use_joint_prob:
        dec_cell = get_dec_cell(
            enc_output,
            enc_output_lengths,
            FLAGS.use_inference_lm,
            fst,
            1,  # FOR GreedyEmbeddingHelper- Karan
            keep_prob)
        initial_state = dec_cell.zero_state(
            dtype=tf.float32, 
            batch_size=FLAGS.batch_size)  # FOR GreedyEmbeddingHelper
        if FLAGS.inflection_task:
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=tf.eye(VOCAB_SIZE_INF),
                start_tokens=start_tokens,
                end_token=VOCAB_TO_INT_INF['</s>'])
                # name='InferenceGreedyEmbeddingHelper')
        else:
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=tf.eye(VOCAB_SIZE),
                start_tokens=start_tokens,
                end_token=VOCAB_TO_INT['</s>'])
                # name='InferenceGreedyEmbeddingHelper')
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=dec_cell,
            helper=inference_helper,
            initial_state=initial_state)
            # name='InferenceBasicDecoder')
    else:
        dec_cell = get_dec_cell(
            enc_output,
            enc_output_lengths,
            FLAGS.use_inference_lm,
            fst,
            FLAGS.beam_width, # FOR BeamSearchDecoder
            keep_prob)
        initial_state = dec_cell.zero_state(
            dtype=tf.float32, 
            batch_size=FLAGS.batch_size*FLAGS.beam_width) # FOR BeamSearchDecoder 
        if FLAGS.inflection_task:
            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=dec_cell,
                embedding=tf.eye(VOCAB_SIZE_INF),
                start_tokens=start_tokens,
                end_token=VOCAB_TO_INT_INF['</s>'],
                initial_state=initial_state,
                beam_width=FLAGS.beam_width)
        else:
            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=dec_cell,
                embedding=tf.eye(VOCAB_SIZE),
                start_tokens=start_tokens,
                end_token=VOCAB_TO_INT['</s>'],
                initial_state=initial_state,
                beam_width=FLAGS.beam_width)

    predictions, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder,
        output_time_major=False,
        maximum_iterations=FLAGS.max_output_len)
        # name='inference_dynamic_decode')

    return predictions

def seq2seq_model(
        input_data,
        target_data,
        input_lengths,
        target_lengths,
        fst,
        keep_prob):
    ''' Attention based model

    Returns:
        Logits, Predictions, Training operation, Cost, Step, Scores of beam
        search
    '''

    enc_output, enc_state, enc_lengths = encoding_layer(
        input_lengths,
        input_data,
        keep_prob)

    with tf.variable_scope('decode'):
        training_logits, training_final_state = training_decoding_layer(
            target_data,
            target_lengths,
            enc_output,
            enc_state,
            enc_lengths,
            fst,
            keep_prob)
    with tf.variable_scope('decode', reuse=True):
        predictions = inference_decoding_layer(
            enc_output,
            enc_state,
            enc_lengths,
            fst,
            keep_prob)

    # Create tensors for the training logits and predictions
    training_logits = tf.identity(
        training_logits.rnn_output,
        name='logits')

    if FLAGS.use_joint_prob:
        predictions = tf.identity(
            predictions.sample_id,
            name='predictions')
    else:
        scores = tf.identity(
            predictions.beam_search_decoder_output.scores,
            name='scores')
        predictions = tf.identity(
            predictions.predicted_ids,
            name='predictions')
        # Create the weights for sequence_loss
        masks = tf.sequence_mask(
            target_lengths,
            tf.reduce_max(target_lengths),
            dtype=tf.float32,
            name='masks')
        
    with tf.name_scope('optimization'):
        # Loss function
        if FLAGS.use_joint_prob:
            beam_alignments_length = tf.shape(training_final_state.beam_alignments)[1]
            cost_indices = tf.concat([tf.expand_dims(target_lengths-1, 1), 
                tf.expand_dims(tf.range(FLAGS.batch_size), 1), 
                tf.expand_dims(beam_alignments_length*tf.ones(FLAGS.batch_size, dtype=tf.int32), 1)-1], axis=1)
            beam_probs_history = training_final_state.beam_probs_history.stack(name="Beam_Probs_History_Stack")
            # beam_probs_history = tf.Print(beam_probs_history, [tf.shape(beam_probs_history)], message="BEAM PROBS HISTORY:", summarize=1000)
            # cost_indices = tf.Print(cost_indices, [cost_indices], message="COST INDICES:", summarize=1000)
            samples_log_probs = tf.gather_nd(beam_probs_history,
                cost_indices)
            # beam_probs_history = tf.Print(samples_log_probs, [samples_log_probs], message="LOG PROBS:", summarize=1000)
            cost = -1*tf.reduce_sum(samples_log_probs)
        else:
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                target_data,
                masks)

        tf.summary.scalar('cost', cost)

        step = tf.train.get_or_create_global_step()

        # Optimizer
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [
            (tf.clip_by_value(grad, -5., 5.), var)
            for grad, var in gradients if grad is not None]
        # capped_gradients = [
            # (tf.Print(grad, [tf.reduce_min(grad), tf.reduce_max(grad)]), var)
            # for grad, var in capped_gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, step)

    if FLAGS.use_joint_prob:
        return training_logits, predictions, train_op, cost, step, target_lengths, beam_probs_history, beam_alignments_length
    else:
        return training_logits, predictions, train_op, cost, step

