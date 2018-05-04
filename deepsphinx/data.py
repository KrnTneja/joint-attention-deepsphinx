'''Data utilities'''
import threading
import random
import numpy as np
from python_speech_features.base import fbank, delta
import tensorflow as tf
from deepsphinx.vocab import VOCAB_TO_INT, VOCAB_TO_INT_INF
from deepsphinx.utils import FLAGS
from deepsphinx.fst import in_fst
import soundfile as sf
import csv

def get_features(audio_file):
    '''Get features from a file'''
    signal, sample_rate = sf.read(tf.gfile.FastGFile(audio_file, 'rb'))
    feat, energy = fbank(signal, sample_rate, nfilt=FLAGS.nfilt)
    feat = np.log(feat)
    dfeat = delta(feat, 2)
    ddfeat = delta(dfeat, 2)
    return np.concatenate([feat, dfeat, ddfeat, np.expand_dims(energy, 1)],
                          axis=1)

def get_speaker_stats(set_ids):
    '''Get mean and variance of a speaker'''
    tf.logging.info('Getting speaker stats')
    trans = tf.gfile.FastGFile(FLAGS.trans_file).readlines()
    sum_speaker = {}
    sum_sq_speaker = {}
    count_speaker = {}
    for _, set_id, speaker, audio_file in csv.reader(trans):
        if set_id in set_ids:
            n_feat = 3 * FLAGS.nfilt + 1
            if speaker not in sum_speaker:
                sum_speaker[speaker] = np.zeros(n_feat)
                sum_sq_speaker[speaker] = np.zeros(n_feat)
                count_speaker[speaker] = 0
            feat = get_features(audio_file)
            sum_speaker[speaker] += np.mean(feat, 0)
            sum_sq_speaker[speaker] += np.mean(np.square(feat), 0)
            count_speaker[speaker] += 1
    mean = {k: sum_speaker[k] / count_speaker[k] for k, v in sum_speaker.items()}
    var = {k: sum_sq_speaker[k] / count_speaker[k] -
              np.square(mean[k]) for k, v in sum_speaker.items()}
    return mean, var


def read_data_queue(
        set_id,
        queue,
        sess,
        mean_speaker,
        var_speaker,
        fst):
    '''Start a thread to add data in a queue'''
    if FLAGS.inflection_task:
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, 26+9])
    else:
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.nfilt * 3 + 1])
    input_length = tf.placeholder(dtype=tf.int32, shape=[])
    output_data = tf.placeholder(dtype=tf.int32, shape=[None])
    output_length = tf.placeholder(dtype=tf.int32, shape=[])
    enqueue_op = queue.enqueue(
        [input_data, input_length, output_data, output_length])
    close_op = queue.close()

    thread = threading.Thread(
        target=read_data_thread,
        args=(
            set_id,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op,
            close_op,
            mean_speaker,
            var_speaker,
            fst))
    thread.daemon = True  # Thread will close when parent quits.
    thread.start()


def read_data_thread(
        set_id,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op,
        close_op,
        mean_speaker,
        var_speaker,
        fst):
    '''Enqueue data to queue'''
    if FLAGS.inflection_task:
        trans = tf.gfile.FastGFile(FLAGS.trans_file).readlines()
        random.shuffle(trans)
        # print("csv.reader(trans, delimiter=',')", csv.reader(trans))
        for input_seq, output_seq, inflection in csv.reader(trans):
            # tf.logging.info("Adding SAMPLE: {} {} {}".format(input_seq, output_seq, inflection))
            # print("input_seq", input_seq, "output_seq", output_seq, "inflection", inflection)
            text = [VOCAB_TO_INT_INF[c]
                    for c in output_seq.split()]
            input_text = np.array([VOCAB_TO_INT_INF[c]
                    for c in input_seq.split()])
            feat = np.zeros((len(input_text), 26+9), dtype=np.float32)
            feat[np.arange(len(input_text)), input_text] = 1
            # tf.logging.info("feat", feat, len(feat), text, len(text))
            sess.run(enqueue_op, feed_dict={
                input_data: feat,
                input_length: len(input_text),
                output_data: text,
                output_length: len(text)})
        sess.run(close_op) 
    else:
        trans = tf.gfile.FastGFile(FLAGS.trans_file).readlines()
        # random.shuffle(trans)
        for text, set_id_trans, speaker, audio_file in csv.reader(trans):
            # tf.logging.info("Adding SAMPLE: {} {} {} {}".format(text, set_id_trans, speaker, audio_file))
            text = [VOCAB_TO_INT[c]
                    for c in text.split()] + [VOCAB_TO_INT['</s>']]
            if (set_id == set_id_trans and
                    ((not FLAGS.use_train_lm) or in_fst(fst, text))):
                feat = get_features(audio_file)
                feat = feat - mean_speaker[speaker]
                feat = feat / np.sqrt(var_speaker[speaker])
                # tf.logging.info("input_data", feat, feat.shape[0], text, len(text))
                sess.run(enqueue_op, feed_dict={
                    input_data: feat,
                    input_length: feat.shape[0],
                    output_data: text,
                    output_length: len(text)})
                # tf.logging.info("Added SAMPLE: {} {} {} {}".format(text, set_id_trans, speaker, audio_file))
                # tf.logging.info("{}".format(feat))
        sess.run(close_op)
