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

"""Run a full test run for one or more algorithmic tasks from CLRS."""

import os
import re
import shutil
import sys
import time
from absl import app
from absl import flags
from absl import logging

import clrs
import jax
import jax.numpy as jnp
import requests
import tensorflow as tf

from clrs._src import processors

flags.DEFINE_string('algorithm', '', 'Which algorithm to run.')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 100,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_items', 320000,
                     'Number of items (i.e., individual examples, possibly '
                     'repeated) processed during training. With non-chunked'
                     'training, this is the number of training batches times '
                     'the number of training steps. For chunked training, '
                     'as many chunks will be processed as needed to get these '
                     'many full examples.')
flags.DEFINE_integer('eval_every', 320,
                     'Logging frequency (in training examples).')
flags.DEFINE_boolean('verbose_logging', False, 'Whether to log aux losses.')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden size units of the model.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing_noise', 0.5,
                   'Probability that rematerialized hints are encoded during '
                   'training instead of ground-truth teacher hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')

flags.DEFINE_enum('hint_mode', 'encoded_decoded_nodiff',
                  ['encoded_decoded', 'decoded_only',
                   'encoded_decoded_nodiff', 'decoded_only_nodiff',
                   'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`). The `no_diff` variants '
                  'try to predict all hint values instead of just the values '
                  'that change from one timestep to the next.')

flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_string('use_memory', "all",
                    'Whether to insert memory after message passing.')
flags.DEFINE_enum(
    'processor_type', 'gatv2',
    ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
     'gat', 'gatv2', 'gat_full', 'gatv2_full',
     'memnet_full', 'memnet_masked'],
    'The processor type to use.')

flags.DEFINE_string('checkpoint_path', '/tmp/CLRS30',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')
flags.DEFINE_integer('memory_size', 20,
                     'Size of differentiable data structure memory.')
flags.DEFINE_integer('skip_to_step', 0,
                     'Will read model from a pickle file and skip first n steps if non-zero')
flags.DEFINE_boolean('load_from_last', False,
                     'If true, skip_to_step will load the .pkl file ending in _last instead of _best')
flags.DEFINE_boolean('algo_list_reverse',False,
                     'Whether or nto to reverse the list of algorithms we test (for parallel runs)')
FLAGS = flags.FLAGS


def unpack(v):
    try:
        return v.item()  # DeviceArray
    except (AttributeError, ValueError):
        return v


def evaluate(rng_key, model, feedback, spec, extras=None, verbose=False):
    """Evaluates a model on feedback."""
    out = {}
    predictions, aux = model.predict(rng_key, feedback.features)
    out.update(clrs.evaluate(feedback.outputs, predictions))
    if model.decode_hints and verbose:
        hint_preds = [clrs.decoders.postprocess(spec, x) for x in aux[0]]
        out.update(clrs.evaluate_hints(feedback.features.hints,
                                       feedback.features.lengths,
                                       hint_preds))
    if extras:
        out.update(extras)
    if verbose:
        out.update(model.verbose_loss(feedback, aux))
    return {k: unpack(v) for k, v in out.items()}


def evaluate_preds(preds, outputs, hints, lengths, hint_preds, spec, extras):
    """Evaluates predictions against feedback."""
    out = {}
    out.update(clrs.evaluate(outputs, preds))
    if hint_preds:
        hint_preds = [clrs.decoders.postprocess(spec, x) for x in hint_preds]
        out.update(clrs.evaluate_hints(hints, lengths, hint_preds))
    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}


def _concat(dps, axis):
    return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, spec, extras):
    """Collect batches of output and hint preds and evaluate them."""
    verbose = FLAGS.verbose_logging
    processed_samples = 0
    preds = []
    hint_preds = []
    outputs = []
    hints = []
    lengths = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        outputs.append(feedback.outputs)
        rng_key, new_rng_key = jax.random.split(rng_key)
        cur_preds, (cur_hint_preds, _, _) = predict_fn(rng_key, feedback.features)
        preds.append(cur_preds)
        if verbose:
            hints.append(feedback.features.hints)
            lengths.append(feedback.features.lengths)
            hint_preds.append(cur_hint_preds)
        rng_key = new_rng_key
        processed_samples += FLAGS.batch_size
    outputs = _concat(outputs, axis=0)
    preds = _concat(preds, axis=0)
    if verbose:
        # for hints, axis=1 because hints have time dimension first
        hints = _concat(hints, axis=1)
        lengths = _concat(lengths, axis=0)
        # for hint_preds, axis=0 because the time dim is unrolled as a list
        hint_preds = _concat(hint_preds, axis=0)

    return evaluate_preds(preds, outputs, hints, lengths, hint_preds, spec,
                          extras)


def maybe_download_dataset():
    """Downloads CLRS30 dataset if not already downloaded."""
    dataset_folder = os.path.join(FLAGS.dataset_path, clrs.get_clrs_folder())
    if os.path.isdir(dataset_folder):
        logging.info('Dataset found at %s. Skipping download.', dataset_folder)
        return dataset_folder
    logging.info('Dataset not found in %s. Downloading...', dataset_folder)
    clrs_url = clrs.get_dataset_gcp_url()
    request = requests.get(clrs_url, allow_redirects=True)
    clrs_file = os.path.join(FLAGS.dataset_path, os.path.basename(clrs_url))
    os.makedirs(dataset_folder)
    open(clrs_file, 'wb').write(request.content)
    shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
    os.remove(clrs_file)
    return dataset_folder


def main():
    # Use canonical CLRS-30 samplers.
    clrs30_spec = clrs.CLRS30
    logging.info('Using CLRS30 spec: %s', clrs30_spec)
    dataset_folder = maybe_download_dataset()

    if FLAGS.hint_mode == 'encoded_decoded_nodiff':
        encode_hints = True
        decode_hints = True
        decode_diffs = False
    elif FLAGS.hint_mode == 'decoded_only_nodiff':
        encode_hints = False
        decode_hints = True
        decode_diffs = False
    elif FLAGS.hint_mode == 'encoded_decoded':
        encode_hints = True
        decode_hints = True
        decode_diffs = True
    elif FLAGS.hint_mode == 'decoded_only':
        encode_hints = False
        decode_hints = True
        decode_diffs = True
    elif FLAGS.hint_mode == 'none':
        encode_hints = False
        decode_hints = False
        decode_diffs = False
    else:
        raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

    common_args = dict(folder=dataset_folder,
                       algorithm=FLAGS.algorithm,
                       batch_size=FLAGS.batch_size)
    # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):
        if FLAGS.chunked_training:
            train_sampler, spec = clrs.create_chunked_dataset(
                **common_args, split='train', chunk_length=FLAGS.chunk_length)
            train_sampler_for_eval, _, _ = clrs.create_dataset(
                split='train', **common_args)
            train_sampler_for_eval = train_sampler_for_eval.as_numpy_iterator()
        else:
            train_sampler, _, spec = clrs.create_dataset(**common_args, split='train')
            train_sampler = train_sampler.as_numpy_iterator()
            train_sampler_for_eval = None

        val_sampler, val_samples, _ = clrs.create_dataset(
            **common_args, split='val')
        val_sampler = val_sampler.as_numpy_iterator()
        test_sampler, test_samples, _ = clrs.create_dataset(
            **common_args, split='test')
        test_sampler = test_sampler.as_numpy_iterator()

    processor_factory = processors.get_processor_factory(FLAGS.processor_type,
                                                         use_ln=FLAGS.use_ln,
                                                         nb_heads=FLAGS.nb_heads)
    model_params = dict(
        processor_factory=processor_factory,
        hidden_dim=FLAGS.hidden_size,
        encode_hints=encode_hints,
        decode_hints=decode_hints,
        decode_diffs=decode_diffs,
        use_memory=FLAGS.use_memory,
        memory_size=FLAGS.memory_size,
        learning_rate=FLAGS.learning_rate,
        checkpoint_path=FLAGS.checkpoint_path,
        freeze_processor=FLAGS.freeze_processor,
        dropout_prob=FLAGS.dropout_prob,
        hint_teacher_forcing_noise=FLAGS.hint_teacher_forcing_noise,
    )

    eval_model = clrs.models.BaselineModel(
        spec=spec,
        dummy_trajectory=next(val_sampler),
        **model_params
    )
    if FLAGS.chunked_training:
        train_model = clrs.models.BaselineModelChunked(
            spec=spec,
            dummy_trajectory=next(train_sampler),
            **model_params
        )
    else:
        train_model = eval_model

    # Training loop.
    best_score = -1.0  # Ensure that there is overwriting
    rng_key = jax.random.PRNGKey(FLAGS.seed)
    current_train_items = 0
    step = 0
    next_eval = 0
    this_is_first_time_we_see_this_score = True
    target_step = FLAGS.skip_to_step


    while current_train_items < FLAGS.train_items:
        currently_skipping = (target_step != 0) and (step <= target_step)


        feedback = next(train_sampler)

        # Initialize model.
        if current_train_items == 0:
            t = time.time()
            train_model.init(feedback.features, FLAGS.seed + 1)

        # Training step step.
        rng_key, new_rng_key = jax.random.split(rng_key)
        if not currently_skipping:
            cur_loss = train_model.feedback(rng_key, feedback)
        rng_key = new_rng_key
        if current_train_items == 0:
            logging.info('Compiled feedback step in %f s.', time.time() - t)
        if FLAGS.chunked_training:
            examples_in_chunk = jnp.sum(feedback.features.is_last)
        else:
            examples_in_chunk = len(feedback.features.lengths)
        current_train_items += examples_in_chunk





        # Periodically evaluate model.
        if current_train_items >= next_eval:
            common_extras = {'examples_seen': current_train_items,
                             'step': step}
            eval_model.params = train_model.params
            # Training info.
            if FLAGS.chunked_training:
                train_feedback = next(train_sampler_for_eval)
            else:
                train_feedback = feedback
            rng_key, new_rng_key = jax.random.split(rng_key)
            if not currently_skipping:
                train_stats = evaluate(
                    rng_key,
                    eval_model,
                    train_feedback,
                    spec=spec,
                    extras=dict(loss=cur_loss, **common_extras),
                    verbose=FLAGS.verbose_logging,
                )
                logging.info('(train) step %d: %s', step, train_stats)

            rng_key = new_rng_key

            # Validation info.
            rng_key, new_rng_key = jax.random.split(rng_key)
            if not currently_skipping:
                val_stats = collect_and_eval(
                    val_sampler,
                    eval_model.predict,
                    val_samples,
                    rng_key,
                    spec=spec,
                    extras=common_extras)
                logging.info('(val) step %d: %s', step, val_stats)
                # with open(f"logs_{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}.txt", "a+") as myfile:
                #     myfile.write(f"(val) step {step}: {val_stats}\n")

                # If best scores, update checkpoint.
                score = val_stats['score']
                if (score > best_score):
                    logging.info('Saving new checkpoint...')
                    best_score = score
                    train_model.save_model(
                        f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_best.pkl")
                    this_is_first_time_we_see_this_score = True
                    if score==1.0:
                        test_stats = collect_and_eval(
                            test_sampler,
                            eval_model.predict,
                            test_samples,
                            rng_key,
                            spec=spec,
                            extras=common_extras)
                        # rng_key = new_rng_key
                        logging.info('(test first best) step %d: %s', step, test_stats)
                        with open("results.txt", "a+") as myfile:
                            myfile.write(
                                f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_best.pkl: (test) step {step}: {test_stats}\n")

                elif score == best_score:
                    logging.info('Saving new checkpoint (same score)...')
                    train_model.save_model(
                        f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_last.pkl")
                    this_is_first_time_we_see_this_score = False
            else:
                if step == target_step:
                    if FLAGS.load_from_last:
                        restore_string = "last"
                        this_is_first_time_we_see_this_score = False
                    else:
                        restore_string= "best"
                        this_is_first_time_we_see_this_score = True
                    train_model.restore_model(
                        f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_{restore_string}.pkl",
                        only_load_processor=False)
                    eval_model.params = train_model.params
                    val_stats = collect_and_eval(
                        val_sampler,
                        eval_model.predict,
                        val_samples,
                        rng_key,
                        spec=spec,
                        extras=common_extras)
                    logging.info(f"Restored model at step {target_step}")
                    logging.info('(val) step %d: %s', step, val_stats)
                    score = val_stats['score']
                    best_score=score
                    if score==1.0 and this_is_first_time_we_see_this_score:
                        test_stats = collect_and_eval(
                            test_sampler,
                            eval_model.predict,
                            test_samples,
                            rng_key,
                            spec=spec,
                            extras=common_extras)
                        # rng_key = new_rng_key
                        logging.info('(test first best) step %d: %s', step, test_stats)
                        with open("results.txt", "a+") as myfile:
                            myfile.write(
                                f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_best.pkl: (test) step {step}: {test_stats}\n")


            rng_key = new_rng_key

            next_eval += FLAGS.eval_every
        step += 1

    # Training complete, evaluate on test set.
    logging.info('Restoring best model from checkpoint...')
    eval_model.restore_model(f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_best.pkl", only_load_processor=False)

    rng_key, new_rng_key = jax.random.split(rng_key)
    test_stats = collect_and_eval(
        test_sampler,
        eval_model.predict,
        test_samples,
        rng_key,
        spec=spec,
        extras=common_extras)
    # rng_key = new_rng_key
    logging.info('(test first best) step %d: %s', step, test_stats)

    with open("results.txt", "a+") as myfile:
        myfile.write(
            f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_best.pkl: (test) step {step}: {test_stats}\n")

    if not this_is_first_time_we_see_this_score:
        eval_model.restore_model(
            f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_last.pkl",
            only_load_processor=False)

        test_stats = collect_and_eval(
            test_sampler,
            eval_model.predict,
            test_samples,
            rng_key,
            spec=spec,
            extras=common_extras)
        # rng_key = new_rng_key
        logging.info('(test last best) step %d: %s', step, test_stats)

        with open("results.txt", "a+") as myfile:
            myfile.write(
                f"{FLAGS.algorithm}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_last.pkl: (test) step {step}: {test_stats}\n")



def main_wrapper(unused_argv):
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    if FLAGS.algo_list_reverse:
        reverse_func = reversed
    else:
        reverse_func = lambda x: x
    GAT_BEST = reverse_func([
        'dfs',
        'jarvis_march',
        'kmp_matcher',
        'lcs_length',
        'quickselect',
        'task_scheduling'
    ])
    # MPNN algos
    MPNN_BEST = reverse_func([
        'articulation_points',
        'activity_selector',
        'bfs',
        'bridges',
        'dijkstra',
        'graham_scan',
        'mst_kruskal',
        'mst_prim',
        'naive_string_matcher',
        'segments_intersect',
        'strongly_connected_components',
    ])
    PGN_best = reverse_func([
        'activity_selector',
        'bellman_ford',
        'binary_search',
        'dag_shortest_paths',
        'find_maximum_subarray_kadane',
        'floyd_warshall',
        'matrix_chain_order',
        'minimum',
        'mst_prim',
        'optimal_bst',
        'quickselect',
        'strongly_connected_components',
        'task_scheduling',
        'topological_sort',
    ])

    # memory_type = "NTM"
    # model="gatv2"
    # memory_size=20

    if FLAGS.use_memory=="all":
        memories = ["NTM","DeQue","DNC"]
    else:
        memories = [FLAGS.use_memory]

    # FLAGS.memory_size = 100
    # for model in ["gatv2", "mpnn"]:
    if FLAGS.algorithm == "":
        if FLAGS.processor_type == "gatv2" or FLAGS.processor_type == "gat":
            algo_list = GAT_BEST
        elif FLAGS.processor_type == "mpnn":
            algo_list = MPNN_BEST
        else:
            algo_list = PGN_best
    else:
        algo_list = [FLAGS.algorithm]

    for memory in memories:
        FLAGS.use_memory = memory
        for algo in algo_list:
            FLAGS.algorithm = algo

            FLAGS.skip_to_step = 0
            FLAGS.load_from_last = False

            with open("results.txt") as myfile:
                txt = myfile.read()
            print("\n\n")
            print(txt)
            print("\n\n")
            if (not (f"{algo}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}" in txt)) and (not (
                    f"{algo}_best_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}" in txt)):

                print("Checking if pkl file exists...")
                pkl_string = f"{algo}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_best.pkl"

                file_exists = os.path.exists(pkl_string)
                if file_exists:
                    print("pkl file exists!")
                    log_files = sorted([filename for filename in os.listdir('.') if filename.startswith(f"logs_{algo}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}")])
                    log_filename=log_files[-1]
                    print(f"Reading {log_filename}")
                    with open(log_filename, 'r') as f:
                        contents = f.readlines()
                    targets = [line_index for (line_index,line) in enumerate(contents) if "Saving new checkpoint..." in line]
                    final_target = targets[-1]
                    line_before_best_target = final_target-1
                    relevant_line = contents[line_before_best_target]
                    print(relevant_line)

                    p = re.compile("step ([0-9]*)")
                    resulting_step = p.findall(relevant_line)[0]

                    print(f"Skipping to step {resulting_step}")
                    FLAGS.skip_to_step=int(resulting_step)

                    targets_same_score = [line_index for (line_index, line) in enumerate(contents) if
                               "Saving new checkpoint (same score)..." in line]
                    pkl_string_last = f"{algo}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}_last.pkl"

                    file_exists_last = os.path.exists(pkl_string_last)
                    if len(targets_same_score)>0 and file_exists_last:
                        last_target = targets_same_score[-1]
                        line_before_last_target = last_target-1
                        if line_before_last_target>line_before_best_target:
                            relevant_line = contents[line_before_last_target]
                            print(relevant_line)

                            p = re.compile("step ([0-9]*)")
                            resulting_step = p.findall(relevant_line)[0]

                            print(f"Skipping to step {resulting_step} instead, as this had the same val score")
                            FLAGS.skip_to_step = int(resulting_step)
                            FLAGS.load_from_last=True

                print(
                    f"running with specs: {algo}, {FLAGS.use_memory}, {FLAGS.processor_type}, {FLAGS.memory_size}")
                logging.get_absl_handler().use_absl_log_file(
                    f"logs_{algo}_{FLAGS.processor_type}_{FLAGS.use_memory}_{FLAGS.memory_size}.txt", "./")
                main()

if __name__ == '__main__':
    app.run(main_wrapper)
