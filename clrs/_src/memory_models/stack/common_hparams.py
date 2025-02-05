# coding=utf-8
# Copyright 2022 The Tensor2Tensor Authors.
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

"""Hyperparameters and ranges common to multiple models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip  # pylint: disable=redefined-builtin

from clrs._src.memory_models.stack import hparam


# from tensor2tensor.utils import hparam
# from tensor2tensor.utils import registry

# import tensorflow.compat.v1 as tf


def basic_params1():
    """A set of basic hyperparameters."""
    return hparam.HParams(
        # If the problem consists of variable-length sequences
        # (see problem.batch_size_means_tokens()), then this is the number
        # of tokens per batch per GPU or per TPU core.  Otherwise, this is
        # the number of examples per GPU or per TPU core.
        batch_size=4096,
        batch_shuffle_size=512,
        # If True, then if the features are of variable length, the batch_size is
        # used as the actual batch size (and not tokens per batch).
        use_fixed_batch_size=False,
        num_hidden_layers=4,
        kernel_height=3,
        kernel_width=1,
        hidden_size=64,
        compress_steps=0,
        # All hyperparameters ending in "dropout" are automatically set to 0.0
        # when not in training mode.
        dropout=0.2,
        clip_grad_norm=2.0,
        grad_noise_scale=0.0,
        summarize_grads=False,
        # Flag for whether mlperf mode is on
        mlperf_mode=False,
        # Whether to log the name and size of every variable
        summarize_vars=False,
        initializer="orthogonal",
        initializer_gain=1.5,
        label_smoothing=0.1,
        optimizer="adam",
        optimizer_adam_epsilon=1e-6,
        optimizer_adam_beta1=0.85,
        optimizer_adam_beta2=0.997,
        optimizer_momentum_momentum=0.9,
        optimizer_momentum_nesterov=False,
        optimizer_adafactor_beta1=0.0,
        optimizer_adafactor_beta2=0.999,
        optimizer_adafactor_factored=True,
        optimizer_adafactor_decay_type="pow",
        optimizer_adafactor_memory_exponent=0.8,
        optimizer_adafactor_clipping_threshold=1.0,
        optimizer_adafactor_multiply_by_parameter_scale=True,
        # Number of accumulating steps for multi step optimizers.
        optimizer_multistep_accumulate_steps=0,
        # Loss scaling used.
        # Generally only necessary with mixed precision training.
        # Mixed precision training only supports exponential scaling currently
        # To disable the scaler, see to 0/False
        mixed_precision_optimizer_loss_scaler="exponential",
        # Determines the initial loss scaling value for mixed precision
        mixed_precision_optimizer_init_loss_scale=2 ** 15,
        # Whether to zero gradients that were not computed, so that the
        # appropriate slots are created. Useful for sharing checkpoints between
        # models with different sets of heads.
        optimizer_zero_grads=False,
        weight_decay=1e-6,
        weight_noise=0.0,
        # Defines the learning rate as a product of named functions.
        # Available functions are listed in learning_rate._LEARNING_RATE_FUNCTIONS
        # e.g. "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size"
        learning_rate_schedule="legacy",
        learning_rate_constant=1.0,
        # If learning_rate_schedule=="legacy",
        # then we specify decay scheme here.  Warmup is always exponential,
        # except with "noam" learning rate decay scheme.
        # see optimize.legacy_learning_rate_schedule()
        learning_rate_decay_scheme="none",
        # decay_steps and decay_staircase for learning_rate_decay_scheme=="exp"
        learning_rate_decay_steps=5000,
        learning_rate_decay_staircase=False,
        learning_rate_minimum=None,
        learning_rate_decay_rate=1.0,
        learning_rate_warmup_steps=100,
        learning_rate_cosine_cycle_steps=250000,
        learning_rate=0.1,
        sampling_method="argmax",  # "argmax" or "random"
        sampling_temp=1.0,  # temperature for sampling
        sampling_keep_top_k=-1,  # If >0, ignore all but the top k logits
        # expand the logits a piece at a time - saves memory.
        factored_logits=False,
        multiply_embedding_mode="sqrt_depth",
        # Parameters related to mixtures of experts.
        moe_hidden_sizes="2048",  # hidden layer sizes (comma-separated)
        moe_num_experts=64,  # number of experts per layer
        moe_k=2,  # how many experts to use for each batch element
        moe_loss_coef=1e-2,
        # Sequences of operations to perform on layer input and layer output.
        # Used by common_layers.layer_preprocess, common_layers.layer_postprocess
        # Each character represents an operation:
        # none: no preprocessing
        #    d: apply dropout
        #    n: apply normalization (see norm_type and norm_epsilon)
        #    a: add layer input (residual connection - only during postprocess)
        # The special string "none" is used instead of the empty string
        # to indicate no pre/postprocessing, since the empty string causes
        # trouble for hyperparameter tuning.
        # of the transformer.  ("n", "da") seems better for harder-to-learn
        # models, so it should probably be the default.
        layer_preprocess_sequence="none",
        layer_postprocess_sequence="dan",
        # dropout rate to use during layer_preprocess and layer_postprocess
        layer_prepostprocess_dropout=0.1,
        # broadcast dimensions for layer_prepostprocess_dropout
        # a comma-separated list of integers.
        # see common_layers.dropout_with_broadcast_dims()
        # Change this to "1" to save memory.
        layer_prepostprocess_dropout_broadcast_dims="",
        # dropout some symbols (set them to 0) before embedding.
        symbol_dropout=0.0,
        # What type of normalization to use
        norm_type="layer",  # "batch", layer", "noam", "none".
        # epsilon parameter to normalization function
        norm_epsilon=1e-6,
        # pad vocabularies so that this value divides the vocabulary size.
        vocab_divisor=1,
        # During training, we drop sequences whose inputs and targets are shorter
        # than min_length
        min_length=0,
        # During training, we drop sequences whose inputs or targets are longer
        # than max_length.
        # If max_length==0, we use hparams.batch_size instead.
        max_length=0,
        # Pack examples on the fly.
        pack_dataset=False,
        # Use custom ops not included in standard tensorflow.
        use_custom_ops=True,
        # Split targets on the first axis into chunks of this length.
        split_targets_chunk_length=0,
        split_targets_max_chunks=100,
        split_targets_strided_training=False,
        # Maximum length in the smallest length bucket.  Setting this
        # flag too high will result in wasteful padding of short
        # sequences.  Due to some (hopefully) temporary hacks in the
        # data reading and batching code, setting this flag too low
        # results in a very long batch-shuffling queue.
        min_length_bucket=8,
        # This flag controls the number of length buckets in the data
        # reader.  The buckets have maximum lengths from
        # min_bucket_length to (max_length or batch_size), increasing
        # (approximately) by factors of length_bucket_step.
        length_bucket_step=1.1,
        # If set to True, drop sequences longer than max_length during eval.
        # This affects the validity of the evaluation metrics.
        eval_drop_long_sequences=False,
        # If True, run the model autoregressively instead of teacher-forcing
        # during eval
        eval_run_autoregressive=False,
        # (For features with symbol modality) If True, share all of the
        # input embeddings, target embeddings, and softmax weights.
        shared_embedding_and_softmax_weights=False,
        # (For features with symbol modality) If True, share the input embeddings
        # and target embeddings.
        shared_embedding=False,
        # (For features with symbol modality) Number to shard embeddings by.
        symbol_modality_num_shards=1,
        # Feature transformations are optional dictionaries comprising key-value
        # pairs of a feature name (str) and its transformation (function). If not
        # specified, T2TModel applies a default transformation according to the
        # feature's modality. Bottom is applicable to all features; loss, top, and
        # weights_fn are only applicable to target features.
        # defining variable scope names. Remove this hparam in the future.
        bottom={},
        loss={},
        name={},
        top={},
        weights_fn={},
        # The maximum length of "input" sequence.
        # Sequences longer than this value will be truncated. 0 or negative values
        # mean there is no maximum or truncation.
        # You can change this behavior by overriding preprocess_example() method
        # in your problem class.
        max_input_seq_length=0,
        # The maximum length of "target" sequence.
        # Sequences longer than this value will be truncated. 0 or negative values
        # mean there is no maximum or truncation.
        # You can change this behavior by overriding preprocess_example() method
        # in your problem class.
        max_target_seq_length=0,
        # if nonzero, we split the target sequences on example read.
        # This is for use with language modeling problems with fixed length
        # examples.  e.g.  The examples may be written with length 65536, but we
        # want to split each example into 64 examples of length 1024.
        split_to_length=0,
        # Video settings: how many frames to batch on input and targets.
        video_num_input_frames=1,
        video_num_target_frames=1,
        # This flag allows us to optionally treat a seq-to-seq problem
        # as a language model.  Legal values are:
        #
        # "none" - Do not prepend the inputs to the targets.
        # "prepend_inputs_masked_attention"
        #     replace "targets" in preprocessing with
        #     tf.concat([inputs, [0], targets], axis=1)
        #     i.e. we prepend the inputs to the targets with a single
        #     padding token in between.  Use masked self-attention on the
        #     entire resulting sequence.  During training, we compute losses on
        #     the combined sequence.  During eval, we compute the metrics
        #     on only the targets portion.
        # "prepend_inputs_full_attention"
        #     similar to the previous option except that each
        #     position in the inputs portion can see the
        #     entire inputs portion.  This removes the challenge of
        #     autoregressively predicting the inputs portion.
        prepend_mode="none",
        # Scheduled sampling is interesting for auto-regressive models.
        # It runs an additional step using the generated output as autoregressive
        # targets, which can improve the models inference results later. The
        # parameter scheduled_sampling_prob determines with what probability
        # will such additional step be run. It's turned off (0.0) by default.
        # This probability will exponentially warm up for the number of
        # steps determined by scheduled_sampling_warmup_steps.
        # The tensor used for the n-th pass will consist of outputs from
        # the (n-1)-th pass mixed with gold truth, with the proportion of gold
        # determined by scheduled_sampling_gold_mixin_prob. Control the number
        # of passes with scheduled_sampling_num_passes.
        scheduled_sampling_prob=0.0,
        scheduled_sampling_method="parallel",  # parallel or sequential.
        scheduled_sampling_warmup_steps=50000,
        scheduled_sampling_gold_mixin_prob=0.5,
        scheduled_sampling_num_passes=1,
        scheduled_sampling_warmup_schedule="exp",  # exp, linear, or sigmoid.

        # This setting controls whether to copy variables around in a daisy chain
        # (if true) or leave their placement to TensorFlow. It only affects multi
        # device training and mostly should be turned on for performance. One
        # exception are recurrent models: with dynamic loops it must be off.
        daisy_chain_variables=True,
        # If True in PREDICT mode, then last-position-only optimizations are not
        # used.
        force_full_predict=False,
        # Set this for pure model parallelism.  There is only one data shard.
        no_data_parallelism=False,
        # dtype used for activations. - "float32" or "bfloat16"
        # activation_dtype="bfloat16" currently only works on TPU.
        #    It lowers activation-memory usage
        #    and does not appear to affect quality.
        #    You can train on TPU with activation_dtype="bfloat16" and evaluate
        #    on CPU/GPU with activation_dtype="float32"
        activation_dtype="float32",
        # dtype used for parameters: "float32" or "bfloat16"
        # bfloat16 currently only works with optimizer="adafactor".
        #   The savings in memory allow for training larger models.
        #   Weights are encoded as (w*128)^8, using pseudostochastic
        #   roundoff.  Initial experiments show that model quality is similar
        #   to baseline for about 3M training steps, but worse thereafter.
        weight_dtype="float32",
        # Directory containing a checkpoint for a pretrained model. This will only
        # be used if a new run is being started. Parameters not found in the
        # pretrained model will be randomly initialized. Superfluous parameters in
        # the pretrained model will be ignored.
        pretrained_model_dir="",
        # Threshold used for two cases: the primary task probability for the
        # constant mixing schedule, and the exponential schedule limit for when
        # mixing should stop (eg: 0.5 means stop at 50-50 mixing, 0.8 means stop
        # at 20-80 mixing for the primary-others mixing case.)
        multiproblem_schedule_threshold=0.5,
        # For more than 2 tasks, we may want to specify per-task thresholds here.
        # In that case, this needs to be a string with as many floating point
        # numbers as the number of tasks in the multi-problem. These numbers
        # are later normalized to add up to 1 and taken as probabilities for
        # each task. This enforces a constant mixing schedule and if this is
        # empty then the threshold from above is used for the first task and
        # the other tasks get the remaining probability split uniformly.
        multiproblem_per_task_threshold="",
        # The number of examples at which the proportion of the mixed in datasets
        # is multiproblem_schedule_threshold
        multiproblem_schedule_max_examples=1e7,
        # When training multiproblems, we can mix the data according to different
        # schedules. Example: a constant schedule mixing 20-80 between the primary
        # and other tasks.
        # A list of supported schedules can be found in
        # `data_generators.multi_problem.py`.
        multiproblem_mixing_schedule="constant",
        # A boolean that decides whether input sequence losses and target label
        # losses in classification problems should be reweighted.
        multiproblem_reweight_label_loss=False,
        # How much weight the targets in classification problems receive. Inputs
        # receive 1 minus this weight.
        multiproblem_label_weight=0.5,
        # Hyperparameters for relative attention.
        # The maximum relative positional distance to learn an embedding for.
        max_relative_position=0,
        # If heads share the same relative embedding.
        heads_share_relative_embedding=False,
        # If relative embedding terms are added to values too.
        add_relative_to_values=False,
        # If enable the host_call which is executed every training step.
        # There could be a performance drop if host_call function is slow and
        # cannot keep up with the TPU-side computation.
        tpu_enable_host_call=False,
        # Pad batch dim of inputs to nearest multiple of batch multiple.
        pad_batch=False,
        # When true, do not evaluate on the language model data when running the
        # multiproblem since it can take a while. If False, set eval_steps to
        # something large like 6000 or 10000.
        multiproblem_target_eval_only=False,
        # Max out the vocab size to a power of 2 for efficiency and to reserve
        # extra space in the vocabulary for new task ids and label classes.
        multiproblem_vocab_size=-1,
        # When using multiproblem with generation tasks, need to truncate the
        # inputs and targets manually before concatenating them.
        multiproblem_max_input_length=-1,
        multiproblem_max_target_length=-1,
        # If positive, makes training targets fixed-length in MultiProblem.
        multiproblem_fixed_train_length=-1,
        # Load weights from a second model. For instance, when using
        # pre-trained weights, you might want to initialize the encoder
        # and decoder by loading different models.
        warm_start_from_second="",
        # Area attention hyper parameters
        area_value_mode="none",
        area_key_mode="none",
        # Using area attention for the number of layers from the bottom
        num_area_layers=0,
        max_area_width=1,
        max_area_height=1,
        memory_height=1,
        # Whether to use GPU automatic mixed precision (via graph rewrite)
        gpu_automatic_mixed_precision=False,
    )


class RangedHParams(object):
    """Defines parameter ranges for tuning."""

    # From ParameterConfig proto
    LINEAR_SCALE = 1
    LOG_SCALE = 2
    REVERSE_LOG_SCALE = 3

    SCALES_STR = {
        LINEAR_SCALE: "UNIT_LINEAR_SCALE",
        LOG_SCALE: "UNIT_LOG_SCALE",
        REVERSE_LOG_SCALE: "UNIT_REVERSE_LOG_SCALE",
    }

    def __init__(self):
        self._categorical_params = {}
        self._discrete_params = {}
        self._float_params = {}
        self._int_params = {}

    def _check_reset_and_type_change(self, name, orig_ctr):
        """Check if name is in orig_ctr or in one of the other type containers."""
        # Resetting a hyperparameter
        if name in orig_ctr:
            print(f"Overwriting hparam {name}")

        ctr_names = [
            (self._categorical_params, "categorical"),
            (self._discrete_params, "discrete"),
            (self._float_params, "float"),
            (self._int_params, "int"),
        ]
        ctrs, names = list(zip(*ctr_names))
        orig_name = names[ctrs.index(orig_ctr)]

        for ctr, ctr_name in ctr_names:
            if ctr is orig_ctr:
                continue

            # Using a different type for the same hyperparameter name
            if name in ctr:
                raise ValueError("Setting hyperparameter %s as type %s, but a "
                                 "hyperparemeter of the same name was originally "
                                 "registered as type %s" % (name, ctr_name, orig_name))

    def set_categorical(self, name, categories, length=None):
        self._check_reset_and_type_change(name, self._categorical_params)
        self._categorical_params[name] = (name, categories, length)

    def set_discrete(self, name, feasible_points, scale=None, length=None):
        self._check_reset_and_type_change(name, self._discrete_params)
        self._discrete_params[name] = (name, feasible_points, scale, length)

    def set_float(self, name, min_val, max_val, scale=None, length=None):
        self._check_reset_and_type_change(name, self._float_params)
        self._float_params[name] = (name, min_val, max_val, scale, length)

    def set_int(self, name, min_val, max_val, scale=None, length=None):
        self._check_reset_and_type_change(name, self._int_params)
        self._int_params[name] = (name, min_val, max_val, scale, length)

    def fix_select_params(self, hp):
        ctrs = [
            self._categorical_params, self._discrete_params, self._float_params,
            self._int_params
        ]
        for key, val in hp.values().iteritems():
            for ctr in ctrs:
                if key in ctr:
                    del ctr[key]
            self.set_discrete(key, [val])

    def to_parameter_specs(self, name_prefix=""):
        """To list of dicts suitable for Cloud ML Engine hyperparameter tuning."""
        specs = []
        for name, categories, _ in self._categorical_params.values():
            spec = {
                "parameterName": name_prefix + name,
                "type": "CATEGORICAL",
                "categoricalValues": categories,
            }
            specs.append(spec)

        for name, feasible_points, scale, _ in self._discrete_params.values():
            spec = {
                "parameterName": name_prefix + name,
                "type": "DISCRETE",
                "discreteValues": feasible_points,
            }
            if scale:
                spec["scaleType"] = self.SCALES_STR[scale]
            specs.append(spec)

        for name, min_val, max_val, scale, _ in self._float_params.values():
            spec = {
                "parameterName": name_prefix + name,
                "type": "DOUBLE",
                "minValue": min_val,
                "maxValue": max_val,
            }
            if scale:
                spec["scaleType"] = self.SCALES_STR[scale]
            specs.append(spec)

        for name, min_val, max_val, scale, _ in self._int_params.values():
            spec = {
                "parameterName": name_prefix + name,
                "type": "INTEGER",
                "minValue": min_val,
                "maxValue": max_val,
            }
            if scale:
                spec["scaleType"] = self.SCALES_STR[scale]
            specs.append(spec)

        return specs


def basic_range1(ranged_hparams):
    """A basic range of hyperparameters."""
    rhp = ranged_hparams
    rhp.set_discrete("batch_size", [1024, 2048, 4096])
    rhp.set_discrete("num_hidden_layers", [1, 2, 3, 4, 5, 6])
    rhp.set_discrete("hidden_size", [32, 64, 128, 256, 512], scale=rhp.LOG_SCALE)
    rhp.set_discrete("kernel_height", [1, 3, 5, 7])
    rhp.set_discrete("kernel_width", [1, 3, 5, 7])
    rhp.set_discrete("compress_steps", [0, 1, 2])
    rhp.set_float("dropout", 0.0, 0.5)
    rhp.set_float("weight_decay", 1e-4, 10.0, scale=rhp.LOG_SCALE)
    rhp.set_float("label_smoothing", 0.0, 0.2)
    rhp.set_float("clip_grad_norm", 0.01, 50.0, scale=rhp.LOG_SCALE)
    rhp.set_float("learning_rate", 0.005, 2.0, scale=rhp.LOG_SCALE)
    rhp.set_categorical("initializer",
                        ["uniform", "orthogonal", "uniform_unit_scaling"])
    rhp.set_float("initializer_gain", 0.5, 3.5)
    rhp.set_categorical("learning_rate_decay_scheme",
                        ["none", "sqrt", "noam", "exp"])
    rhp.set_float("optimizer_adam_epsilon", 1e-7, 1e-2, scale=rhp.LOG_SCALE)
    rhp.set_float("optimizer_adam_beta1", 0.8, 0.9)
    rhp.set_float("optimizer_adam_beta2", 0.995, 0.999)
    rhp.set_categorical(
        "optimizer",
        ["adam", "adagrad", "momentum", "rms_prop", "sgd", "yellow_fin"])


def basic_moe_range(rhp):
    """Moe range; when this parameter is unused, it allows us to see variance."""
    rhp.set_float("moe_loss_coef", 0.01, 0.02)
