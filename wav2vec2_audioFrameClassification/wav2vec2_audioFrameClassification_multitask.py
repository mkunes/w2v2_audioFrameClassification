# Audio frame classification using wav2vec2, with multitask support
# version 2022-10-24
#
# Marie KUNESOVA (https://github.com/mkunes)
# 2022
# 
#   TODO: fix off-by-one mismatch between the sizes of predicted labels and ref. labels
#       (20s of audio -> 1000 ref. labels, but only 999 audio frames; currently I'm removing the last label during dataset preprocessing)
#   TODO: add full support for num_labels > 1
#   TODO: remove unnecessary imports
#   TODO: check if all "optional" arguments are truly optional and won't cause an error when missing
#   TODO: automatically detect and warn if training loss is not decreasing in the first few epochs? 
#           (the model sometimes gets stuck at the start due to poor initialization...)
#   TODO: verify if loss weights are working as intended. improve auto weights - probably could be done in a better way
#   TODO: is validation/evaluation loss correct? some of the numbers seem strange
#   TODO: report loss as average, not sum (so it does not depend on dataset size)
#   TODO: add support for a config file, there are too many command line arguments
#   TODO: eliminate all other sources of randomness when using a fixed RNG seed
#
#
# Changelog:
#   2022-10-25
#       - fixed crash when using default loss weights during an evaluation-only run (wrong variable name)
#   2022-10-24
#       - initial GitHub commit at https://github.com/mkunes/w2v2_audioFrameClassification/
#   

import numpy as np
import os

import datasets
from datasets import load_dataset, load_metric

try:
    from datasets import disable_caching
    DISABLE_CACHING_SUPPORTED = True
except ImportError: # older versions of datasets (before 2.x?) don't have disable_caching
    DISABLE_CACHING_SUPPORTED = False

import transformers
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioFrameClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2Model
)

from transformers.data import default_data_collator

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler, set_seed
from tqdm.auto import tqdm

from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

import argparse

import warnings

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_SUPPORTED = True
except ImportError: 
    TENSORBOARD_SUPPORTED = False

try:
    import wandb
    WANDB_SUPPORTED = True
except ImportError: 
    WANDB_SUPPORTED = False

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

@dataclass
class TokenClassifierOutput_multitask(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    losses_all: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ForAudioFrameClassification_multitask(transformers.Wav2Vec2ForAudioFrameClassification):
    def __init__(self, config, num_tasks=None,loss_weights=None,use_sigmoid_classification=None):
        # defaults (if not specified and not found in starting checkpoint's config):
        #   num_tasks - is required, throws error if it can't be determined
        #   loss_weights - defaults to 1 for each task (equal contribution from all losses)
        #   use_sigmoid_classification - defaults to False
        super().__init__(config)
        self.num_labels = config.num_labels

        if self.num_labels != 1: 
            raise ValueError(
                "This model curently only supports num_labels == 1"
                # single task classification with num_labels > 1 *might* work, but it hasn't been tested at all
                # multitask with num_labels > 1 *definitely* won't work
            )


        if config.task_specific_params is None:
            config.task_specific_params = {}

        # set the number of classification tasks (-> the number of outputs of the last layer)
        if num_tasks is not None:
            self.num_tasks = num_tasks
            if "num_tasks" in config.task_specific_params and num_tasks != config.task_specific_params["num_tasks"]:
                raise ValueError(
                    "Number of tasks in the dataset does not match the value saved in model config (config.task_specific_params['num_tasks'])"
                )
            config.task_specific_params["num_tasks"] = num_tasks
        elif "num_tasks" in config.task_specific_params:
            self.num_tasks = config.task_specific_params["num_tasks"]
            num_tasks = self.num_tasks # just in case
        else:
            raise ValueError(
                "The number of tasks could not be determined"
            )

        # set the loss weights for each task
        if loss_weights is not None:
            self.loss_weights = loss_weights
            config.task_specific_params["loss_weights"] = loss_weights
        elif "loss_weights" in config.task_specific_params:
            self.loss_weights = config.task_specific_params["loss_weights"]
        else:
            self.loss_weights = [1] * self.num_tasks
            print("Loss weights not specified; defaulting to equal weights for all tasks.")

        if use_sigmoid_classification is not None:
            self.use_sigmoid_classification = use_sigmoid_classification
            config.task_specific_params["use_sigmoid_classification"] = use_sigmoid_classification
        elif "use_sigmoid_classification" in config.task_specific_params:
            self.use_sigmoid_classification = config.task_specific_params["use_sigmoid_classification"]
        else: # defaults
            self.use_sigmoid_classification = False
            config.task_specific_params["use_sigmoid_classification"] = False
            print("'use_sigmoid_classification' not specified; defaulting to False.")

        ## not needed - num_labels > 1 is currently not supported at all
        # if self.num_labels > 1 and self.num_tasks > 1:
        #     raise ValueError(
        #         "Multi-task audio frame classification currently only supports regression (if num_tasks > 1, num_labels must be 1)"
        #     )

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )

        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        if self.num_tasks > 1:
            num_outputs = self.num_tasks
        else:
            num_outputs = config.num_labels

        if self.use_sigmoid_classification:
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, num_outputs),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Linear(config.hidden_size, num_outputs)

        self.init_weights()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None, # ADDED
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        logits = self.classifier(hidden_states)
        

        # ADDED
        loss = None
        losses_all = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()

                loss = 0

                lb = labels.view(-1,self.num_tasks)  # batch_size x N x num_tasks -> (N*batch_size) x num_tasks 
                lg = logits.view(-1,self.num_tasks)
                #lb = lb.squeeze()

                losses_all = []
                for iTask in range(self.num_tasks):

                    task_loss = self.loss_weights[iTask] * loss_fct(lg[:,iTask], lb[:,iTask])
                    #print("loss %d:"%iTask)
                    #print(task_loss)
                    losses_all.append(task_loss)

                    loss += task_loss

            else:
                raise ValueError(
                    "This model curently only supports num_labels == 1"
                    # single task classification with num_labels > 1 *might* work, but it hasn't been tested at all
                    # multitask with num_labels > 1 *definitely* won't work
                )
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # #loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))
            

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output + (losses_all,)) if loss is not None else output

        if self.num_tasks > 1:
            return TokenClassifierOutput_multitask(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                losses_all=losses_all,
            )
        else:
            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


def load_json_dataset_forAudioFrameClassification(file_train,file_eval,file_valid=None,multitask=False):

    data_files = {}
    if file_train is not None:
        data_files["train"] = file_train
    if file_eval is not None:
        data_files["eval"] = file_eval
    if file_valid is not None:
        data_files["valid"] = file_valid

    if multitask:
        features = datasets.Features({'path': datasets.features.Value('string'), 'label': datasets.features.Sequence(datasets.features.Sequence(datasets.features.Value(dtype='float64')))})
    else:
        features = datasets.Features({'path': datasets.features.Value('string'), 'label': datasets.features.Sequence(datasets.features.Value(dtype='float64'))})

    try:
        dataset = datasets.load_dataset("json", data_files=data_files, features=features)
    except TypeError as e:
        print("\nTypeError encountered while trying to load the dataset from a json file. Check if the file matches the required format:\n"
            "  - When using --multitask, the labels for each audio file must be formatted as a 2D list of floats (even if there's only one task).\n"
            "  - Without --multitask, they are expected as a 1D list of floats.\n")
        raise e

    if multitask:
        if file_train is not None:

            num_tasks_train = len(dataset["train"]["label"][0][0])

            if file_valid is not None:
                num_tasks_valid = len(dataset["valid"]["label"][0][0])

                if num_tasks_valid != num_tasks_train:
                    raise ValueError("The number of labels per audio frame in the validation set (%d) does not match the training set (%d)"%(num_tasks_valid,num_tasks_train))

            # note: the number of labels in the training data does not matter

            print("Multi-task classification - found %d labels per audio frame\n"%num_tasks_train)
        else:
            num_tasks_train = None
    else:
        num_tasks_train = 1

    return dataset,num_tasks_train

def get_loss_weights(train_labels,num_tasks):
    # automatically calculate weights for each task

    loss_weights = [1] * num_tasks

    thresh = 0.5

    num_labels_total = 0
    labels_positive = [0] * num_tasks
    labels_negative = [0] * num_tasks

    # count the number of positive labels ( >= 0.5) for each task
    for iSample in range(len(train_labels)):
        num_labels_total += len(train_labels[iSample])

        for iLabel in range(len(train_labels[iSample])):
            for iTask in range(num_tasks):
                if train_labels[iSample][iLabel][iTask] >= thresh:
                    labels_positive[iTask] += 1

    if num_labels_total == 0:
        raise ValueError("No labels found")

    loss_weights = [0] * num_tasks

    # the weight of each task is inversely proportional to the ratio of positive (>= 0.5) or negative (< 0.5) labels, whichever is less common
    # the task's weight is 1/ratio or in other words, N/min(N_pos, N_neg)
    # (e.g. if the ratios of positive labels are [0.5, 0.75, 0.125], first convert the "0.75 positive" to "0.25 negative" (less common label),
    #   and then the weights will be [1/0.5, 1/0.75, 1/0.125], so [2,4,8] )
    for iTask in range(num_tasks):
        labels_negative[iTask] = num_labels_total - labels_positive[iTask]
        loss_weights[iTask] = num_labels_total / min(labels_positive[iTask],labels_negative[iTask])

    print("Automatic loss weights:")
    print(loss_weights)

    return loss_weights

def collator_multitask_audio_classification(batch_old):
    # custom collator function for multitask classification - the default one won't work

    first = batch_old[0]

    if type(first) is dict and "labels" in first:
        labels = torch.stack([torch.stack(f["labels"]) for f in batch_old])
        batch = {"labels": labels}
    elif type(first) is dict and "label" in first:
        labels = torch.stack([torch.stack(f["label"]) for f in batch_old])
        batch = {"labels": labels}
    else:
        batch = {}

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k in first.keys():
        if k not in ("labels","label") and first[k] is not None and not isinstance(first[k], str):
            if torch.is_tensor(first[k]):
                batch[k] = torch.stack([f[k] for f in batch_old])
            else:
                batch[k] = torch.tensor([f[k] for f in batch_old], dtype=torch.long)

    return batch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the model')
    parser.add_argument('--file_train', type=str, required=False, default=None,help='Path to the training dataset (a file in json format)')
    parser.add_argument('--file_valid', type=str, required=False, default=None, help='Path to the validation dataset (a file in json format)')
    parser.add_argument('--file_eval', type=str, required=False, default=None, help='Path to the evaluation (test) dataset (a file in json format)')
    parser.add_argument('--num_epochs', type=int, required=False, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--feature_extractor_name', type=str, required=False, default=None, help='Name or path of the feature extractor')
    parser.add_argument('--file_output', type=str, required=False, default="output.txt", help='Path for the output file')
    parser.add_argument('--model_save_dir', type=str, required=False, default=None, help='Directory for saving the training log and the finetuned model')
    parser.add_argument('--max_duration', type=float, required=False, default=30.0, help='Maximum duration of audio files, default = 30s')
    parser.add_argument('--seed', type=int, required=False, default=None, help='Training seed')
    parser.add_argument('--mode', type=str, required=False, default="both", choices=['train','eval','both'], help='Mode: ''train'', ''eval'' or ''both'' (default is ''both'')')
    parser.add_argument('--epochs_between_checkpoints', type=int, required=False, default=0, help='Number of epochs between saved checkpoints during training. Default is 0 - no checkpoints.')
    parser.add_argument('--dataset_caching', type=int, required=False, default=1, choices=[0,1], help='Enable/disable dataset caching: 0 = no caching, 1 = enabled caching. Default: 1' )
    parser.add_argument('--tensorboard_logging', dest="tensorboard_logging", action='store_true', help='Enable Tensorboard logging')
    parser.add_argument('--wandb_logging', dest="wandb_logging", action='store_true', help='Upload results via Weights & Biases')
    parser.add_argument('--wandb_project', type=str, required=False, default=None, help='Weights & Biases project name' )
    parser.add_argument('--wandb_entity', type=str, required=False, default=None, help='Weights & Biases entity name' )
    parser.add_argument('--lr_init', type=float, required=False, default=5e-5)
    parser.add_argument('--lr_num_warmup_steps', type=int, required=False, default=0)
    parser.add_argument('--multitask',dest="multitask", action='store_true', help='Enable multi-task classification')
    parser.add_argument('--remove_last_label', type=int, required=False, choices=[0,1], default=1, help='Remove the last ref. label (to match the length of outputs)? (0 - no, 1 - yes) Default: 1')
    parser.add_argument('--sigmoid',dest="use_sigmoid_classification", action='store_true', help='Use sigmoid function on last layer output')
    parser.add_argument('--loss_weights',type=str,default=None,help='Loss weights for all tasks. Options: a) "auto" - smart automatic weights, or b) a string of comma-delimited values, e.g. "1,2,3". '
        'Defaults to whatever is stored in the model (if any) or the same weight for all tasks.')
    
    # deprecated:
    parser.add_argument('--finetuning_framework', type=str, required=False, default=None, help='DEPRECATED: Finetuning framework (pytorch is the only option now)')
    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    file_train = args.file_train
    file_valid = args.file_valid
    file_eval = args.file_eval
    num_epochs = args.num_epochs #20
    batch_size = args.batch_size #1
    feature_extractor_name = args.feature_extractor_name
    file_output = args.file_output
    model_save_dir = args.model_save_dir
    max_duration = args.max_duration
    seed = args.seed
    mode = args.mode
    epochs_between_checkpoints = args.epochs_between_checkpoints
    dataset_caching = args.dataset_caching
    tensorboard_logging = args.tensorboard_logging
    wandb_logging = args.wandb_logging
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    lr_init = args.lr_init
    lr_num_warmup_steps = args.lr_num_warmup_steps
    multitask = args.multitask
    use_sigmoid_classification = args.use_sigmoid_classification

    use_auto_weights = False
    arg_loss_weights = None
    if args.loss_weights is not None:
        if args.loss_weights == "auto":
            # automatic loss weights
            use_auto_weights = True
        else:
            # user-specified loss weights
            arg_loss_weights = args.loss_weights.split(",")
            arg_loss_weights = list(map(float, arg_loss_weights))

    if args.remove_last_label > 0:
        remove_extra_label = True # in a 20.0 s audio, there will be 1000 labels but only 999 logits -> remove the last label so the numbers match
    else: # if the labels are already fixed elsewhere
        remove_extra_label = False


    num_labels = 1 # this is currently not configurable - the model only supports regression, not softmax


    do_train = mode in ['train','both']
    do_eval = mode in ['eval','both']

    if args.finetuning_framework is not None:
        warnings.warn(
            "Argument '--finetuning_framework' is deprecated and will be removed in a future version. Pytorch is the only finetuning option now and does not need to be specified.", 
            DeprecationWarning
        )

    if dataset_caching == 0:
        if DISABLE_CACHING_SUPPORTED:
            disable_caching() # disable dataset caching to avoid cluttering the disk
        else:
            warnings.warn("Could not import 'datasets.disable_caching', so dataset caching will remain ENABLED. Check your version of datasets - it may be too old (the minimum requirement seems to be v. 2.0).")


    # TensorBoard logging
    if tensorboard_logging and do_train:
        if TENSORBOARD_SUPPORTED:
            writerTensorboard = SummaryWriter(model_save_dir + '/TensorBoard/')
            print("\n======  Loss will be saved in TensorBoard in path: " + model_save_dir + "/TensorBoard/  ======\n")
        else:
            warnings.warn("Could not import 'SummaryWriter' from 'torch.utils.tensorboard', so TensorBoard logging will NOT be enabled.")
            tensorboard_logging = False

    # Weights & Biases logging
    if wandb_logging and do_train:
        if WANDB_SUPPORTED:
            config = {}
            if wandb_project is None:
                if wandb_entity is None:
                    wandb.init()
                else:
                    wandb.init(entity=wandb_entity)
            else:
                if wandb_entity is None:
                    wandb.init(project=wandb_project)
                else:
                    wandb.init(project=wandb_project, entity=wandb_entity)
            wandb.config = {
                "learning_rate": lr_init,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "model_checkpoint": model_checkpoint,
                "file_train:": file_train,
                "file_eval:": file_eval,
                "file_valid:": file_valid
            }
        else:
            warnings.warn("Could not import 'wandb', so results will NOT be uploaded via Weights & Biases.")
            wandb_logging = False


    if epochs_between_checkpoints < 0:
        raise ValueError("''--epochs_between_checkpoints'' must be >= 0")

    if do_train:
        if file_train is None:
            raise ValueError("Training requires path to the training dataset (argument '--file_train <path>'). "
                             "To disable training and only run evaluation using the existing model, use '--mode 'eval''")
        if num_epochs is None:
            raise ValueError("For training the model, the number of epochs must be specified (argument '--num_epochs <number>'). "
                             "To disable training and only run evaluation using the existing model, use '--mode 'eval''")
        if model_save_dir is None:
            warnings.warn("argument ''--model_save_dir'' is not set -> the finetuned model will NOT be saved.")
            if epochs_between_checkpoints > 0:
                print("Checkpoints during training will also NOT be saved.")

        if file_valid is None:
            print("There is no validation set. Loss will be calculated only on the training set.")
            do_validation = False
        else:
            do_validation = True
    else: 
        do_validation = False

    if do_eval:
        if file_eval is None:
            raise ValueError("Evaluation requires path to the evaluation dataset (argument '--file_eval <path>'). "
                             "To disable evaluation and only perform training, use '--mode 'train''")

    if model_save_dir is None or epochs_between_checkpoints == 0:
        save_checkpoints = False
    else:
        save_checkpoints = True

    # set a fixed random seed, if it was specified
    # Note: This doesn't currently ensure reproducibility of training - there still appears to be some other source of randomness
    #    However, the main point of this is to guarantee *different* seeds when needed
    # TODO: figure out what's missing
    if seed is not None:
        set_seed(seed)
        torch.manual_seed(seed)

    freeze_feature_encoder = True
    freeze_base_model = False

    metric = load_metric("mse")

    dataset,num_tasks = load_json_dataset_forAudioFrameClassification(file_train,file_eval,file_valid,multitask=multitask)

    if use_auto_weights:
        if do_train and multitask: 
            loss_weights = get_loss_weights(dataset["train"]["label"],num_tasks)
        else: # otherwise just use the defaults, or whatever is stored in the model
            loss_weights = None
    else:
        loss_weights = arg_loss_weights
        if loss_weights is not None and len(loss_weights) != num_tasks:
            raise ValueError("Mismatch between the number of values in 'loss_weights' (%d) and the number of tasks in the training data (%d)"%(len(loss_weights),num_tasks))

    model = Wav2Vec2ForAudioFrameClassification_multitask.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels, 
        num_tasks=num_tasks, 
        loss_weights=loss_weights, 
        use_sigmoid_classification=use_sigmoid_classification
    )
    if num_tasks is None:
        num_tasks = model.num_tasks

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        feature_extractor_name or model_checkpoint,
        #return_attention_mask=model_args.attention_mask,
        #cache_dir=model_args.cache_dir,
        #revision=model_args.model_revision,
    )

    dataset = dataset.rename_column("path","audio")
    dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

    def preprocess_function(examples):
        if examples is None:
            return None

        audio_arrays = [x["array"] for x in examples["audio"]]
        labels = examples["label"]

        sampling_rate = 16000

        labels_rate = 50 # labels per second

        num_padded_labels = round(max_duration * labels_rate)

        # TODO: check if the labels' size roughly matches the *unpadded* audio 
        #   (in case the sample rate of the labels is different from what it's supposed to be)

        # TODO: check if audio durations <= max_duration

        # TODO: check if it should be rounded up or down (down I think)

        for label in labels:
            if len(label) < num_padded_labels:
                if num_tasks > 1:
                    for n in range(len(label), num_padded_labels):
                        label.append([0] * num_tasks)
                else:
                    for n in range(len(label), num_padded_labels):
                        label.append(0)
            if remove_extra_label:
                label.pop()

        labels = np.array(labels)

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            padding='max_length', # pad to max_length, not just to the longest sequence
            max_length=int(sampling_rate * max_duration), 
            truncation=False,
        )

        inputs["label"] = labels

        return inputs

    def compute_metrics(eval_pred): # not used anymore, I think?
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        labels = labels.reshape(-1)
        
        predictions = predictions.reshape(-1)

        return metric.compute(predictions=predictions, references=labels)

    # freeze the convolutional waveform encoder
    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    if freeze_base_model:
        model.freeze_base_model()


    if num_tasks > 1:
        data_collator = collator_multitask_audio_classification
    else:
        data_collator = default_data_collator

    # -------------
    # process the train/val/test data
    # -------------

    if do_train:
        print("Processing training data...")
        processed_dataset_train = dataset["train"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_train = processed_dataset_train.rename_column("label", "labels")
        processed_dataset_train.set_format("torch", columns=["input_values", "labels"])
        train_dataloader = DataLoader(processed_dataset_train, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    else:
        processed_dataset_train = None
        num_tasks = model.num_tasks

    if do_validation:
        print("Processing validation data...")
        processed_dataset_valid = dataset["valid"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_valid = processed_dataset_valid.rename_column("label", "labels")
        processed_dataset_valid.set_format("torch", columns=["input_values", "labels"])
        valid_dataloader = DataLoader(processed_dataset_valid, shuffle=False, batch_size=batch_size, collate_fn = data_collator)
    else:
        processed_dataset_valid = None

    if do_eval:
        print("Processing test data...")
        processed_dataset_test = dataset["eval"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_test = processed_dataset_test.rename_column("label", "labels")
        processed_dataset_test.set_format("torch", columns=["input_values", "labels"])
        eval_dataloader = DataLoader(processed_dataset_test, batch_size=1, collate_fn = data_collator)
    else:
        processed_dataset_test = None


    # ----------
    # Training
    # ----------

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if do_train:

        print("Starting training...")

        optimizer = AdamW(model.parameters(), lr=lr_init)

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=lr_num_warmup_steps, num_training_steps=num_training_steps
        )
        
        progress_bar = tqdm(range(num_training_steps))

        if model_save_dir is not None:
            if model_save_dir != "":
                os.makedirs(model_save_dir, exist_ok=True)

            logfile = open(os.path.join(model_save_dir,"log.csv"), "w")
        else:
            out_dir = os.path.dirname(file_output)
            if out_dir != "":
                os.makedirs(out_dir, exist_ok=True)
            logfile = open(file_output + ".log.csv", "w")

        logfile.write("epoch,train loss,val loss")
        if num_tasks > 1:
            for iTask in range(num_tasks):
                logfile.write(",train loss (task %d)"%iTask)
            for iTask in range(num_tasks):
                logfile.write(",val loss (task %d)"%iTask)
        logfile.write("\n")
        logfile.flush()

        
        for epoch in range(num_epochs):

            model.train()

            # save checkpoint every N epochs
            if save_checkpoints and epoch > 0 and (epoch % epochs_between_checkpoints == 0):
                epoch_dir = os.path.join(model_save_dir,"epoch%d"%epoch)
                os.makedirs(epoch_dir, exist_ok=True)
                model.save_pretrained(epoch_dir)


            train_loss = 0
            train_losses_all = [0] * num_tasks
            for i, batch in enumerate(train_dataloader):

                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)

                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                train_loss = train_loss + loss.detach().item()

                if num_tasks > 1:
                    losses_all = outputs.losses_all
                    for iTask in range(num_tasks):
                        train_losses_all[iTask] += losses_all[iTask].detach().item()


                if tensorboard_logging:
                    writerTensorboard.add_scalar("Loss_Train_individual",  loss.detach().item(), i+len(train_dataloader)*epoch)
                    for iTask in range(num_tasks):
                        writerTensorboard.add_scalar("Loss_Train_individual_task%d"%iTask, losses_all[iTask].detach().item(), i+len(train_dataloader)*epoch)
                    #writerTensorboard.add_scalar("LearningRate_individual", optimizer.lr, i+len(train_dataloader)*epoch)
                    writerTensorboard.flush()

                if wandb_logging:
                    #wandb.log({"LearningRate_individual": optimizer.lr})
                    wandb.log({"Loss_Train_individual": loss.detach().item()})
                    for iTask in range(num_tasks):
                        wandb.log({"Loss_Train_individual_task%d"%iTask: losses_all[iTask].detach().item()})

            if do_validation:
                model.eval()

                val_loss = 0
                val_losses_all = [0] * num_tasks

                for i, batch in enumerate(valid_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    val_loss = val_loss + loss.detach().item()

                    if num_tasks > 1:
                        losses_all = outputs.losses_all
                        for iTask in range(num_tasks):
                            val_losses_all[iTask] += losses_all[iTask].detach().item()

                    if tensorboard_logging:
                        writerTensorboard.add_scalar("Loss_Val_individual", loss.detach().item(), i + len(valid_dataloader) * epoch)
                        for iTask in range(num_tasks):
                            writerTensorboard.add_scalar("Loss_Val_individual_task%d"%iTask, losses_all[iTask].detach().item(), i+len(valid_dataloader)*epoch)
                        writerTensorboard.flush()

                    if wandb_logging:
                        wandb.log({"Loss_Val_individual": loss.detach().item()})
                        for iTask in range(num_tasks):
                            wandb.log({"Loss_Val_individual_task%d"%iTask: losses_all[iTask].detach().item()})

                

                if tensorboard_logging:
                    writerTensorboard.add_scalars("Loss", {'Train':train_loss,'Val':val_loss}, epoch)
                    #writerTensorboard.add_scalar("LearningRate",optimizer.lr, epoch)
                    for iTask in range(num_tasks):
                        writerTensorboard.add_scalar("Loss_Train_individual_task%d"%iTask, train_losses_all[iTask], epoch)
                        writerTensorboard.add_scalar("Loss_Val_individual_task%d"%iTask, val_losses_all[iTask], epoch)
                    writerTensorboard.flush()

                if wandb_logging:
                    wandb.log({"Loss_Train": train_loss})
                    wandb.log({"Loss_Val": val_loss})
                    for iTask in range(num_tasks):
                        wandb.log({"Loss_Train_task%d"%iTask: train_losses_all[iTask]})
                        wandb.log({"Loss_Val_task%d"%iTask: val_losses_all[iTask]})
                    #wandb.log({"LearningRate": optimizer.lr})
                    # Optional
                    wandb.watch(model)

            else:

                if tensorboard_logging:
                    writerTensorboard.add_scalar("Loss_Train", train_loss, epoch)
                    #writerTensorboard.add_scalar("LearningRate", lr_scheduler.item() , epoch)
                    for iTask in range(num_tasks):
                        writerTensorboard.add_scalar("Loss_Train_individual_task%d"%iTask, train_losses_all[iTask], epoch)
                    writerTensorboard.flush()

                if wandb_logging:
                    wandb.log({"Loss_Train": train_loss})
                    for iTask in range(num_tasks):
                        wandb.log({"Loss_Train_task%d"%iTask: train_losses_all[iTask]})

            # logging to a CSV file:
            logfile.write("%d,%f"%(epoch,train_loss))
            if do_validation:
                logfile.write(",%f"%val_loss)
            else:
                logfile.write(",N/A")
            if num_tasks > 1:
                for iTask in range(num_tasks):
                    logfile.write(",%f"%train_losses_all[iTask])
                if do_validation:
                    for iTask in range(num_tasks):
                        logfile.write(",%f"%val_losses_all[iTask])
                else:
                    for iTask in range(num_tasks):
                        logfile.write(",N/A")
            logfile.write("\n")
            logfile.flush()


        logfile.close()
        progress_bar.close()

        if tensorboard_logging:
            writerTensorboard.close()

        if model_save_dir is not None:
            model.save_pretrained(model_save_dir)


    if do_eval:

        print("Starting evaluation...")

        out_dir = os.path.dirname(file_output)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)


        model.eval()
        predictions_all = []

        progress_bar = tqdm(range(len(eval_dataloader)))

        test_loss = 0
        test_losses_all = [0] * num_tasks
        
        for i, batch in enumerate(eval_dataloader):

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            test_loss = test_loss + loss.detach().item()

            if num_tasks > 1:
                losses_all = outputs.losses_all
                for iTask in range(num_tasks):
                    test_losses_all[iTask] += losses_all[iTask].detach().item()

            logits = outputs.logits

            if model.num_labels == 1:
                predictions = logits
            else:
                predictions = torch.argmax(logits, dim=-1)

            labels = batch["labels"]
            predictions = predictions.reshape(-1,num_tasks)
            predictions_all.append(predictions.cpu().detach().numpy())

            metric.add_batch(predictions=predictions.reshape(-1), references=labels.reshape(-1))

            progress_bar.update(1)

        progress_bar.close()
        score = metric.compute()

        # logging to a CSV file:
        out_dir = os.path.dirname(file_output)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
        logfile = open(file_output + ".test-log.csv", "w")

        logfile.write("test loss")
        if num_tasks > 1:
            for iTask in range(num_tasks):
                logfile.write(",test loss (task %d)"%iTask)
        logfile.write("\n")

        logfile.write("%f"%test_loss)
        if num_tasks > 1:
            for iTask in range(num_tasks):
                logfile.write(",%f"%test_losses_all[iTask])
        logfile.write("\n")
        logfile.flush()
        logfile.close()

        print(score)

        # save all predictions to a file
        with open(file_output, 'w') as file:
            for ii,prediction in enumerate(predictions_all):
                file.write(dataset["eval"][ii]["audio"]["path"])
                for iTask in range(num_tasks):
                    file.write(",[")
                    prediction[:,iTask].tofile(file,sep=" ", format="%s")
                    file.write("]")
                file.write("\n")

        if num_tasks > 1:
            # save predictions from each task to separate files

            # create N empty files
            file_list = []
            for iTask in range(num_tasks):
                if file_output.endswith('.txt'):
                    output_file = file_output[:-4] + '.task%d.txt'%(iTask+1)
                else:
                    output_file = file_output + '.task%d.txt'%(iTask+1)

                file = open(output_file,'w')
                file_list.append(file)

            # write predictions to those files
            for ii,prediction in enumerate(predictions_all):
                path = dataset["eval"][ii]["audio"]["path"]
                for iTask in range(num_tasks):
                    file_list[iTask].write(path)
                    file_list[iTask].write(",[")
                    prediction[:,iTask].tofile(file_list[iTask],sep=" ", format="%s")
                    file_list[iTask].write("]\n")
              
            # close those files
            for iTask in range(num_tasks):
                file_list[iTask].close()


    

if __name__ == '__main__':
    main()


