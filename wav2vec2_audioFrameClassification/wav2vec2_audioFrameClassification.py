# WIP, v2022-09-29
# Audio frame classification using wav2vec2
#   TODO: fix off-by-one mismatch between the sizes of predicted labels and ref. labels
#       (20s of audio -> 1000 ref. labels, but only 999 audio frames; currently I'm removing the last label during dataset preprocessing)
#   TODO: check if everything still works when batch_size > 1 or num_labels > 1
#   TODO: remove unnecessary imports
#   TODO: check if all "optional" arguments are truly optional and won't cause a crash when missing
#   TODO: automatically detect and warn if training loss is not decreasing in the first few epochs? 
#           (the model sometimes gets stuck at the start due to poor initialization...)
#
# mkunes, 2022
#
# Changelog:
#   2022-09-29
#       - removed the outdated option to use transformers.Trainer for finetuning. Pytorch is the only option now
#         -> generate a warning if the argument '--finetuning_framework' is used, regardless of value (the argument will be removed in a future version)
#       - added option to disable the workaround for off-by-one mismatch between the sizes of predicted labels and ref. labels:
#         (20s of audio -> 1000 ref. labels, but only 999 predictions; currently I'm removing the last ref. label during dataset preprocessing)
#           a) by default (or with "--remove_last_label 1"), the last label for each wav file is removed, as before
#           b) "--remove_last_label 0" will keep all labels as they are
#       - fix crash when '--model_save_dir' is not set
#   2022-09-14
#       - incorporated edits by zzajic:
#           save loss via Tensorboard or Weights & Biases (optional, enabled via "--tensorboard_logging" and "--wandb_logging")
#           NOT TESTED YET
#       - learning rate and warmup are now configurable
#   2022-08-11
#       dataset caching is now enabled by default again. To disable it, use "--dataset_caching 0"
#       restored support for older versions of 'datasets' which do not have 'disable_caching' (pre-2.0)': 
#           if 'disable_caching' is not available, caching will remain enabled, but otherwise everything will function as before
#   2022-07-01
#       added option to disable dataset caching to avoid cluttering the disk with frequently changing custom datasets:
#           "--dataset_caching 1" to enable caching, "--dataset_caching 0" to disable it
#       Note: in this version, dataset caching was disabled by default, but it has since been reenabled (in the 2022-08-11 version)
#       (disabling caching requires a relatively recent version of 'datasets', probably 2.0 - 1.18.3 is too old, but 2.3.2 is definitely ok)
#   2022-06-20
#       added "--file_valid" for logging validation loss during training



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


# Note: at the time this code was originally written, transformers.Wav2Vec2ForAudioFrameClassification was incomplete
#   -> this adds the then-missing parts
class Wav2Vec2ForAudioFrameClassification_custom(transformers.Wav2Vec2ForAudioFrameClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        labels = labels.reshape(-1,1) # 1xN -> Nx1

        # ADDED
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                #loss = loss_fct(logits.squeeze(), labels.squeeze())
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




def load_json_dataset_forAudioFrameClassification(file_train,file_eval,file_valid=None):

    data_files = {}
    if file_train is not None:
        data_files["train"] = file_train
    if file_eval is not None:
        data_files["eval"] = file_eval
    if file_valid is not None:
        data_files["valid"] = file_valid

    #data_files = {"train": file_train, "eval": file_eval}
    phrasing_features = datasets.Features({'path': datasets.features.Value('string'), 'label': datasets.features.Sequence(datasets.features.Value(dtype='float64'))})
    dataset = datasets.load_dataset("json", data_files=data_files, features=phrasing_features)

    return dataset



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the main model - pretrained base (e.g. "facebook/wav2vec2-base") or a finetuned checkpoint')
    parser.add_argument('--feature_extractor_name', type=str, required=False, default=None, help='Path to the feature extractor model, if it is different from the main model (optional)')
    parser.add_argument('--file_train', type=str, required=False, default=None,help='Path to the training dataset (a JSON file)')
    parser.add_argument('--file_valid', type=str, required=False, default=None, help='Path to the validation dataset (a JSON file)')
    parser.add_argument('--file_eval', type=str, required=False, default=None, help='Path to the evaluation (test) dataset (a JSON file)')
    parser.add_argument('--num_epochs', type=int, required=False, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--file_output', type=str, required=False, default="output.txt", help='Path for the output file')
    parser.add_argument('--model_save_dir', type=str, required=False, default=None, help='Directory for saving the training log and the finetuned model')
    parser.add_argument('--max_duration', type=float, required=False, default=30.0, help='Maximum duration of audio files, default = 30s (must be >= duration of the longest file)')
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
    parser.add_argument('--remove_last_label', type=int, required=False, choices=[0,1], default=1, help='Remove the last value from ref. labels to match the number of predictions? 0 = no, 1 = yes (Default: yes)')
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

    if args.remove_last_label > 0:
        remove_extra_label = True # in a 20.0 s audio, there will be 1000 labels but only 999 logits -> remove the last label so the numbers match
    else: # if the labels are already fixed elsewhere
        remove_extra_label = False

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
    #  TODO: check if this is sufficient
    #   Note: I don't really care about reproducibility though - this is more to ensure *different* seeds when needed
    if seed is not None:
        set_seed(seed)
        torch.manual_seed(seed)

    freeze_feature_encoder = True
    freeze_base_model = False

    metric = load_metric("mse")

    dataset = load_json_dataset_forAudioFrameClassification(file_train,file_eval,file_valid)

    model = Wav2Vec2ForAudioFrameClassification_custom.from_pretrained(model_checkpoint, num_labels=1)

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
        #   (in case the sample rate of the labels is lower than it's supposed to be)

        # TODO: check if audio durations <= max_duration

        # TODO: check if it should be rounded up or down (down I think)

        for label in labels:
            for n in range(len(label), num_padded_labels):
                label.append(0)
            if remove_extra_label:
                label.pop()

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            padding='max_length', # pad to max_length, not just to the longest sequence
            max_length=int(sampling_rate * max_duration), 
            truncation=False,
        )

        inputs["label"] = labels

        return inputs

    def compute_metrics(eval_pred): # TODO: replace with something more suitable
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


    # -------------
    # process the train/val/test data
    # -------------

    if do_train:
        print("Processing training data...")
        processed_dataset_train = dataset["train"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_train = processed_dataset_train.rename_column("label", "labels")
        processed_dataset_train.set_format("torch", columns=["input_values", "labels"])
        train_dataloader = DataLoader(processed_dataset_train, shuffle=True, batch_size=batch_size)

    else:
        processed_dataset_train = None

    if do_validation:
        print("Processing validation data...")
        processed_dataset_valid = dataset["valid"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_valid = processed_dataset_valid.rename_column("label", "labels")
        processed_dataset_valid.set_format("torch", columns=["input_values", "labels"])
        valid_dataloader = DataLoader(processed_dataset_valid, shuffle=False, batch_size=batch_size)
    else:
        processed_dataset_valid = None

    if do_eval:
        print("Processing test data...")
        processed_dataset_test = dataset["eval"].map(preprocess_function, remove_columns=["audio","label"], batched=True)

        processed_dataset_test = processed_dataset_test.rename_column("label", "labels")
        processed_dataset_test.set_format("torch", columns=["input_values", "labels"])
        eval_dataloader = DataLoader(processed_dataset_test, batch_size=1)
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

        logfile.write("epoch,train loss,val loss\n")
        
        for epoch in range(num_epochs):

            model.train()

            # save checkpoint every N epochs
            if save_checkpoints and epoch > 0 and (epoch % epochs_between_checkpoints == 0):
                epoch_dir = os.path.join(model_save_dir,"epoch%d"%epoch)
                os.makedirs(epoch_dir, exist_ok=True)
                model.save_pretrained(epoch_dir)


            train_loss = 0
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

                if tensorboard_logging:
                    writerTensorboard.add_scalar("Loss_Train_individual",  loss.detach().item(), i+len(train_dataloader)*epoch)
                    #writerTensorboard.add_scalar("LearningRate_individual", optimizer.lr, i+len(train_dataloader)*epoch)
                    writerTensorboard.flush()

                if wandb_logging:
                    #wandb.log({"LearningRate_individual": optimizer.lr})
                    wandb.log({"Loss_Train_individual": loss.detach().item()})

            if do_validation:
                model.eval()

                val_loss = 0

                for i, batch in enumerate(valid_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    val_loss = val_loss + loss.detach().item()

                    if tensorboard_logging:
                        writerTensorboard.add_scalar("Loss_Val_individual", loss.detach().item(), i + len(valid_dataloader) * epoch)
                        writerTensorboard.flush()

                    if wandb_logging:
                        wandb.log({"Loss_Val_individual": loss.detach().item()})

                logfile.write("%d,%f,%f\n"%(epoch,train_loss,val_loss))
                logfile.flush()

                if tensorboard_logging:
                    writerTensorboard.add_scalars("Loss", {'Train':train_loss,'Val':val_loss}, epoch)
                    #writerTensorboard.add_scalar("LearningRate",optimizer.lr, epoch)
                    writerTensorboard.flush()

                if wandb_logging:
                    wandb.log({"Loss_Train": train_loss})
                    wandb.log({"Loss_Val": val_loss})
                    #wandb.log({"LearningRate": optimizer.lr})
                    # Optional
                    wandb.watch(model)

            else:
                logfile.write("%d,%f,N/A\n"%(epoch,train_loss))
                logfile.flush()

                if tensorboard_logging:
                    writerTensorboard.add_scalar("Loss_Train", train_loss, epoch)
                    #writerTensorboard.add_scalar("LearningRate", lr_scheduler.item() , epoch)
                    writerTensorboard.flush()

                if wandb_logging:
                    wandb.log({"Loss_Train": train_loss})

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

        

        for i, batch in enumerate(eval_dataloader):

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits

            if model.num_labels == 1:
                predictions = logits
            else:
                predictions = torch.argmax(logits, dim=-1)

            labels = batch["labels"].reshape(-1)
            predictions = predictions.reshape(-1)
            predictions_all.append(predictions.cpu().detach().numpy())

            metric.add_batch(predictions=predictions, references=labels)

            progress_bar.update(1)

        progress_bar.close()
        score = metric.compute()

        print(score)

        with open(file_output, 'w') as file:
            for ii,prediction in enumerate(predictions_all):
                file.write(dataset["eval"][ii]["audio"]["path"])
                file.write(",[")
                prediction.tofile(file,sep=" ", format="%s")
                file.write("]\n")


if __name__ == '__main__':
    main()


