# Example script for multitask audio frame classification

# Training only
python3 wav2vec2_audioFrameClassification.py \
    --model_checkpoint facebook/wav2vec2-base \
    --feature_extractor_name facebook/wav2vec2-base \
    --file_train /path/to/multitask_train_data.json \
    --file_valid /path/to/multitask_valid_data.json \
    --num_epochs 5 \
    --max_duration 20 \
    --batch_size 1 \
    --model_save_dir ../experiments/new_multitask_model \
    --mode train \
    --multitask

# Testing only
python3 wav2vec2_audioFrameClassification.py \
    --model_checkpoint ../experiments/new_model \
    --feature_extractor_name facebook/wav2vec2-base \
    --file_eval /path/to/multitask_test_data.json \
    --max_duration 20 \
    --model_save_dir ../experiments/new_multitask_model \
    --file_output ../experiments/new_multitask_model/output.txt \
    --mode test \
    --multitask


# Both
python3 wav2vec2_audioFrameClassification.py \
    --model_checkpoint facebook/wav2vec2-base \
    --feature_extractor_name facebook/wav2vec2-base \
    --file_train /path/to/multitask_train_data.json \
    --file_valid /path/to/multitask_valid_data.json \
    --file_eval /path/to/multitask_test_data.json \
    --num_epochs 5 \
    --max_duration 20 \
    --batch_size 1 \
    --model_save_dir ../experiments/new_multitask_model2 \
    --file_output ../experiments/new_multitask_model2/output.txt \
    --mode both \
    --multitask
