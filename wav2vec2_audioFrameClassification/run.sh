# Example script for audio frame classification

# Training only
python3 wav2vec2_audioFrameClassification.py \
    --model_checkpoint facebook/wav2vec2-base \
    --feature_extractor_name facebook/wav2vec2-base \
    --file_train /path/to/train_data.json \
    --file_valid /path/to/valid_data.json \
    --num_epochs 5 \
    --max_duration 30 \
    --model_save_dir ../experiments/new_model \
    --mode train


# Testing only
python3 wav2vec2_audioFrameClassification.py \
    --model_checkpoint ../experiments/new_model \
    --feature_extractor_name facebook/wav2vec2-base \
    --file_eval /path/to/test_data.json \
    --max_duration 30 \
    --model_save_dir ../experiments/new_model \
    --file_output ../experiments/new_model/output.txt \
    --mode test


# Both
python3 wav2vec2_audioFrameClassification.py \
    --model_checkpoint facebook/wav2vec2-base \
    --feature_extractor_name facebook/wav2vec2-base \
    --file_train /path/to/train_data.json \
    --file_valid /path/to/valid_data.json \
    --file_eval /path/to/test_data.json \
    --num_epochs 5 \
    --max_duration 30 \
    --model_save_dir ../experiments/new_model2 \
    --file_output ../experiments/new_model2/output.txt \
    --mode both
