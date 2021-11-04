ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
# preprocess
INPUT_DATA_PATH=$(python setenv.py preprocess input-data-path)
OUTPUT_DATA_PATH=$(python setenv.py preprocess input-data-path)
# train
TRAIN_JOB_NAME=$(python setenv.py train train-job-name)
TRAIN_IMAGE_URL=$(python setenv.py train train-image-url)
TRAIN_DATA_PATH=$(python setenv.py train data-path)
BATCH_SIZE=$(python setenv.py train hyper-parameters batch-size)
EPOCH=$(python setenv.py train hyper-parameters epoch)
# evaluate
MODEL_PATH=$(python setenv.py evaluate model-path)
EVALUATE_RESULT=$(python setenv.py evaluate evaluate-result)