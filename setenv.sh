ECR_REPOGITORY_PREFIX=$(python setenv.py config image-name-prefix)
JOB_NAME=$(python setenv.py config job-name)
ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
TRAIN_IMAGE_URL=$(python setenv.py train image-uri)
python --version
echo "ddddd ${TRAIN_IMAGE_URL}"
echo "aaaaa ${ACCOUNT_ID}"
# REGION=$(python setenv.py config region)
