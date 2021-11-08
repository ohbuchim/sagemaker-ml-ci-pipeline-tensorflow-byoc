ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
ECR_REPOGITORY_PREFIX=$(python setenv.py config image-name-prefix)
JOB_NAME=$(python setenv.py config job-name)
echo "ddddd ${JOB_NAME}"
echo "aaaaa ${ACCOUNT_ID}"
# REGION=$(python setenv.py config region)
