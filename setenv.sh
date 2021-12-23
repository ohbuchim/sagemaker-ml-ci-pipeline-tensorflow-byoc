ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
ECR_REPOGITORY_PREFIX=$(python setenv.py config image-name-prefix)
TRAIN_JOB_NAM=$(python setenv.py config image-name-prefix)-train-${EXEC_ID}
PREP_JOB_NAME=$(python setenv.py config image-name-prefix)-prep-${EXEC_ID}
EVAL_JOB_NAME=$(python setenv.py config image-name-prefix)-train-${EXEC_ID}
# REGION=$eval
# REGION=$(python setenv.py config region)
