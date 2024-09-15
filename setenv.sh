ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
ECR_REPOGITORY_PREFIX=$(python setenv.py config image-name-prefix)
TRAIN_JOB_NAME=$(python setenv.py config image-name-prefix)-train-${EXEC_ID}
PREP_JOB_NAME=$(python setenv.py config image-name-prefix)-prep-${EXEC_ID}
EVAL_JOB_NAME=$(python setenv.py config image-name-prefix)-eval-${EXEC_ID}
TRAINED_MODEL_S3=$(python setenv.py train output-path)/${TRAIN_JOB_NAME}/output/model.tar.gz
EVAL_REPORT_S3=$(python setenv.py evaluate result-path)/${EVAL_JOB_NAME}/evaluation.json
MLFLOW_SERVER_URI=$(python setenv.py experiments mlflow-server-uri)
# REGION=$eval
# REGION=$(python setenv.py config region)

echo "1st"
echo ${EXEC_ID}
echo "2nd"
echo $EXEC_ID
