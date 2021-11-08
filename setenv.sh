ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
ECR_REPOGITORY_PREFIX=$(python setenv.py config image-name-prefix)
python setenv.py config image-name-prefix
echo "aaaaa ${ECR_REPOGITORY_PREFIX}"
# REGION=$(python setenv.py config region)
