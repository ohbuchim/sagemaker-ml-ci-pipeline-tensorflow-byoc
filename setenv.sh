ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
# REGION=$(python setenv.py config region)
