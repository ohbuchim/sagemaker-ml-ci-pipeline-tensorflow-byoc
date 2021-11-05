import argparse
import boto3
import logging
import os
import uuid
import yaml

import stepfunctions

import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

from stepfunctions import steps
from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow

stepfunctions.set_stream_logger(level=logging.INFO)
id = uuid.uuid4().hex
config_name = 'flow.yaml'

with open(config_name) as file:
    config = yaml.safe_load(file)
    test_param = config['train']['train-job-name']
    print('------------------')
    print(test_param)


# REGION='us-east-1'
# BUCKET='sagemaker-us-east-1-420964472730'
# FLOW_NAME='flow_{}'.format(id) 
# TRAINING_JOB_NAME='sf-train-{}'.format(id) # JobNameの重複NGなのでidを追加している
# SAGEMAKER_ROLE = 'arn:aws:iam::420964472730:role/service-role/AmazonSageMaker-ExecutionRole-20201204T095531'
# WORKFLOW_ROLE='arn:aws:iam::420964472730:role/StepFunctionsWorkflowExecutionRole'

# def create_estimator():
#     hyperparameters = {'batch_size': args.batch_size,'epochs': args.epoch}
#     output_path = 's3://{}/output'.format(BUCKET)
#     estimator = Estimator(
#                         image_uri=args.train_url,
#                         role=SAGEMAKER_ROLE,
#                         hyperparameters=hyperparameters,
#                         train_instance_count=1,
#                         train_instance_type='ml.p2.xlarge',
#                         output_path=output_path)
#     return estimator


# if __name__ == '__main__':
#     # flow.yaml の定義を環境変数経由で受け取る
#     # buildspec.yaml の ENV へ直接書き込んでも良いかも
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input-data-path', type=str, default=os.environ['INPUT_DATA_PATH'])
#     parser.add_argument('--processed-data-path', type=str, default=os.environ['PROCESSED_DATA_PATH'])
#     parser.add_argument('--train-job-name', type=str, default=os.environ['TRAIN_JOB_NAME'])
#     parser.add_argument('--train-image-url', type=str, default=os.environ['TRAIN_IMAGE_URL'])
#     parser.add_argument('--batch-size', type=str, default=os.environ['BATCH_SIZE'])
#     parser.add_argument('--epoch', type=str, default=os.environ['EPOCH'])
#     parser.add_argument('--model-path', type=str, default=os.environ['MODEL_PATH'])
#     parser.add_argument('--evaluate-data-path', type=str, default=os.environ['EVAL_DATA_PATH'])
#     parser.add_argument('--evaluate-result-path', type=str, default=os.environ['EVALUATE_RESULT'])
#     args = parser.parse_args()

#     # SFn の実行に必要な情報を渡す際のスキーマを定義します
#     schema = {'TrainJobName': str}
#     execution_input = ExecutionInput(schema=schema)

#     # SFn のワークフローの定義を記載します
#     inputs = {'TrainJobName': TRAINING_JOB_NAME}

#     # SageMaker の学習ジョブを実行するステップ
#     estimator = create_estimator()
#     data_path = {'train': args.data_path}

#     training_step = steps.TrainingStep(
#         'Train Step', 
#         estimator=estimator,
#         data=data_path,
#         job_name=execution_input['TrainJobName'],  
#         wait_for_completion=True
#     )

#     # 各 Step を連結
#     chain_list = [training_step]
#     workflow_definition = steps.Chain(chain_list)

#     # Workflow の作成
#     workflow = Workflow(
#         name=FLOW_NAME,
#         definition=workflow_definition,
#         role=WORKFLOW_ROLE,
#         execution_input=execution_input
#     )
#     workflow.create()

#     # Workflow の実行
#     execution = workflow.execute(inputs=inputs)