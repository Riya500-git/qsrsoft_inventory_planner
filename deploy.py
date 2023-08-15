import boto3
from sagemaker import image_uris
#from sagemaker.model import Model
from sagemaker.pytorch.model import PyTorchModel

def upload_model_to_s3(model_file, bucket, key):
    """Uploads a model file to an S3 bucket."""

    # Create the S3 client.
    s3 = boto3.client('s3')

    # Upload the model file to the S3 bucket.
    s3.upload_file(model_file, bucket, key)

def deploy_model_to_sagemaker(bucket, key):
    """Deploys a model to Sagemaker."""

    # Create the SageMaker client.
    sagemaker_role = "arn:aws:iam::178528245441:role/stevwang-sagemaker"
    model_url = 's3://{}/{}'.format(bucket, key)
    instance_type="ml.m4.xlarge"
    container = image_uris.retrieve(region='us-east-1',
                            framework='pytorch',
                            version="1.6",
                            image_scope="inference",
                            py_version="py3",
                            instance_type=instance_type)
    model = PyTorchModel(image_uri=container, 
              model_data=model_url,
              role=sagemaker_role)
    model.deploy(initial_instance_count=1,
                  instance_type=instance_type,
                  endpoint_name="stevwang-inventory-predictor")

    print('The model has been deployed to Sagemaker!')

if __name__ == '__main__':
    model_file = './model.tar.gz'

    # Get the S3 bucket and key for the model file.
    bucket = 'inventory-planning-stevwang'
    key = 'model.tar.gz'

    upload_model_to_s3(model_file, bucket, key)

    # Deploy the model to Sagemaker.
    deploy_model_to_sagemaker(bucket, key)
                              
