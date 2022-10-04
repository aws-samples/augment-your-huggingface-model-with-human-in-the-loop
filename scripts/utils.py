import argparse
import boto3
import logging
import os
from botocore.exceptions import ClientError
import tarfile
import zipfile
from sagemaker import ModelPackage
from time import gmtime, strftime

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return approved_packages[0]
        # return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
        
        
def deploy_or_update_endpoint(endpoint_name, model_package_arn, role, sagemaker_session, instance_type="ml.m5.xlarge"):
    """Deploys or redeploys a sagemaker endpoint.

    Args:
        endpoint_name: The name of the endpoint
        model_package_group_name: The model package group name.
        role: IAM execution role to be used with the sagemaker model

    Returns:
        None
    """
    
    
    if len(sm_client.list_endpoints(NameContains=endpoint_name)["Endpoints"]) \
        and (sm_client.list_endpoints(NameContains=endpoint_name)["Endpoints"][0]["EndpointName"]==endpoint_name):
        
        # Redeployment of existing model
        print("Endpoint already exists, updating existing enpoint with latest model")
        container_list = [{'ModelPackageName': model_package_arn}]
        
        model_name = 'nlp-summit-model' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        endpoint_config_name = 'nlp-summit-endpoint-config' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        
        create_model_response = sm_client.create_model(
            ModelName = model_name,
            ExecutionRoleArn = role,
            Containers = container_list
        )
        
        create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName = endpoint_config_name,
            ProductionVariants=[{
                'InstanceType': instance_type,
                'InitialInstanceCount': 1,
                'InitialVariantWeight': 1,
                'ModelName': model_name,
                'VariantName': 'AllTraffic'}])
        
        update_endpoint_response = sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
    else:
        # First deployment
        model = ModelPackage(
            role=role, model_package_arn=model_package_arn, sagemaker_session=sagemaker_session
        )

        print("EndpointName= {}".format(endpoint_name))
        model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint_name)