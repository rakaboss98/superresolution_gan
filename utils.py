import torch
import boto3
from train_config import training_parameteres
from cloudpathlib import CloudPath
import os

def save_checkpoint(model, optimizer, filename):
    print('==> Saving checkpoint')
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, device):
    print('==> Loading checkpoint')
    ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])

def downloadDataFroms3():
    bucketName = training_parameteres['bucket_name']
    remoteDirectoryName = training_parameteres['high_res_optical']

    s3_resource = boto3.resource('s3', aws_access_key_id = training_parameteres['access_key'],
    aws_secret_access_key=training_parameteres['secret_access_key'])
    bucket = s3_resource.Bucket(bucketName) 
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        try:
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key) # save to same path
        except:
            pass

    remoteDirectoryName = training_parameteres['low_res_optical']
    s3_resource = boto3.resource('s3', aws_access_key_id = training_parameteres['access_key'],
    aws_secret_access_key=training_parameteres['secret_access_key'])
    bucket = s3_resource.Bucket(bucketName) 
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        try:
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key) # save to same path
        except:
            pass
    low_res_data = os.path.dirname(obj.key)+'/'
    

def downloadCheckpointsFroms3(ckpt_name):
    client = boto3.client('s3', aws_access_key_id = training_parameteres['access_key'], 
    aws_secret_access_key=training_parameteres['secret_access_key'])
    bucket_name = training_parameteres['bucket_name']
    folder = training_parameteres['checkpoints']
    client.download_file(bucket_name, folder+ckpt_name, 'checkpoints/'+ckpt_name )

def uploadCheckpointsTos3(file):
    client = boto3.client('s3', aws_access_key_id = training_parameteres['access_key'], 
    aws_secret_access_key=training_parameteres['secret_access_key'])
    bucket_name = training_parameteres['bucket_name']
    folder = training_parameteres['checkpoints']
    client.upload_file('checkpoints/'+file, bucket_name, folder+file) 

def downloadSamples(file):
    client = boto3.client('s3', aws_access_key_id = training_parameteres['access_key'], 
    aws_secret_access_key=training_parameteres['secret_access_key'])
    bucket_name = training_parameteres['bucket_name']
    folder = training_parameteres['samples']
    client.download_file(bucket_name, folder+file, 'samples/'+file)

def uploadSamples(file):
    client = boto3.client('s3', aws_access_key_id = training_parameteres['access_key'], 
    aws_secret_access_key=training_parameteres['secret_access_key'])
    bucket_name = training_parameteres['bucket_name']
    folder = training_parameteres['samples']
    client.upload_file('samples/'+file, bucket_name, folder+file) 




# def uploadLogsTos3(file):



if __name__ == '__main__':
    # downloadDataFroms3()
    # downloadCheckpointsFroms3('gen_yd2x.pt')
    # uploadCheckpointsTos3('gen_yd2x.pt')
    downloadSamples('raw.tif')
    uploadSamples('raw.tif')
