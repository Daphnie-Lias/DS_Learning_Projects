"""Package Imports"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
from collections import OrderedDict
import time
from PIL import Image
import argparse

"""Function Definitions"""

def arg_parser():
    
    parser = argparse.ArgumentParser (description = "Neural Network Training Parser")
    parser.add_argument ('data_dir', help = 'Default flowers directory provided inside code. Optionalargument ', type = str)
    parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
    parser.add_argument ('--arch', help = 'Default is vgg16,otherwise alternate option densenet should be given as argument', type = str)
    parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 512', type = int)
    parser.add_argument ('--epochs', help = 'Number of epochs or passes , default 5', type = int)
    parser.add_argument ('--GPU', help = "Specify GPU or default mode is cpu, optional argument", type = str)
    parser.add_argument('--learning_rate',type=float,help='Define gradient descent learning rate as float. Default 0.001. Optional argument')
#     This implementation accepts 2 pre trained models (vgg16 : default and alexnet)
#     Sample Input:
#     python train.py data_dir --arch ‘alexnet' --learning_rate 0.01 --hidden_units 512 --epochs 3
    args = parser.parse_args()
    return args


"""Training Data Augmentation"""
def train_transform(train_dir):
     
    data_transforms_train = transforms.Compose ([transforms.RandomRotation (30),
                                             transforms.RandomResizedCrop (224),
                                             transforms.ColorJitter(),
                                             transforms.RandomHorizontalFlip (),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

    # Load the Data using Image Folder
    """Data Loading"""
    image_datasets_train = dset.ImageFolder (train_dir, transform = data_transforms_train)
    return image_datasets_train

def valid_transform(valid_dir):
   
    data_transforms_valid = transforms.Compose ([transforms.Resize (255),
                                             transforms.CenterCrop (224),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

    # Load the Data using Image Folder
    image_datasets_valid = dset.ImageFolder (valid_dir, transform = data_transforms_valid)
    return image_datasets_valid

def test_transform(test_dir):
   
    data_transforms_test = transforms.Compose ([transforms.Resize (255),
                                             transforms.CenterCrop (224),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

    # Load the Data using Image Folder
    image_datasets_test = dset.ImageFolder (test_dir, transform = data_transforms_test)
    return image_datasets_test


"""Data Batching"""
def data_loader(train_dir,valid_dir,test_dir):
   
    data_loader_train = torch.utils.data.DataLoader(train_transform(train_dir), batch_size = 64, shuffle = True)
    data_loader_valid = torch.utils.data.DataLoader(valid_transform(valid_dir), batch_size = 64, shuffle = True)
    data_loader_test = torch.utils.data.DataLoader(test_transform(test_dir), batch_size = 64, shuffle = True)

    image_datasets = [train_transform(train_dir), valid_transform(valid_dir), test_transform(test_dir)]
    dataloaders = [data_loader_train, data_loader_valid, data_loader_test]
    return dataloaders,image_datasets


def check_gpu(gpu):
    #defining device: either cuda or cpu
#     If gpu parameter is not specified 
    if not gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    elif gpu:
        device = gpu
    
    return device
    
    
#Load predefined models based ona arch argument
"""Usage of Pre-trained network"""
def model_loader(architecture):
    
    if type(architecture) == type(None):
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Setting default Network architecture as vgg16.")
    elif architecture == 'densenet':
        model = models.densenet161(pretrained=True)
        model.name = "densenet161"
        print("Network architecture specified as densenet161")
    else:
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Setting default Network architecture as vgg16.")
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model
    


"""Usage of Feedforward Classifier"""
def classifier(model,hidden_units,architecture):
    if type(hidden_units) == type(None): 
        hidden_units = 4096 #hyperparamters
    
    # Find Input Layers
    if type(architecture) == type(None):
#         Loads input features for vgg architecture
        input_features = model.classifier[0].in_features 
    elif architecture == 'densenet':
#         densenet part
        input_features = model.classifier.in_features
    else:
        input_features = model.classifier[0].in_features 

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier
   



"""Testing Accuracy"""
def test_accuracy(model, test_loader,device):
    model.to (device)
    model.eval()
    accuracy = 0
    pass_count = 0
    accuracy = 0
    
    start = time.time()
    print('Testing  started')
    for data in test_loader:
        pass_count += 1
        images, labels = data   
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Testing Accuracy: {:.4f}".format(accuracy/pass_count))
    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))


# Function network_trainer represents the training of the network model
"""Training and Validatin Accuracy and Accuracy"""
def network_trainer(model, dataloaders,criterion,optimizer,device, epochs):
    
    running_loss = 0
    accuracy = 0
    
    if type(epochs) == type(None):
        epochs = 5
        print("Number of Epochs specificed as 5.")    
   
    start = time.time()
    print('Training started')

    for e in range(epochs):
    
        train_mode = 0
        valid_mode = 1
    
        for mode in [train_mode, valid_mode]:   
            if mode == train_mode:
                model.train()
            else:
                model.eval()

            pass_count = 0

            for data in dataloaders[mode]:
                pass_count += 1
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                # Forward
                output = model.forward(inputs)
                loss = criterion(output, labels)
                # Backward
                if mode == train_mode:
                    loss.backward()
                    optimizer.step()                

                running_loss += loss.item()
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()

            if mode == train_mode:
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                  "Accuracy: {:.4f}".format(accuracy))

            running_loss = 0

    time_elapsed = time.time() - start
    print("\nTotal time for training: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

    return model

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
"""Saving Model"""
def initial_checkpoint(model, Train_data,save_dir):
    
    model.class_to_idx = Train_data.class_to_idx
    checkpoint = {'architecture': model.name,
             'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}

    if type(save_dir) == type(None):
        
        torch.save (checkpoint, 'checkpoint.pth')
        
    else:
        torch.save (checkpoint, save_dir + '/checkpoint.pth')
    
    print('Checkpoint saved')
       
            
def main():
         
    #Get inputs as argument parameters
    args = arg_parser()
    
    # Set data directory
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Invoke transform functions
    train_data = train_transform(train_dir)
    valid_data = valid_transform(valid_dir)
    test_data =  test_transform(test_dir)
    
    #Load data through ImageFolder
    dataloaders,image_datasets = data_loader(train_dir,valid_dir,test_dir)
   
    # Load Model
    model = model_loader(architecture=args.arch)
    
    # Build Classifier
    model.classifier = classifier(model,hidden_units=args.hidden_units,architecture=args.arch)
     
    # Check for GPU
    device = check_gpu(gpu=args.GPU)
    
    # Send model to device
    model.to(device);
    
    # Set learning rate
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
    else: learning_rate = args.learning_rate
        
    # Define loss and optimizer
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)    
         
    # Train the classifier layers 
    trained_model = network_trainer(model,dataloaders,criterion,optimizer,device,args.epochs)
    
    test_accuracy(model, dataloaders[2],device)
    
    # Save the model as checkpoint
    
    initial_checkpoint(trained_model, image_datasets[0],args.save_dir)


# =============================================================================
# Run Program
  
# =============================================================================
if __name__ == '__main__': main()            
