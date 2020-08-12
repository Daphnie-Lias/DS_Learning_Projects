"""Package Imports"""
import argparse
from PIL import Image
import torch
import numpy as np
from train import check_gpu
from train import train_transform
from torch import nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import json

def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Prediction")

    # Point towards image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to impage file for prediction. like "flowers/train/13/image_05765.jpg"',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=False)
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to labels')
    
    # Add GPU Option to parser
    parser.add_argument('--GPU', 
                        help='Option to use GPU', type = str)
    
#     Sample Input:
# 	python predict.py --image "flowers/train/13/image_05765.jpg"  --top_k 3
    # Parse args
    args = parser.parse_args()
    
    return args

# Function load_checkpoint(checkpoint_path) loads our saved deep learning model from checkpoint
"""Loading Checkpoints"""
def load_checkpoint(checkpoint_path,device,GPU):
#     Override if checkpoint not specified
    if type(checkpoint_path) == type(None):
       checkpoint_path = 'checkpoint.pth'
    
# If GPU enabled as argument    
    if  GPU:
        device = torch.device("cuda:0")
        checkpoint =torch.load(checkpoint_path,map_location=lambda storage, loc: storage)
#  If GPU disabled , use CPU mode           
    elif not GPU:
        device = torch.device("cpu")
        checkpoint =torch.load(checkpoint_path)
# If no option specified, run in GPU by default        
    else:
        torch.cuda.set_device(0) 
        checkpoint =torch.load(checkpoint_path)
        
    # Load Defaults if none specified
    if checkpoint ['architecture'] == 'densenet161':
        model = models.densenet161(pretrained=True)
        model.name = "densenet161"
        print("Checkpoint Loading model densenet161")
    
    else: 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Checkpoint Loading model vgg16")
    
     # Load stuff from checkpoint
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx'] 
    model.load_state_dict(checkpoint['state_dict'])
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    return model


"""Image Processing"""
def process_image(image_path):
    test_image = Image.open(image_path)

    # Get original dimensions
    orig_width, orig_height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

"""Class Prediction"""
def predict(image_path, model,cat_to_name,topk,device):
    
    if type(topk) == type(None):
        topk = 3
    model = model.to(device)    
    # Set model to evaluate
    model.eval();
    image = process_image(image_path)  
    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()   
    image = image.to(device)   
    output = model.forward(image)
    
    with torch.no_grad():
        image.unsqueeze_(0) 
        probabilities=torch.exp(output)
        top_probs, top_labs = probabilities.topk(topk)    
        top_probs = top_probs.cpu().numpy().tolist()[0] 
        top_labs = top_labs.cpu().numpy().tolist()[0]
        # Convert indices to classes
        idx_to_class = {val: key for key, val in  model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_labs]
        top_flowers = [cat_to_name [item] for item in top_labels]
    
    for output_index in range (topk):
     print("K most classes Probability: {:.3f}..% ".format(top_probs [output_index]*100),
            "Rank: {}.. ".format(top_labels [output_index]),
            "Class name: {}.. ".format(top_flowers [output_index]))

    return top_probs, top_labels


def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Pre load categories to names json file
    if type(args.category_names) == type(None):
        category_names = 'cat_to_name.json'
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Check for GPU
    device = check_gpu(gpu=args.GPU);

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint,device,GPU=args.GPU)
    
    # Process Image
    image_tensor = process_image(args.image)
    
    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_index = predict(args.image, model,cat_to_name,args.top_k,device)
   
    
    
   
# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()    
    

