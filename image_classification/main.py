"""
Guideline of your submission of HW3.
If you have any questions in regard to submission,
please contact TA: Ma Zhiyuan <e0983565@u.nus.edu>
"""
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
from torchmetrics import Accuracy
import optuna
import random
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

global_best_accuracy = 0
global_history = []

#function is referenced from https://www.kaggle.com/code/toygarr/resnet-implementation-for-image-classification
# Check the availability of the device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # mps is the apple's GPU acceleration
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
# Function to handle the data transfer between device according to the data type
def to_device(data,device):
    if isinstance(data,(list,tuple)):
        # if the data is a list or tuple, transfer each element to the device
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

# Transfer the data to the correct device
class ToDeviceLoader:
    def __init__(self,data,device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch,self.device)

    def __len__(self):
        return len(self.data)
    
# helper function to get the mean and std of the dataset
def get_mean_std(path):

    # Basic transformation for the dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # make sure the image is in grayscale
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Load the dataset from the folder
    dataset = datasets.ImageFolder(root= path, transform=transform)

    device = get_device()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataset = ToDeviceLoader(dataloader, device)

    # Initialize the sums and the number of batches
    sum = 0.0
    sum_squared = 0.0
    num_batches = 0

    # Loop through the data in the DataLoader
    for data, _ in  dataset:
        sum += data.mean()
        sum_squared += (data ** 2).mean()
        num_batches += 1

    # Divide by the total number of batches to get the mean
    mean = sum / num_batches

    # Calculate the std deviation
    std = (sum_squared / num_batches - mean ** 2) ** 0.5

    return mean, std

#fuction is referenced from https://discuss.pytorch.org/t/what-is-the-simplest-way-to-change-class-to-idx-attribute/86778/4
# Set correct label to index mapping by inheriting and overriding the ImageFolder class
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
    
    def find_classes(self, dir):
        """
        Override this method to load from setting file instead of scanning directory
        """
        # True labels to index mapping
        labels = {
            'bedroom': 1,
            'Coast': 2,
            'Forest': 3,
            'Highway': 4,
            'industrial': 5,
            'Insidecity': 6,
            'kitchen': 7,
            'livingroom': 8,
            'Mountain': 9,
            'Office': 10,
            'OpenCountry': 11,
            'store': 12,
            'Street': 13,
            'Suburb': 14,
            'TallBuilding': 15
        }

        classes = list(labels.keys())
        classes_to_idx = labels
        return classes, classes_to_idx

#function is referenced from https://stackoverflow.com/questions/74920920/pytorch-apply-data-augmentation-on-training-data-after-random-split
# Apply the transformation separately for training and validation dataset
class TrDataset(Dataset):
    def __init__(self, base_dataset, transformations):
        super(TrDataset, self).__init__()
        self.base = base_dataset
        self.transformations = transformations

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transformations(x), y
    
def load_data(mean, std, path, batch_size, train_size = 0.8):
    # Transformations for the training data with augmentation
    train_transform = transforms.Compose([
    # # Data augmentation
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0)),
    transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.Resize((256, 256)),       # Resize images to 256x256
    transforms.Grayscale(num_output_channels=1),  # Ensure there is only one channel
    transforms.ToTensor(),               # Convert images to PyTorch tensors
    # Normalize the images
    transforms.Normalize(mean= [mean], std= [std]),  # Use the mean and std dev for grayscale images
    # Random erasing to avoid overfitting
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0)
    ])

    # Transformations for the validation data without augmentation
    validation_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    
    # load the overall dataset with the true labels
    full_dataset = CustomImageFolder(root=path)

    # Split dataset into training and validation sets by 80-20 ratio
    train_size = int(train_size * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    
    if train_size != 0:
        raw_train_dataset, raw_validation_dataset = random_split(full_dataset, [train_size, validation_size])

        # Create custom datasets with the respective transformations
        train_dataset = TrDataset(raw_train_dataset, train_transform)
        validation_dataset = TrDataset(raw_validation_dataset, validation_transform)

        # Data loaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    else:
        # if train_size is 0, then only validation set is returned
        validation_dataset = TrDataset(full_dataset, validation_transform)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        train_loader = None
        
    return train_loader, validation_loader

#function is referenced from https://www.kaggle.com/code/toygarr/resnet-implementation-for-image-classification
# Helper function to add a convolutional layer and a batch normalization layer
def conv_shortcut(in_channel, out_channel, stride):
    layers = [nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(stride, stride)),
             nn.BatchNorm2d(out_channel)]
    return nn.Sequential(*layers)

# Helper function to add a single block for the ResNet model
def block(in_channel, out_channel, k_size,stride):
    layers = None

    first_layers = [nn.Conv2d(in_channel,out_channel[0], kernel_size=(1,1),stride=(stride,stride)),
                    nn.BatchNorm2d(out_channel[0]),
                    nn.ReLU()
                    ]

    second_layers = [nn.Conv2d(out_channel[0], out_channel[1], kernel_size=(k_size, k_size), stride=(1,1), padding=1),
                    nn.BatchNorm2d(out_channel[1]),
                    nn.ReLU()
                    ]

    layers = first_layers + second_layers

    return nn.Sequential(*layers)


class ResNet(nn.Module):

    def __init__(self, in_channels, num_classes, stage2_loop, stage3_loop, dropout1, dropout2, out_channels):
        super().__init__()
        
        # assign the external parameters to the class variables, for easier adjustment of the model hyperparameters
        self.stage2_loop = stage2_loop
        self.stage3_loop = stage3_loop
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.out_channels = out_channels

        self.stg1 = nn.Sequential(
                                   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3),
                                             stride=(1), padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))

        ##stage 2
        self.convShortcut2 = conv_shortcut(out_channels,out_channels*2,1)

        self.conv2 = block(out_channels,[out_channels//2,out_channels*2],3,1)
        self.ident2 = block(out_channels*2,[out_channels,out_channels*2],3,1)


        # ##stage 3
        self.convShortcut3 = conv_shortcut(out_channels*2,out_channels*4,2)

        self.conv3 = block(out_channels*2,[out_channels,out_channels*4],3,2)
        self.ident3 = block(out_channels*4,[out_channels*2,out_channels*4],3,1)

        ##Classify
        self.classifier = nn.Sequential(
                                       nn.AvgPool2d(2),
                                       nn.Dropout(self.dropout1),
                                       nn.Flatten(),
                                       nn.Dropout(self.dropout2),
                                       nn.LazyLinear(num_classes))

    def forward(self,inputs):
        #stage 1
        out = self.stg1(inputs)

        #stage 2
        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        for _ in range(self.stage2_loop):
            out = F.relu(self.ident2(out) + out)

        # Stage 3
        out = F.relu(self.conv3(out) + self.convShortcut3(out))
        for _ in range(self.stage3_loop):
            out = F.relu(self.ident3(out) + out)

        #Classify
        out = self.classifier(out)

        return out
    
#function is referenced from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
# Early stopping class to stop the training if the validation loss does not improve
class EarlyStopping:
    def __init__(self, patience=7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stop = False

    # Check if the validation loss has improved
    def __call__(self, val_loss):
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
    # If the validation loss does not improve, increment the counter
        elif val_loss > self.min_validation_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # If the counter reaches the patience, stop the training
            if self.counter >= self.patience:
                self.early_stop = True

# evalueate the model on the validation set
def evaluate(model, accuracy, criterion, val_loader):
    model.eval()
    val_loss = 0
    validation_loss = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f'Validation'):
            output = model(data)
            val_loss = criterion(output, target - 1).detach() #target - 1 because the labels start from 1
            validation_loss.append(val_loss)
            preds = torch.argmax(output, dim=1)  # Get the predicted class labels
            accuracy(preds , target-1)  # Update the running accuracy
    
    # Compute the overall accuracy and loss
    val_acc = accuracy.compute()
    accuracy.reset()
    overall_loss = torch.stack(validation_loss).mean().item()

    return val_acc, overall_loss

def fit(model, train_loader, val_loader, optimizer, n_epochs, early_stopping, criterion, grad_clip, scheduler, accuracy, model_dir):
    history = []
    local_best_accuracy = 0

    for epoch in range(n_epochs):

        result = {}
        train_loss = []

        model.train()
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target - 1) # Subtract 1 from the target since the labels start from 1
            train_loss.append(loss)
            loss.backward()
            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            # Update the weights
            optimizer.step()
            scheduler.step()

        result['train_loss'] = torch.stack(train_loss).mean().item()
        
        # Evaluate the model on the validation set,call the evaluate function
        val_acc, result['val_loss'] = evaluate(model, accuracy, criterion, val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        result['val_acc'] = val_acc
        history.append(result)
        
        # Print the results of the epoch at the end
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, current_lr, result['train_loss'], result['val_loss'], result['val_acc']))
        
        # local best accuracy for the current trial when running the auto-hyperparameter optimization
        if val_acc > local_best_accuracy:
            local_best_accuracy = val_acc
        
        # global best accuracy is the best accuracy across all trials
        global global_best_accuracy
        
        # Save the model if the validation accuracy is better than the previous best global accuracy
        if val_acc > global_best_accuracy:
          global_best_accuracy = val_acc
          torch.save(model, model_dir)
          print(f'Saved new best model with accuracy: {val_acc}%')

        # Call early stopping to check if the training should be stopped
        early_stopping(result['val_loss'])
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return history, local_best_accuracy


# Optuna objective function to optimize the hyperparameters
def objective(trial, train_data_dir, model_dir):
    n_epochs = 300
    num_classes = 15
    patience = 20
    channels = 1
    path = train_data_dir
    mean, std = get_mean_std(train_data_dir)

    # Hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-4, 1e-1,log = True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1,log = True)
    grad_clip = trial.suggest_float('grad_clip', 0.1, 5.0)
    state1_loop = trial.suggest_int('state1_loop', 0, 5)
    state2_loop = trial.suggest_int('state2_loop', 0, 5)
    dropout1 = trial.suggest_float('dropout1', 0, 0.5)
    dropout2 = trial.suggest_float('dropout2', 0, 0.5)
    out_channels = trial.suggest_categorical('out_channels', [8, 16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    # Create the model with the optimizable hyperparameters
    model = ResNet(channels, num_classes, state1_loop, state2_loop, dropout1, dropout2, out_channels)
    
    # Choose the optimizer based on the hyperparameter
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr,weight_decay = weight_decay)

    # Load the data and create the data loaders
    train_loader, validation_loader = load_data(mean, std,path,batch_size)

    # Move model and dataset to GPU/CPU by checking the availability of the device
    device = get_device()

    train_loader = ToDeviceLoader(train_loader, device)
    valid_loader = ToDeviceLoader(validation_loader, device)
    resnet_model = to_device(model, device)
    
    if device == 'cuda':
        #free up the cuda memory
        torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=15).to(device)
    early_stopping = EarlyStopping(patience=patience)

    # OneCycleLR scheduler to adjust the learning rate
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=n_epochs)
    history, best_accuracy = fit(resnet_model, train_loader, valid_loader, optimizer, n_epochs, early_stopping, criterion, grad_clip, scheduler, accuracy, model_dir)

    global_history.append(history)

    return best_accuracy

def plot_acc(history):
    plt.figure(figsize=(10, 6))
    plt.plot([Tensor.cpu(x["val_acc"]) for x in history],"x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot([x.get("train_loss") for x in history], "-bx")
    plt.plot([x["val_loss"] for x in history],"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss","val loss"])


###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""

def train(train_data_dir, model_dir, **kwargs):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
        
    # Train the new model with hyperparameter tuning
    study = optuna.create_study(study_name='hyperparams_optimization', direction='maximize')
    # pass the train_data_dir and model_dir to training function
    study.optimize(lambda trial: objective(trial, train_data_dir, model_dir), n_trials=30)
    print("Best trial:")
    trial_ = study.best_trial
    print(f"  Accuracy: {trial_.value}")
    print("  Params: ")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")
    
    # return the accuracy of the best trial
    return trial_.value


def test(test_data_dir, model_dir, **kwargs):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """

    # Check if the model exists
    if os.path.exists(model_dir):

        device = get_device()

        # Load the model 
        model = torch.load(model_dir, map_location=torch.device(device))
        model.eval()
        
        accuracy = Accuracy(task="multiclass", num_classes=15).to(device)
        criterion = nn.CrossEntropyLoss()
        mean, std = get_mean_std(test_data_dir)

        # Set the training size to 0 to only load the validation set
        _, validation_loader = load_data(mean, std, test_data_dir,batch_size=32, train_size=0)
        device = get_device()
        valid_loader = ToDeviceLoader(validation_loader, device)
        
        val_acc, _ = evaluate(model, accuracy, criterion, valid_loader)
        
    else:
        print("Model not found, please train the model first.")
        val_acc = 0
    
    return val_acc



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pt', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print("training accuracy: " + str(training_accuracy.item()))

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print("testing accuracy: " + str(testing_accuracy.item()))






