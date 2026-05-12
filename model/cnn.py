import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

# From https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Input image must be 3x32x32
class CNN_32x32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # The number of outputs from the final layer must be the number of
        # categories which is 38 for the plant disease dataset
        self.fc3 = nn.Linear(84, 38)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_256x256(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=6 , kernel_size=5, stride=1, bias=True) # 6 filters
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)

        # Compute the size after conv+pool to define the first linear layer
        # Input image = 256x256
        # conv1: 256-5+1=252, pool: 252/2=126
        # conv2: 126-5+1=122, pool: 122/2=61
        self.fc1 = nn.Linear(in_features=16 * 61 * 61, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # The number of outputs from the final layer must be the number of categories
        self.out = nn.Linear(84, out_features=38)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    

class CNN_2(nn.Module):
    
    def __init__(self, kernel_size =5, init_outputs =8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=8 , kernel_size=kernel_size, padding="same") # 6 filters
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, padding="same")
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding="same")
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding="same")
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding="same")
        self.pool5 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=84)
        self.out = nn.Linear(84, out_features=38)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# A more automated version of making a cnn 
class CNN(nn.Module):

    #def __init__(self, kernel_sizes=[5,5,3,3,3], linear_size=[128,64], init_outputs=16, dropout=0.5, categories=38):
    def __init__(self, kernel_sizes, linear_size, init_outputs, dropout, categories):

        super().__init__()

        n = init_outputs

        # Convolution layer 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n, kernel_size=kernel_sizes[0], padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        # Convolution layers 2, 3, ... 
        for i in range (1, len(kernel_sizes)):
            self.conv.extend(nn.Sequential(
                nn.Conv2d(in_channels=n, out_channels=n*2, kernel_size=kernel_sizes[i], padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            ))
            n *= 2

        # Input size is 256x256 and each MaxPool2d halves the size
        side_length = int(256 / (2 ** len(kernel_sizes))) # e.g. in this case we are halving 256 5 times

        # Input linear layer
        self.linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=n * side_length * side_length, out_features=linear_size[0])
        )

        # Hidden linear layers
        for i in range (1, len(linear_size)):
            self.linear.extend(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features=linear_size[i - 1], out_features=linear_size[i])
            ))

        # Output linear layer (no dropout)
        self.linear.append(
            nn.Linear(in_features=linear_size[-1], out_features=categories)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

    # verbose (int):
    #   0 (quiet): No output
    #   1 (default): Print model summary
    #   2 (verbose): Show weight and bias layers in full detail
    # https://github.com/TylerYep/torchinfo
    def summary(self, batch_size=1, verbose=0):
        return torchinfo.summary(self, (batch_size, 3, 256, 256), verbose=verbose)


def CNN_3(dropout =0.5, categories=38):
    return CNN(kernel_sizes=[5,5,3,3,3],linear_size=[128,64], init_outputs=16,dropout=dropout, categories=categories)

# same as CNN_3 but
# made the kernel sizes constant throughout
def CNN_4(dropout =0.5, categories=38):
    return CNN(kernel_sizes=[3,3,3,3,3],linear_size=[128,64], init_outputs=16,dropout=dropout, categories=categories)

def CNN_5(dropout =0.5, categories=38):
    return CNN(kernel_sizes=[5,5,5,3,3,3,3],linear_size=[128,64,48], init_outputs=16,dropout=dropout, categories=categories)

def CNN_6(dropout =0.5, categories=38):
    return CNN(kernel_sizes=[5,5,5,3,3,3],linear_size=[128,64,48], init_outputs=16,dropout=dropout, categories=categories)

def CNN_7(dropout =0.5, categories=38):
    return CNN(kernel_sizes=[5,5,3,3,3,3],linear_size=[128,64,48], init_outputs=16,dropout=dropout, categories=categories)

def CNN_8(dropout =0.5, categories=38):
    return CNN(kernel_sizes=[5,5,3,3,3],linear_size=[128,64], init_outputs=16,dropout=dropout, categories=categories)

def CNN_9(dropout =0.5, categories=38):
    return CNN(kernel_sizes=[3,3,3,3,3],linear_size=[128,64], init_outputs=16,dropout=dropout, categories=categories)


class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self,x):
        residual = self.downsample(x)
        return F.relu(self.conv(x) + residual)

class Simple_ResNet1(nn.Module):
    def __init__(self, init_outputs=16, dropout=0.2, categories=38):
        super().__init__()

        n_in = 3
        n_out = init_outputs

        self.conv = nn.Sequential()
        for i in range(5):
            self.conv.extend(nn.Sequential(
                SimpleResidualBlock(in_channels=n_in, out_channels=n_out),
                nn.MaxPool2d(2,2)
            ))
            n_in = n_out
            n_out *= 2

        size = int(256 / (2 ** 5))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=n_in * size * size, out_features=categories)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def summary(self, batch_size=1, verbose=0):
        return torchinfo.summary(self, (batch_size, 3, 256, 256), verbose=verbose)
    

class Simple_ResNet2(nn.Module):
    def __init__(self, init_outputs=16, dropout=0.2, categories=38):
        super().__init__()

        n_in = 3
        n_out = init_outputs

        self.conv = nn.Sequential()
        for i in range(5):
            self.conv.extend(nn.Sequential(
                SimpleResidualBlock(in_channels=n_in, out_channels=n_out, stride=2),
            ))
            n_in = n_out
            n_out *= 2

        size = int(256 / (2 ** 5))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=n_in * size * size, out_features=categories)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def summary(self, batch_size=1, verbose=0):
        return torchinfo.summary(self, (batch_size, 3, 256, 256), verbose=verbose)
    

class Simple_ResNet3(nn.Module):
    def __init__(self, init_outputs=16, dropout=0.5, categories=38, linear_layers=[128,64]):
        super().__init__()

        n_in = 3
        n_out = init_outputs

        self.conv = nn.Sequential()
        for i in range(5):
            self.conv.extend(nn.Sequential(
                SimpleResidualBlock(in_channels=n_in, out_channels=n_out, stride=2),
            ))
            n_in = n_out
            n_out *= 2

        size = int(256 / (2 ** 5))
        n_in = n_in * size * size

        self.linear = nn.Sequential()
        for layer_size in linear_layers:
            self.linear.extend(nn.Sequential(
                nn.Dropout(dropout), 
                nn.Linear(n_in, layer_size)
            ))
            n_in = layer_size
    
        self.fc = nn.Linear(in_features=n_in, out_features=categories)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.linear(x)
        x = self.fc(x)
        return x

    def summary(self, batch_size=1, verbose=0):
        return torchinfo.summary(self, (batch_size, 3, 256, 256), verbose=verbose)
    
   
class CNN_10(nn.Module):

    def __init__(self, kernel_sizes=[3,3,3,3,3,3,3], linear_size=[128,64], init_outputs=16, dropout=0.2, categories=38):
        super().__init__()

        n_in = 3
        n_out = init_outputs

        self.conv = nn.Sequential()
        for k in kernel_sizes:
            self.conv.extend(nn.Sequential(
                nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_sizes=k, padding="same"),
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            ))
            n_in = n_out
            n_out *= 2

        size = int(256 / (2 ** len(kernel_sizes)))
        n_in = n_in * size * size

        self.linear = nn.Sequential()
        for l in linear_size:
            self.linear.extend(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features=n_in, out_features=l)
            ))
            n_in = l

        self.linear.append(
            nn.Linear(in_features=n_in, out_features=categories)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

    def summary(self, batch_size=1, verbose=0):
        return torchinfo.summary(self, (batch_size, 3, 256, 256), verbose=verbose)
    
class CNN_11(nn.Module):

    def __init__(self, kernel_size=[3,3,3,3,3], linear_size=[128,64], init_outputs=16, dropout=0.5, categories=38):
        super().__init__()

        n_in = 3
        n_out = init_outputs

        self.conv = nn.Sequential()
        for k in kernel_size:
            self.conv.extend(nn.Sequential(
                nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=k, padding="same"),
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            ))
            n_in = n_out
            n_out *= 2

        size = int(256 / (2 ** len(kernel_size)))
        n_in = n_in * size * size

        self.linear = nn.Sequential()
        for l in linear_size:
            self.linear.extend(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features=n_in, out_features=l)
            ))
            n_in = l

        self.linear.append(
            nn.Linear(in_features=n_in, out_features=categories)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

    def summary(self, batch_size=1, verbose=0):
        return torchinfo.summary(self, (batch_size, 3, 256, 256), verbose=verbose)

# Testing
if __name__ == "__main__":
    # Create a random input tensor for 256x256 RGB image
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 256, 256)

    def test(cnn):
        cnn.forward(input_tensor)

    test(CNN_3())
    test(CNN_4())
    test(CNN_5())
    test(CNN_6())
    test(CNN_7())
    test(CNN_8())
    test(CNN_9())

    test(CNN_10())
    test(CNN_11())
    
    test(Simple_ResNet1())

    test(Simple_ResNet2())

    test(Simple_ResNet3())

    


    




