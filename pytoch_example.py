import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


model = Model(input_size, output_size)
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)


def run_model():
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("Outside: input size", input.size(),
              "output_size", output.size())


run_model()
import torch.nn
# One aspect where static and dynamic graphs differ is control flow.
# For some models we may wish to perform different computation for each data point;
# for example a recurrent network might be unrolled for different numbers of time steps for each data point;
# this unrolling can be implemented as a loop.
# With a static graph the loop construct needs to be a part of the graph;
# for this reason TensorFlow provides operators such as tf.scan for embedding loops into the graph.
# With dynamic graphs the situation is simpler: since we build graphs on-the-fly for each example,
# we can use normal imperative flow control to perform computation that differs for each input.


# 一些额外的例子，来自pytorch官网，帮助理解计算图和pytorch工作原理
import random


def raw_NN_with_Pytorch():
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    N, D_in, H, D_out = 64, 1000, 100, 10
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Tensors during the backward pass.
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the a scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())
        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
