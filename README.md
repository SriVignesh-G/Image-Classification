# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

To Classify the given image to the respective category using Convolutional Neural Network.

## Neural Network Model
![423398622-acb92196-8a29-40c9-9505-82bf31af77b3](https://github.com/user-attachments/assets/4d6db80c-cbb8-419d-9ac0-808433963c94)

## DESIGN STEPS

### STEP 1:
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

### STEP 2:
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

### STEP 3:
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

### STEP 4:
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

### STEP 5:
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 6:
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name: Sri Vignesh G
### Register Number: 212223040204
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

![image](https://github.com/user-attachments/assets/9dc61c5c-ec9f-4292-84ac-0f0794c597a6)


### Confusion Matrix
![download](https://github.com/user-attachments/assets/ccd50849-f32d-41c3-93bd-462dc844aead)


### Classification Report

![image](https://github.com/user-attachments/assets/0d58becd-0b25-43bf-8056-7ae2a4dd138c)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/26a64767-564b-4ec4-872d-9d1b02d2cb9b)
 

## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
