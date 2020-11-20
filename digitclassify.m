%reference: https://ww2.mathworks.cn/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html?lang=en


%load data
digitDatasetPath = '/home/wang/Downloads/digit/DigitDataset';
imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,"LabelSource","foldernames");


%split data. 1000 images for each 0-9. Take 75% of each for training.
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomized');


%building CNN, need layers, options and network.
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


% Set the Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(imdsTrain,layers,options);


% Do a prediction by using net.
YPred = classify(net,imdsValidation);
accuracy = sum(YPred == imdsValidation)/numel(imdsValidation.Labels)


%this net has a accuracy of 98.52%.
