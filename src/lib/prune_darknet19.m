function pruned_darknet19 = prune_darknet19()
%prune_darknet19 Prunes the built-in DarkNet-19 to a 660-channel model.
%   This function fine-tunes the pre-trained DarkNet-19 network by
%   modifying the layers starting from 'conv18' to reduce the number of
%   channels to 660.
%
%   Output:
%       netSlim - The modified and assembled network (a DAGNetwork object).

% Set a fixed random seed for reproducibility.
rng(20250622,'twister');

% Load the pre-trained DarkNet-19 model.
baseNet = darknet19;
lgraph  = layerGraph(baseNet);

% --- 1. Prune 'conv18' to 660 channels ----------------------------------
% Get the original convolutional layer.
oldConv = lgraph.Layers(strcmp({lgraph.Layers.Name},'conv18'));

% Create a new convolutional layer with 660 filters.
newConv = convolution2dLayer(oldConv.FilterSize, 660, ...
          'Stride', oldConv.Stride, 'Padding', oldConv.PaddingSize, ...
          'Name', 'conv18', 'BiasLearnRateFactor', oldConv.BiasLearnRateFactor, ...
          'WeightLearnRateFactor', oldConv.WeightLearnRateFactor);

% Copy the weights and biases for the first 660 channels.
newConv.Weights = oldConv.Weights(:,:,:,1:660);
newConv.Bias    = oldConv.Bias(1:660);

% Replace the old layer with the new one.
lgraph = replaceLayer(lgraph, 'conv18', newConv);

% --- 2. Match the subsequent 'batchnorm18' layer -----------------------
% Get the original batch normalization layer.
oldBN = lgraph.Layers(strcmp({lgraph.Layers.Name},'batchnorm18'));

% Create a new batch normalization layer matching the new 660 channels.
newBN = batchNormalizationLayer('Name','batchnorm18','Epsilon',oldBN.Epsilon);

% Copy the parameters for the first 660 channels.
newBN.Offset = oldBN.Offset(1:660);
newBN.Scale = oldBN.Scale(1:660);
newBN.TrainedMean = oldBN.TrainedMean(1:660);
newBN.TrainedVariance = oldBN.TrainedVariance(1:660);

% Replace the old layer with the new one.
lgraph = replaceLayer(lgraph, 'batchnorm18', newBN);

% --- 3. Adjust the input depth of 'conv19' to 660 ---------------------
% Get the final convolutional layer.
oldC19 = lgraph.Layers(strcmp({lgraph.Layers.Name},'conv19'));

% Create a new convolutional layer that accepts 660 input channels.
newC19 = convolution2dLayer(oldC19.FilterSize, oldC19.NumFilters, ...
          'NumChannels', 660, 'Stride', oldC19.Stride, 'Padding', oldC19.PaddingSize, ...
          'Name', 'conv19', 'BiasLearnRateFactor', oldC19.BiasLearnRateFactor, ...
          'WeightLearnRateFactor', oldC19.WeightLearnRateFactor);

% Copy the weights corresponding to the first 660 input channels.
newC19.Weights = oldC19.Weights(:,:,1:660,:);
newC19.Bias    = oldC19.Bias; % Bias is per output channel, so it remains unchanged.

% Replace the old layer with the new one.
lgraph = replaceLayer(lgraph, 'conv19', newC19);

% --- 4. Assemble the fine-tuned network --------------------------------
pruned_darknet19 = assembleNetwork(lgraph);

end