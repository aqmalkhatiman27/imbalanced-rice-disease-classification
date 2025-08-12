function phase2_trainCNNs()
% Reproducible fine-tuning of DarkNet-19, MobileNet-v2, and ResNet-18.
% This script is modified to train DarkNet-19 first.
% Hyper-params come straight from Algorithm 1/§4.2 of the paper
% (mini-batch 10, epochs 20, LR 1e-4).

%% 0. Load split + augmenter from Phase 1
S = load('temp_imds.mat');                % Contains: imdsTrain, imdsVal, aug

%% 1. Common training options
miniBatch  = 10;
maxEpochs  = 20;
learnRate  = 1e-4;

baseOpts = trainingOptions('sgdm', ...
           'MiniBatchSize', miniBatch, ...
           'MaxEpochs', maxEpochs, ...
           'InitialLearnRate', learnRate, ...
           'Shuffle', 'every-epoch', ...
           'Verbose', false, ...
           'Plots', 'training-progress');   % Show live curves

%% 2. Networks + layer names (reordered for DarkNet-19 first)
% MODIFICATION: 'darknet19' is now the first element to ensure it's trained first.
nets = {'darknet19', 'mobilenetv2', 'resnet18'};

% MODIFICATION: 'layerPair' is reordered and corrected to match the 'nets' array.
% The k-th row here corresponds to the k-th network in the 'nets' cell array.
layerPair = { ...
   'conv19',                     'avg1';   ... % Row 1: For darknet19
   'global_average_pooling2d_1', 'new_fc'; ... % Row 2: For mobilenetv2
   'pool5',                      'new_fc'  ... % Row 3: For resnet18
};

rng(20250622, 'twister');        % Master seed for reproducibility (YYYYMMDD)

outDir = fullfile(pwd,'models');
if ~exist(outDir, 'dir'); mkdir(outDir); end % Create 'models' directory using the full path

% --- Main Training Loop ---
for k = 1:numel(nets)
     name = nets{k};
     fprintf('🔧 Fine-tuning %s … (%d of %d)\n', name, k, numel(nets));

     %% 2.1 Load backbone and swap head
     if strcmp(name, 'darknet19')
         % For darknet19, use the custom pruned version
         baseNet = prune_darknet19;
     else
         % For other networks, load the standard pre-trained model
         baseNet = feval(name);    
     end
     
     lgraph = layerGraph(baseNet);
     [oldFC, cls] = findLayersToReplace(lgraph);
     nClasses = numel(categories(S.imdsTrain.Labels));

     % Replace the final classification head
     if isa(oldFC, 'nnet.cnn.layer.FullyConnectedLayer')
         newHead = fullyConnectedLayer(nClasses, 'Name', 'new_fc', ...
                    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
     else
         % This handles networks like DarkNet-19 that end with a conv layer
         newHead = convolution2dLayer(1, nClasses, 'Name', 'new_conv', ...
                    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
     end
     lgraph = replaceLayer(lgraph, oldFC.Name, newHead);
     lgraph = replaceLayer(lgraph, cls.Name, ...
                classificationLayer('Name', 'new_cls'));

     %% 2.2 Build augmented datastores for training and validation
     inSize   = baseNet.Layers(1).InputSize;
     augTrain = augmentedImageDatastore(inSize, S.imdsTrain, ...
                            'DataAugmentation', S.aug);
     augVal   = augmentedImageDatastore(inSize, S.imdsVal);

     %% 2.3 Deterministic training (CPU)
     opts = baseOpts;
     opts.ExecutionEnvironment = 'cpu';      % Switch to 'auto' for GPU
     opts.ValidationData       = augVal;
     opts.ValidationFrequency  = ...
             max(1, floor(numel(augTrain.Files) / miniBatch));
             
     rng(20250622, 'twister'); % Re-seed before each training for repeatability
     tunedNet = trainNetwork(augTrain, lgraph, opts);

     %% 2.4 Save weights + layer map for later use
     meta.L1 = layerPair{k, 1};
     meta.L2 = layerPair{k, 2};
     fprintf('💾 Saving model to models/Tuned_%s.mat\n\n', name);
     save(fullfile(outDir, sprintf('Tuned_%s.mat', name)), ...
          'tunedNet', 'meta', '-v7.3');
end

disp('✔ Phase 2 complete — all models have been trained and saved.');
end