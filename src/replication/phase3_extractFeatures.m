function phase3_extractFeatures()
% Bilayer feature extraction from fine-tuned models.
% Includes special handling for pruned DarkNet-19's 'conv18' layer
% and reproducible "DTCWT" size-halving for all L1 features.

%% 0. Load datastores & models
S = load('temp_imds.mat'); % Contains: imdsTrain, imdsVal, aug

% Reordered to match the training order from phase2 for project consistency.
nets = {'darknet19', 'mobilenetv2', 'resnet18'}; 

if ~exist('features', 'dir'), mkdir features, end
rng(20250622, 'twister'); % Set seed for reproducibility

% --- Main Extraction Loop ---
for k = 1:numel(nets)
    name = nets{k};
    
    % Load the tuned network and its metadata (L1, L2 layer names)
    fprintf('Loading model: Tuned_%s.mat\n', name);
    M = load(fullfile('models', sprintf('Tuned_%s.mat', name)));
    netT = M.tunedNet;
    map = M.meta;
    
    fprintf('📤  %s — extracting features …\n', name);
    inSize = netT.Layers(1).InputSize;
    
    % Create augmented datastores for activation extraction
    augT = augmentedImageDatastore(inSize, S.imdsTrain);
    augV = augmentedImageDatastore(inSize, S.imdsVal);
    
    %% Layer-2 Feature Extraction (L2)
    % This layer is extracted directly for all networks using the name from metadata.
    fprintf('   • Extracting from Layer-2 (%s)\n', map.L2);
    F2tr = activations(netT, augT, map.L2, 'OutputAs', 'rows', 'MiniBatchSize', 10);
    F2va = activations(netT, augV, map.L2, 'OutputAs', 'rows', 'MiniBatchSize', 10);
    save(fullfile('features', sprintf('L2_%s.mat', name)), 'F2tr', 'F2va', '-v7');
    
    %% Layer-1 Feature Extraction (L1)
    if strcmp(name, 'darknet19')    
        % Special handling *only* for Pruned DarkNet-19.
        % We MUST use the hardcoded name 'conv18' here because it is the
        % layer that was modified to have 660 channels in prune_darknet19.m.
        % The metadata 'map.L1' contains 'conv19', which is the wrong layer
        % for this specific averaging task.
        
        fprintf('   • Extracting from Layer-1 (Special handling for DarkNet-19)\n');

        % 1. Pull conv18 features: [N × (8*8*660)] -> [N x 42240]
        actTrain = activations(netT, augT, 'conv18', 'OutputAs', 'rows', 'MiniBatchSize', 10);
        actVal   = activations(netT, augV, 'conv18', 'OutputAs', 'rows', 'MiniBatchSize', 10);
    
        % 2. Reshape to separate spatial dimensions: [N × 64 × 660]
        actTrain = reshape(actTrain, [], 64, 660);
        actVal   = reshape(actVal,   [], 64, 660);
    
        % 3. Average every block of 64 spatial values -> [N x 660]
        F1tr = squeeze(mean(actTrain, 2));
        F1va = squeeze(mean(actVal,   2));
    else
        % For MobileNet-v2 and ResNet-18, extract L1 features as usual using metadata.
        fprintf('   • Extracting from Layer-1 (%s)\n', map.L1);
        F1tr = activations(netT, augT, map.L1, 'OutputAs', 'rows', 'MiniBatchSize', 10);
        F1va = activations(netT, augV, map.L1, 'OutputAs', 'rows', 'MiniBatchSize', 10);    
    end

    %% Post-process Layer-1 features with DTCWT
    fprintf('   • Applying DTCWT reduction to Layer-1 features\n');
    F1tr_cdt = dtcwt_reduce(F1tr); % [N × d] → [N × d/2]
    F1va_cdt = dtcwt_reduce(F1va);
    
    % Save the final, processed L1 features
    save(fullfile('features', sprintf('L1_%s.mat', name)), ...
         'F1tr_cdt', 'F1va_cdt', '-v7');
    fprintf('   ✔ Done.\n\n');
end

disp('✔ Phase 3 complete — six .mat files created in /features');
end