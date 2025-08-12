%% phase6_Hybrid.m
% This script runs the final HYBRID experiment, combining Borderline-SMOTE
% data with a Cost-Sensitive SVM (at ratio=1.00).

clear; clc;
addpath(genpath('toolbox'));
rng(20250622,'twister');

fprintf('====== Starting Final Hybrid Remedy Experiment ======\n');

% --- This loop will run everything for both 'pca' and 'dct' ---
for featureSet = {'pca', 'dct'}
    
    current_set = featureSet{1};
    fprintf('\n--- Processing stream: %s ---\n', upper(current_set));

    % 1. Define File Paths and Tags
    if strcmp(current_set, 'pca')
        selFile = 'features_selected/X_pca_sel.mat';
        augFile = 'features_augmented/X_PCA300_borderlineSMOTE.mat';
        XteField = 'Xte_pca_sel';
        tag = 'S3_PCA300_hybrid';
    else % dct
        selFile = 'features_selected/X_dct_sel.mat';
        augFile = 'features_augmented/X_DCT300_borderlineSMOTE.mat';
        XteField = 'Xte_dct_sel';
        tag = 'S3_DCT300_hybrid';
    end
    
    % 2. Build the Borderline-SMOTE augmented data file if it doesn't exist
    phase6_buildBorderlineSMOTE(current_set);
    assert(isfile(selFile) && isfile(augFile), 'Missing required data files.');
    
    % 3. Load Datasets
    Ssel = load(selFile); % Original test set
    Saug = load(augFile); % Augmented training set
    Xtr = Saug.X_aug;
    Ytr = Saug.Y_aug;
    Xte = Ssel.(XteField);
    Yte = Ssel.Yte;
    
    % 4. Generate the Cost Matrix (using the CORRECTED formula)
    class_info = tabulate(Ssel.Ytr); % IMPORTANT: Use original training counts
    class_counts = cell2mat(class_info(:,2));
    num_classes = size(class_info, 1);
    
    % Using the standardized "Inverse Total Frequency" weighting at r=1.00
    class_weights = (sum(class_counts) ./ class_counts) * 1.00;
    
    cost_matrix = ones(num_classes, num_classes);
    for i = 1:num_classes
        cost_matrix(i, :) = class_weights(i);
    end
    cost_matrix(logical(eye(num_classes))) = 0;

    % 5. Train All Three SVM Kernels with Both Remedies
    fprintf('Training HYBRID SVMs for: %s\n', tag);
    metrics = runSVMsuite(Xtr, Ytr, Xte, Yte, tag, 'Cost', cost_matrix);
    
    % 6. Display Summary
    fprintf('\n--- Final Accuracies for %s Stream (Hybrid Remedy) ---\n', upper(current_set));
    T_results = struct2table(metrics);
    disp(T_results(:, {'kernel', 'acc'}));
end

fprintf('\n====== Hybrid Remedy Experiment Complete ======\n');