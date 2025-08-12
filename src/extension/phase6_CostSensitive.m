%% phase6_CostSensitive.m (Revised with Timing)
% This script systematically runs the cost-sensitive SVM experiments and
% now also measures the runtime for each experimental condition.

clear; clc;
addpath(genpath('toolbox'));
rng(20250622,'twister');

fprintf('====== Starting Full Cost-Sensitive SVM Experiment ======\n');

%% 1. Load Base Data (Labels and Features)
load('features_selected/X_pca_sel.mat'); % Loads Xtr_pca_sel, Xte_pca_sel, Ytr, Yte
load('features_selected/X_dct_sel.mat'); % Loads Xtr_dct_sel, Xte_dct_sel

%% 2. Define Cost Ratios to Test
cost_ratios_to_test = [0.50, 0.75, 1.00];

%% 3. Main Experiment Loop
for ratio = cost_ratios_to_test
    
    fprintf('\n--- Processing Cost Ratio: %.2f ---\n', ratio);
    
    % --- 3a. Generate the Cost Matrix for this Ratio ---
    class_info = tabulate(Ytr);
    class_counts = cell2mat(class_info(:,2));
    num_classes = size(class_info, 1);
    
    class_weights = (sum(class_counts) ./ class_counts) * ratio;
    
    cost_matrix = ones(num_classes, num_classes);
    for i = 1:num_classes
        cost_matrix(i, :) = class_weights(i);
    end
    cost_matrix(logical(eye(num_classes))) = 0;
    
    % --- 3b. Run Experiments for Both PCA and DCT Streams ---
    for featureSet = {'pca', 'dct'}
        current_set = featureSet{1};
        
        if strcmp(current_set, 'pca')
            Xtr = Xtr_pca_sel;
            Xte = Xte_pca_sel;
            tag = sprintf('S3_PCA300_cost%03d', ratio*100);
        else % dct
            Xtr = Xtr_dct_sel;
            Xte = Xte_dct_sel;
            tag = sprintf('S3_DCT300_cost%03d', ratio*100);
        end
        
        fprintf('Training SVMs for: %s\n', tag);
        
        % --- ADDED TIMING COMMANDS ---
        tic; % Start the timer
        metrics = runSVMsuite(Xtr, Ytr, Xte, Yte, tag, 'Cost', cost_matrix);
        elapsed_time = toc; % Stop the timer
        
        fprintf('--> This run took %.2f seconds.\n', elapsed_time);
        
        % Display a summary for this run
        fprintf('--- Accuracies for %s ---\n', tag);
        T_results = struct2table(metrics);
        disp(T_results(:, {'kernel', 'acc'}));
        fprintf('\n');
    end
end

fprintf('====== All Cost-Sensitive Experiments Complete ======\n');