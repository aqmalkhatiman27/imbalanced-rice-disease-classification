%% phase6_recogniseWithSVM  –  final metrics & figures (Scenario-3)
%  Generates final results for both the hold-out test and 5-fold CV
%  to fully replicate Table 6 from the research article.
% -------------------------------------------------------------------------
clear; clc; rng(20250622,'twister');
addpath( genpath('toolbox') );

%% 1) Load All Necessary Data ---------------------------------------------
fprintf('--> Step 1: Loading all prerequisite data...\n');
% Load mRMR-selected feature sets from Phase 5
Sd = load('features_selected/X_dct_sel.mat');
Sp = load('features_selected/X_pca_sel.mat');

% Load the full, un-split fused features from Phase 4 (for CV)
load('fusion/BothLayers_DCT.mat', 'X_dct', 'Y');
load('fusion/BothLayers_PCA.mat', 'X_pca');

% Load class labels
S_labels = load('temp_imds.mat');
labels = categories(S_labels.imdsTrain.Labels);


%% 2) Hold-Out Test Set Evaluation (70/30 Split) --------------------------
%  This section generates the "Hold-Out Test Set" results for Table 6.
% -------------------------------------------------------------------------
fprintf('--> Step 2: Running Hold-Out Test Set evaluation...\n');
Ytr_holdout = Sd.Ytr;
Yte_holdout = Sd.Yte;

% Run SVMs on the hold-out split
m_dct_holdout = runSVMsuite(Sd.Xtr_dct_sel, Ytr_holdout, Sd.Xte_dct_sel, Yte_holdout, 'S3_dct_holdout');
m_pca_holdout = runSVMsuite(Sp.Xtr_pca_sel, Ytr_holdout, Sp.Xte_pca_sel, Yte_holdout, 'S3_pca_holdout');

% Compute metrics and build the results table for the hold-out test
T_holdout = table;
rows_holdout = {};
for method = {'DCT','PCA'}
    if strcmp(method{1},'DCT'), M = m_dct_holdout; else, M = m_pca_holdout; end
    for k = 1:3
        meas = computeMetrics(M(k).confMat);
        T_holdout = [T_holdout; struct2table(meas)]; %#ok<AGROW>
        tagLetters = 'LQC';
        rows_holdout{end+1} = sprintf('%s-%c', method{1}, tagLetters(k)); %#ok<AGROW>
    end
end
T_holdout.Properties.RowNames = rows_holdout;


%% 3) 5-Fold Cross-Validation Evaluation ----------------------------------
%  This section generates the "Five-Fold Cross-Validation" results for Table 6.
% -------------------------------------------------------------------------
fprintf('--> Step 3: Running 5-Fold Cross-Validation...\n');
cv = cvpartition(Y, 'KFold', 5);
allConf_dct = zeros(10, 10, 3); % Accumulators for 3 kernels
allConf_pca = zeros(10, 10, 3);

% Apply mRMR indices to the full datasets
X_dct_mRMR = X_dct(:, Sd.idx_dct);
X_pca_mRMR = X_pca(:, Sp.idx_pca);

for f = 1:cv.NumTestSets
    fprintf('    ...Processing fold %d of 5\n', f);
    idxTr = cv.training(f);
    idxTe = cv.test(f);
    
    % DCT branch for this fold
    m_dct_cv = runSVMsuite(X_dct_mRMR(idxTr,:), Y(idxTr), ...
                           X_dct_mRMR(idxTe,:), Y(idxTe), sprintf('S3_dct_cv_fold%d', f));
    
    % PCA branch for this fold
    m_pca_cv = runSVMsuite(X_pca_mRMR(idxTr,:), Y(idxTr), ...
                           X_pca_mRMR(idxTe,:), Y(idxTe), sprintf('S3_pca_cv_fold%d', f));
                           
    % Accumulate confusion matrices
    for k = 1:3
        allConf_dct(:,:,k) = allConf_dct(:,:,k) + m_dct_cv(k).confMat;
        allConf_pca(:,:,k) = allConf_pca(:,:,k) + m_pca_cv(k).confMat;
    end
end

% Compute final metrics from the summed confusion matrices
% Compute final metrics from the summed confusion matrices
T_cv5 = table;
rows_cv5 = {};
tagLetters = 'LQC'; % Define the letters once
for k = 1:3
    % DCT metrics
    T_cv5 = [T_cv5; struct2table(computeMetrics(allConf_dct(:,:,k)))]; %#ok<AGROW>
    rows_cv5{end+1} = sprintf('DCT-%c', tagLetters(k)); %#ok<AGROW>
    % PCA metrics
    T_cv5 = [T_cv5; struct2table(computeMetrics(allConf_pca(:,:,k)))]; %#ok<AGROW>
    rows_cv5{end+1} = sprintf('PCA-%c', tagLetters(k)); %#ok<AGROW>
end
T_cv5.Properties.RowNames = rows_cv5;


%% 4) Save Results and Generate Figures -----------------------------------
fprintf('--> Step 4: Saving all results and generating final figures...\n');
if ~exist('results','dir'), mkdir results; end
if ~exist('results/figures','dir'), mkdir('results/figures'); end

% Save CSV files for both test types
writetable(T_holdout, 'results/Table6_replication_HoldOut.csv', 'WriteRowNames', true);
writetable(T_cv5,     'results/Table6_replication_5Fold_CV.csv', 'WriteRowNames', true);

% Generate "pretty" confusion matrices from the hold-out test results
plotConfMatPretty(m_dct_holdout(2).confMat, labels, ...
    'Scenario 3 – DCT-300 · Quadratic-SVM (Hold Out)', ...
    'results/figures/S3_DCT300_TPR_QSVM_holdout.svg');

plotConfMatPretty(m_pca_holdout(2).confMat, labels, ...
    'Scenario 3 – PCA-300 · Quadratic-SVM (Hold Out)', ...
    'results/figures/S3_PCA300_TPR_QSVM_holdout.svg');


%% 5) Final Console Recap -------------------------------------------------
disp('----- Table 6 Replica: Hold-Out Test Set Results -----')
disp(T_holdout(:,{'Accuracy','Sensitivity','Specificity','F1','MCC'}))

disp('----- Table 6 Replica: 5-Fold Cross-Validation Results -----')
disp(T_cv5(:,{'Accuracy','Sensitivity','Specificity','F1','MCC'}))

fprintf('\nAll files written under /results ✅\n');