function phase6_FullSMOTE(featureSet)
% phase6_FullSMOTE (Standardized Version)
% Builds Full-SMOTE data and trains all three SVMs (L, Q, C).

%% 0. Input Guard & File Paths
featureSet = lower(string(featureSet));
assert(ismember(featureSet, ["dct","pca"]), "featureSet must be 'dct' or 'pca'.");

if strcmp(featureSet, 'dct')
    selFile = 'features_selected/X_dct_sel.mat';
    augFile = 'features_augmented/X_DCT300_fullSMOTE.mat';
    XteField = 'Xte_dct_sel';
    tag = 'S3_DCT300_fullSMOTE'; % Generic tag for the experiment
else % pca
    selFile = 'features_selected/X_pca_sel.mat';
    augFile = 'features_augmented/X_PCA300_fullSMOTE.mat';
    XteField = 'Xte_pca_sel';
    tag = 'S3_PCA300_fullSMOTE'; % Generic tag for the experiment
end

%% 1. Build & Load Data
phase6_buildFullSMOTE(featureSet); % This builds the augmented file if needed
assert(isfile(selFile) && isfile(augFile), 'Missing required data files.');

%% 2. Load Datasets
Ssel = load(selFile);
Saug = load(augFile);
Xtr = Saug.X_aug;
Ytr = Saug.Y_aug;
Xte = Ssel.(XteField);
Yte = Ssel.Yte;

%% 3. Train All Three SVM Kernels
fprintf('\n--- Training All SVMs on %s Full-SMOTE Data ---\n', upper(featureSet));

% Call runSVMsuite and let it run its default loop over all three kernels.
metrics = runSVMsuite(Xtr, Ytr, Xte, Yte, tag);

%% 4. Display Summary
fprintf('\n--- Final Accuracies for %s Stream (Full-SMOTE) ---\n', upper(featureSet));
T_results = struct2table(metrics);
disp(T_results(:, {'kernel', 'acc'}));

end