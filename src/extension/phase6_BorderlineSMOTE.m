function phase6_BorderlineSMOTE(featureSet)
% phase6_BorderlineSMOTE (Final Version)
% Builds Borderline-SMOTE data and trains all three SVMs (L, Q, C).

%% 0. Input Guard & File Paths
featureSet = lower(string(featureSet));
assert(ismember(featureSet, ["dct","pca"]), "featureSet must be 'dct' or 'pca'.");

if strcmp(featureSet, 'dct')
    selFile = 'features_selected/X_dct_sel.mat';
    augFile = 'features_augmented/X_DCT300_borderlineSMOTE.mat';
    XteField = 'Xte_dct_sel';
    tag = 'S3_DCT300_borderlineSMOTE';
else % pca
    selFile = 'features_selected/X_pca_sel.mat';
    augFile = 'features_augmented/X_PCA300_borderlineSMOTE.mat';
    XteField = 'Xte_pca_sel';
    tag = 'S3_PCA300_borderlineSMOTE';
end

%% 1. Build the Augmented Data File
phase6_buildBorderlineSMOTE(featureSet);
assert(isfile(selFile) && isfile(augFile), 'Missing required data files.');

%% 2. Load Datasets
Ssel = load(selFile); % Original test set
Saug = load(augFile); % Augmented training set
Xtr = Saug.X_aug;
Ytr = Saug.Y_aug;
Xte = Ssel.(XteField);
Yte = Ssel.Yte;

%% 3. Train All Three SVM Kernels
fprintf('\n--- Training All SVMs on %s Borderline-SMOTE Data ---\n', upper(featureSet));

% Call runSVMsuite and let it run its default loop over all three kernels (L, Q, C).
metrics = runSVMsuite(Xtr, Ytr, Xte, Yte, tag); % No 'Spec' needed

%% 4. Display Summary
fprintf('\n--- Final Accuracies for %s Stream (Borderline-SMOTE) ---\n', upper(featureSet));
T_results = struct2table(metrics);
disp(T_results(:, {'kernel', 'acc'}));

end