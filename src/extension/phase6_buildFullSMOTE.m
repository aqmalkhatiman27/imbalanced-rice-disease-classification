function augFile = phase6_buildFullSMOTE(featureSet)
% phase6_buildFullSMOTE  –  Create (or retrieve) FULL-SMOTE-balanced sets
%
% Usage
%   phase6_buildFullSMOTE('dct')   % -> features_augmented/X_dct_smote.mat
%   phase6_buildFullSMOTE('pca')   % -> features_augmented/X_pca_smote.mat
%
% Notes
% • Idempotent: if the .mat file already exists, nothing is recomputed.
% • Belongs in phase-6, next to phase6_FullSMOTE.m.

% --- 0. Sanity-check -----------------------------------------------------
featureSet = lower(string(featureSet));
assert(ismember(featureSet, ["dct","pca"]), ...
       "featureSet must be 'dct' or 'pca'.");

% --- 1. Resolve I/O ------------------------------------------------------
switch featureSet
    case "dct"
        selFile = "features_selected/X_dct_sel.mat";
        augFile = "features_augmented/X_DCT300_fullSMOTE.mat";
        Xfield  = "Xtr_dct_sel";
    otherwise          % "pca"
        selFile = "features_selected/X_pca_sel.mat";
        augFile = "features_augmented/X_PCA300_fullSMOTE.mat";
        Xfield  = "Xtr_pca_sel";
end

% --- 2. Skip if already on disk -----------------------------------------
if isfile(augFile)
    S = load(augFile);
    return
end

% --- 3. Build via FULL-SMOTE --------------------------------------------
fprintf("[FULL-SMOTE] Generating %s …\n", augFile);

S = load(selFile);
X = S.(Xfield);   Y = S.Ytr;

addpath(genpath(fullfile("phase-6","lib")));   % makes helpers visible
[X_aug, Y_aug] = fullSMOTE(X, Y);

% --- 4. Persist ----------------------------------------------------------
if ~exist("features_augmented","dir"), mkdir features_augmented, end
save(augFile, "X_aug", "Y_aug");

fprintf("✔︎ Saved %s  (%d × %d)\n", augFile, size(X_aug,1), size(X_aug,2));
end
