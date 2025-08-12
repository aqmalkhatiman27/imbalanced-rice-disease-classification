function augFile = phase6_buildBorderlineSMOTE(featureSet)
% phase6_buildBorderlineSMOTE – create / load BORDERLINE-SMOTE sets
%
%   phase6_buildBorderlineSMOTE('dct') → features_augmented/X_DCT_borSMOTE.mat
%   phase6_buildBorderlineSMOTE('pca') → features_augmented/X_PCA_borSMOTE.mat
%
% Idempotent: exits early if the .mat already exists.

% -- 0. guard ----------------------------------------------------------
featureSet = lower(string(featureSet));
assert(ismember(featureSet,["dct","pca"]), ...
       "featureSet must be 'dct' or 'pca'.");

% -- 1. map I/O --------------------------------------------------------
switch featureSet
    case "dct"
        selFile = "features_selected/X_dct_sel.mat";
        augFile = "features_augmented/X_DCT300_borderlineSMOTE.mat";
        Xfield  = "Xtr_dct_sel";
    case "pca"
        selFile = "features_selected/X_pca_sel.mat";
        augFile = "features_augmented/X_PCA300_borderlineSMOTE.mat";
        Xfield  = "Xtr_pca_sel";
end

% -- 2. quick-exit if file exists -------------------------------------
if isfile(augFile), return, end

fprintf("[BORDERLINE-SMOTE] Generating %s …\n", augFile);

% -- 3. load original training split ----------------------------------
S = load(selFile);
X = S.(Xfield);   Y = S.Ytr;

% -- 4. oversample -----------------------------------------------------
addpath(genpath(fullfile("phase-6","lib")));   % exposes borderlineSMOTE.m
[X_aug, Y_aug] = borderlineSMOTE(X, Y);        % ← your lib function

% -- 5. save -----------------------------------------------------------
if ~exist("features_augmented","dir"), mkdir features_augmented, end
save(augFile,"X_aug","Y_aug");

fprintf("✔︎ Saved %s  (%d × %d)\n", augFile,size(X_aug,1),size(X_aug,2));
end
