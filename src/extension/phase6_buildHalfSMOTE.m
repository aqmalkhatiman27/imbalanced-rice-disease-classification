function augFile = phase6_buildHalfSMOTE(featureSet)
% phase6_buildHalfSMOTE  –  Create (or retrieve) HALF-SMOTE-balanced sets
%
% Usage
%   phase6_buildHalfSMOTE('dct')   % → features_augmented/X_dct_halfSMOTE.mat
%   phase6_buildHalfSMOTE('pca')   % → features_augmented/X_pca_halfSMOTE.mat
%
% Notes
% • Idempotent: if the .mat file already exists, nothing is recomputed.
% • Belongs in /phase-6, next to phase6_HalfSMOTE.m.

% ---------- 0. Sanity-check on input ------------------------------------
featureSet = lower(string(featureSet));
assert(ismember(featureSet,["dct","pca"]), ...
       "featureSet must be 'dct' or 'pca'.");

% ---------- 1. Resolve I/O ----------------------------------------------
switch featureSet
    case "dct"
        selFile = "features_selected/X_dct_sel.mat";
        augFile = "features_augmented/X_DCT300_halfSMOTE.mat";
        Xfield  = "Xtr_dct_sel";
    otherwise                         % "pca"
        selFile = "features_selected/X_pca_sel.mat";
        augFile = "features_augmented/X_PCA300_halfSMOTE.mat";
        Xfield  = "Xtr_pca_sel";
end

% ---------- 2. Skip if already on disk ----------------------------------
if isfile(augFile)
    S = load(augFile);      %#ok<NASGU>  % idempotent return
    return
end

% ---------- 3. Build via HALF-SMOTE -------------------------------------
fprintf("[HALF-SMOTE] Generating %s …\n", augFile);

S = load(selFile);
X = S.(Xfield);   Y = S.Ytr;

addpath(genpath(fullfile("phase-6","lib")));   % makes helpers visible
[X_aug, Y_aug] = halfSMOTE(X, Y);              % <-- core oversampling

% ---------- 4. Persist ---------------------------------------------------
if ~exist("features_augmented","dir"), mkdir features_augmented, end
save(augFile,"X_aug","Y_aug");

fprintf("✔︎ Saved %s  (%d × %d)\n", augFile,size(X_aug,1),size(X_aug,2));
end
