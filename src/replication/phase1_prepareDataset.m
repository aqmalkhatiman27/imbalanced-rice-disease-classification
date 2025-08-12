function [imdsTrain,imdsVal] = phase1_prepareDataset(imgDir, splitRatio)
% Reproducible 70/30 split + online augmentation for RiPa-Net replication.

if nargin < 2,  splitRatio = 0.70;  end

rng(20250622,'twister');           % ← fix the seed: YYYYMMDD (choose any)

% 1) Build master datastore
imds = imageDatastore(imgDir, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

% 2) Stratified split (seeded RNG makes this deterministic)
[imdsTrain, imdsVal] = splitEachLabel(imds, splitRatio);

% 3) Online data augmentation (same ranges as paper)
pxShift = [-25 25];  shearR = [-30 30];  scaleR = [1 2];
aug = imageDataAugmenter( ...
      'RandXReflection',true, 'RandYReflection',true, ...
      'RandXTranslation',pxShift, 'RandYTranslation',pxShift, ...
      'RandXShear',shearR, 'RandXScale',scaleR, 'RandYScale',scaleR);

save temp_imds.mat imdsTrain imdsVal aug
fprintf('\n✔ Phase 1 completed — temp_imds.mat saved (seed = 20250622)\n');
end