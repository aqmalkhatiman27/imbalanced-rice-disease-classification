function phase4_fuseFeatures()
% === RiPa-Net Phase-4 · Faithful & Functional Replication of Feature Fusion ===
% This script replicates the author's unique fusion methodology while
% correcting a critical bug in the original PCA implementation.
%
% Builds:
%   fusion/BothLayers_DCT.mat (10407 samples x 530 features)
%   fusion/BothLayers_PCA.mat (10407 samples x 330 features)
% -------------------------------------------------------------------------

fprintf('Executing Phase 4: Multi-deep features fusion...\n\n');

rng(20250622,'twister'); % Use the same seed as earlier phases
if ~exist('fusion','dir'), mkdir fusion; end

%% 1 · Load All Prerequisite Data (Features & Labels)
fprintf('--> Step 1: Loading all features and labels...\n');

% Load labels from the initial data split
data_splits = load('temp_imds.mat');
Ytr = data_splits.imdsTrain.Labels;
Yva = data_splits.imdsVal.Labels;
Y   = [Ytr; Yva]; % Consolidated label vector

% Load all 6 feature files and consolidate them
cnn = {'darknet19','mobilenetv2','resnet18'}; % Order matters for consistency
L1_tr = []; L1_va = []; % Layer 1 Train/Validation
L2_tr = []; L2_va = []; % Layer 2 Train/Validation

for c = 1:numel(cnn)
    A = load(fullfile('features', sprintf('L1_%s.mat', cnn{c}))); % F1*_cdt
    B = load(fullfile('features', sprintf('L2_%s.mat', cnn{c}))); % F2*
    
    L1_tr = [L1_tr, A.F1tr_cdt]; %#ok<AGROW>
    L1_va = [L1_va, A.F1va_cdt]; %#ok<AGROW>
    L2_tr = [L2_tr, B.F2tr];     %#ok<AGROW>
    L2_va = [L2_va, B.F2va];     %#ok<AGROW>
end
% Create master feature matrices by stacking train and validation sets
L1 = [L1_tr; L1_va]; % Final size: 10407 x 1226
L2 = [L2_tr; L2_va]; % Final size: 10407 x 30


%% 2 · DCT-500 Fusion Branch (Faithful replication of author's method)
fprintf('--> Step 2: Performing DCT-500 fusion...\n');

% Apply DCT on the transposed feature matrix (features are now rows)
F_dct = dct(single(L1'), [], 1); 

% Get the zig-zag selection order to mimic the author's helper function
zig_idx = zigzag_order(size(F_dct, 1));

% Select the top 500 DCT coefficients and concatenate with L2 features
% Note: L2 is 30-dim, DCT is 500-dim, for a total of 530 features.
X_dct = [L2' ; F_dct(zig_idx(1:500), :)]';


%% 3 · PCA-300 Fusion Branch (Corrected and functional implementation)
fprintf('--> Step 3: Performing PCA-300 fusion...\n');

% NOTE: The author's original code appears to use the PCA *coefficients*
% (eigenvectors) as features, which is statistically unorthodox and results
% in a dimension mismatch. The correct and intended method is to use the
% PCA *scores* (the transformed data). We implement the corrected version.

% Apply PCA to the L1 feature matrix (samples are rows)
[~, score] = pca(single(L1));

% Select the first 300 principal components (the new features)
F1_pca = score(:, 1:300);

% Concatenate the L2 features with the new PCA-reduced L1 features.
% Note: L2 is 30-dim, PCA is 300-dim, for a total of 330 features.
X_pca = [L2, F1_pca];


%% 4 · Save Final Fused Datasets
fprintf('--> Step 4: Saving final fused datasets to /fusion folder...\n');

save('fusion/BothLayers_DCT.mat', 'X_dct', 'Y', '-v7.3');
save('fusion/BothLayers_PCA.mat', 'X_pca', 'Y', '-v7.3');

fprintf('\n✔ Phase 4 complete — fusion/BothLayers_DCT.mat & fusion/BothLayers_PCA.mat written.\n');
end

% -------------------------------------------------------------------------
% Local helper function to reproduce the author's zig-zag coefficient selection
function ord = zigzag_order(N)
    S = reshape(1:N, N, 1); % Create a dummy matrix to get indices
    ord = [];
    for s = 2:2*N
        if mod(s,2) == 1
            idx = diag(flipud(S), N-s);
        else
            idx = diag(S, s-N);
        end
        ord = [ord; idx(idx>0)];
    end
    % This simplified zigzag is sufficient for a 1-D DCT coefficient vector
    ord = unique(ord, 'stable');
end