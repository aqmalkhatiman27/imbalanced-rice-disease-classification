%% phase5_selectAndReSVM.m  –  Scenario-3 (mRMR + SVM, Table 5/Fig 4)
%  Uses the original "mrmr_mid_d" greedy loop (MID criterion)
%  to select 500 DCT and 300 PCA features, then retrains
%  Linear-, Quadratic-, and Cubic-SVMs.
%  --------------------------------------------------------------
%  Saves:
%    features_selected/X_dct_sel.mat   (idx, Xtr, Xte, Ytr, Yte)
%    features_selected/X_pca_sel.mat
%    results/S3_dct_*.mat   &   results/S3_pca_*.mat
%  --------------------------------------------------------------

clear;  rng(20250622,'twister');             % reproducibility
addpath( genpath('toolbox') );               % mrmr_mid_d + runSVMsuite

%% 1) Load fusion tables and recreate the train / val split --------------
load fusion/BothLayers_DCT.mat   % gives X_dct  Y
load fusion/BothLayers_PCA.mat   % gives X_pca  Y

Ntr = 7286;                                % rows 1…7286 are TRAIN
Xtr_dct = X_dct(1:Ntr,:);    Xte_dct = X_dct(Ntr+1:end,:);
Xtr_pca = X_pca(1:Ntr,:);    Xte_pca = X_pca(Ntr+1:end,:);
Ytr = Y(1:Ntr);              Yte     = Y(Ntr+1:end);

%% 2) Greedy mRMR selection (custom loop) ---------------------------------
fprintf('[mRMR] 300 / 530 DCT dims … ');      tic
idx_dct = mrmr_mid_d(double(Xtr_dct), double(grp2idx(Ytr)), 300);
fprintf('%.1fs\n', toc);

fprintf('[mRMR] 300 / 330 PCA dims … ');      tic
idx_pca = mrmr_mid_d(double(Xtr_pca), double(grp2idx(Ytr)), 300);
fprintf('%.1fs\n', toc);

Xtr_dct_sel = Xtr_dct(:,idx_dct);   Xte_dct_sel = Xte_dct(:,idx_dct);
Xtr_pca_sel = Xtr_pca(:,idx_pca);   Xte_pca_sel = Xte_pca(:,idx_pca);

if ~exist('features_selected','dir'), mkdir features_selected; end
save features_selected/X_dct_sel.mat idx_dct Xtr_dct_sel Xte_dct_sel Ytr Yte
save features_selected/X_pca_sel.mat idx_pca Xtr_pca_sel Xte_pca_sel Ytr Yte

%% 3) Retrain three SVM kernels with runSVMsuite --------------------------
m_dct = runSVMsuite(Xtr_dct_sel,Ytr,Xte_dct_sel,Yte,'S3_dct');
m_pca = runSVMsuite(Xtr_pca_sel,Ytr,Xte_pca_sel,Yte,'S3_pca');

disp('=== Scenario-3 overall accuracies (expect ≈ 97.1–97.5 %) ===')
fprintf('DCT-300 : %.4f  %.4f  %.4f (L  Q  C)\n', [m_dct.acc])
fprintf('PCA-300 : %.4f  %.4f  %.4f (L  Q  C)\n', [m_pca.acc])

%% 4) Quick confusion matrix preview (Quadratic-SVM, DCT branch) ---------
quad_dct = m_dct(2);                % element #2 = quadratic
figure;
cm = confusionchart( quad_dct.confMat, ...
                     'Normalization','row-normalized' );
cm.Title = 'Scenario 3 – DCT-300  Quadratic SVM   (TPR %)';
cm.CellLabelFormat = '%.1f%%';      % show percentages not counts
cm.RowSummary = 'row-normalized';   % optional mini-bars on RHS
cm.ColumnSummary = 'column-normalized';

%% 5)PCA
quad_pca = m_pca(2);
figure;
cm2 = confusionchart( quad_pca.confMat, ...
                      'Normalization','row-normalized' );
cm2.Title = 'Scenario 3 – PCA-300  Quadratic SVM   (TPR %)';
cm2.CellLabelFormat = '%.1f%%';
cm2.RowSummary = 'row-normalized';
cm2.ColumnSummary = 'column-normalized';