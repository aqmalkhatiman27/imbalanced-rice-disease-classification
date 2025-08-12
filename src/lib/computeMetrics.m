function M = computeMetrics(C)
% C  –  confusion matrix (square, integer counts)
% returns a struct with Accuracy, Sensitivity, Specificity,
% Precision, F1, MCC   (all scalar)
TP  = diag(C);
FN  = sum(C,2) - TP;
FP  = sum(C,1)' - TP;
TN  = sum(C(:)) - (TP+FP+FN);

acc = sum(TP) / sum(C(:));
sen = mean(TP ./ (TP+FN));                 % recall / TPR
spe = mean(TN ./ (TN+FP));
pre = mean(TP ./ (TP+FP));
f1  = mean( 2*TP ./ (2*TP + FP + FN) );
mcc = mean( (TP.*TN - FP.*FN) ./ sqrt((TP+FP).*(TP+FN).*(TN+FP).*(TN+FN)) );

M = struct('Accuracy',acc, 'Sensitivity',sen, 'Specificity',spe, ...
           'Precision',pre,'F1',f1,'MCC',mcc);
end
