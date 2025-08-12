function [X_out, Y_out] = fullSMOTE(X_in, Y_in, k)
% FULLSMOTE – balance every class up to the majority count with SMOTE
%
% Works with Bjarke Larsen's smote.m, which needs the 'Class' input to be
% numeric or plain char (not a cell array).  We therefore:
%   1. Encode the labels as integers     1,2,3,…             → smote
%   2. Run smote class-by-class
%   3. Convert the synthetic labels back to the original categorical set.

if nargin < 3,  k = 5;  end
assert(size(X_in,1) == numel(Y_in), 'X/Y size mismatch');

% -- 1. Encode labels -----------------------------------------------------
if ~iscategorical(Y_in)
    Ycat = categorical(Y_in);           % ensures single, clean category set
else
    Ycat = Y_in;
end
classes  = categories(Ycat);            % {'A'  'B'  'C' …}
Y_int    = double(Ycat);                % 1,2,3,…  (numeric codes)
counts   = accumarray(Y_int, 1);

if exist('phase6_getTargetCount','file')
    target = phase6_getTargetCount(counts,'full');   % central rule
else
    target = max(counts);                            % safe fallback
end

X_out = X_in;
Y_out = Ycat;                           % keep original labels

% -- 2. Balance every minority class with SMOTE --------------------------
for c = 1:numel(classes)
    need = target - counts(c);
    if need == 0,  continue,  end       % already at target size
    
    mask   = (Y_int ==  c);             % rows of *this* class
    Xc     = X_in(mask, :);
    Cc     = c * ones(sum(mask), 1);    % numeric class vector
    
    ratio  = need / counts(c);          % e.g. 2.4 → +240 %
    [X_bal, C_bal] = smote(Xc, ratio, k, 'Class', Cc);
    
    % synthetic rows sit after the originals -----------------------------
    X_syn = X_bal(size(Xc,1)+1:end, :);
    nSyn  = size(X_syn,1);
    
    % -- 3. Map back to categorical --------------------------------------
    label_cat = categorical(classes(c), classes);      % single label
    Y_syn     = repmat(label_cat, nSyn, 1);            % expand to column
    
    X_out = [X_out; X_syn];
    Y_out = [Y_out; Y_syn];
end
end
