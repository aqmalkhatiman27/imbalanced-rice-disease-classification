function [X_out, Y_out] = borderlineSMOTE(X_in, Y_in, k)
% BORDERLINESMOTE – class-balanced oversampling using Michio Inoue's
% `myBorderlineSMOTE.m` helper (table-based implementation).
%
%   [X_out, Y_out] = borderlineSMOTE(X_in, Y_in)
%   [X_out, Y_out] = borderlineSMOTE(X_in, Y_in, k)   % k = #neighbours
%
% * X_in : N×D numeric matrix   (features)
% * Y_in : N×1 categorical      (labels)
% * k    : neighbours for the danger test (default 5)
%
% Returns the original data **plus** the synthetic rows, so the size of
% every class equals the majority-class count.

if nargin < 3,  k = 5;  end
assert(size(X_in,1) == numel(Y_in), 'X/Y size mismatch');

% ---- 1. Ensure labels are categorical ----------------------------------
if ~iscategorical(Y_in)
    Ycat = categorical(Y_in);
else
    Ycat = Y_in;
end

classes = categories(Ycat);
Y_int   = double(Ycat);                % numeric codes 1..C
counts  = accumarray(Y_int,1);

% target count = majority size  (reuse central rule if present)
if exist('phase6_getTargetCount','file')
    target = phase6_getTargetCount(counts,'full');
else
    target = max(counts);
end

% pre-allocate output (start with originals)
X_out = X_in;
Y_out = Ycat;

% ---- 2. Work class-by-class with Inoue's table routine -----------------
for c = 1:numel(classes)
    curMask   = (Y_int == c);
    curCount  = counts(c);

    % ----- keep looping until this class hits the target ---------------
    while curCount < target
        need = target - curCount;

        % ---- 1.  isolate current rows & build table -------------------
        Xc        = X_out(curMask,:);                 % NOTE: use X_out
        featNames = "F" + string(1:size(Xc,2));
        tbl       = array2table(Xc,'VariableNames',featNames);

        lblStr    = string(classes{c});               % scalar string
        tbl.Class = repmat(lblStr,height(tbl),1);     % right-most col.

        % ---- 2.  try Borderline-SMOTE first ---------------------------
        [tblNew,~] = myBorderlineSMOTE(tbl, lblStr, need);

        % ---- 3.  if nothing generated, fall back to plain SMOTE -------
        if isempty(tblNew)
            ratio   = need / curCount;                % e.g. 0.7
            [X_bal,~] = smote(Xc, ratio, k);          % Bjarke Larsen's
            X_syn  = X_bal(size(Xc,1)+1:end,:);
            Y_syn  = repmat(classes(c), size(X_syn,1), 1);
        else
            X_syn  = tblNew{:,1:end-1};
            Y_syn  = categorical(tblNew.Class, classes);
        end

        % ---- 4.  update the accumulators ------------------------------
        X_out   = [X_out ; X_syn];
        Y_out   = [Y_out ; Y_syn];

        % ---- 5.  recompute mask / count for this class ----------------
        curMask  = (double(Y_out) == c);
        curCount = sum(curMask);
    end
end
end
