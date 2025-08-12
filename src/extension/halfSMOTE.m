function [X_out, Y_out] = halfSMOTE(X_in, Y_in, k)
% HALFSMOTE (Corrected): Balances each minority class halfway to the majority count.
%
% This corrected version calculates the target count for each class individually,
% ensuring a true "halfway" balancing strategy.

% --- 1. Defaults & Input Validation ---
if nargin < 3, k = 5; end
assert(size(X_in, 1) == numel(Y_in), 'X_in and Y_in must have the same number of rows.');

if iscategorical(Y_in)
    Ycat = Y_in;
else
    Ycat = categorical(Y_in);
end
classes = categories(Ycat);
counts = countcats(Ycat);
majority_count = max(counts);

% --- 2. Initialize Augmented Dataset with Originals ---
X_out = X_in;
Y_out = Ycat;

% --- 3. Oversample Each Minority Class to its "Halfway" Target ---
fprintf('Applying correct Half-SMOTE logic...\n');
for c = 1:numel(classes)
    
    n_orig = counts(c);
    
    % Calculate the true "halfway" target for this specific class
    target_count = round(n_orig + (majority_count - n_orig) / 2);
    
    need_to_add = target_count - n_orig;
    
    if need_to_add <= 0
        continue; % Skip majority classes or classes already past the halfway mark
    end
    
    % Isolate the data for the current minority class
    mask = (Ycat == classes{c});
    Xc = X_in(mask, :);
    
    % Calculate the required ratio for the smote function
    ratio = need_to_add / n_orig;
    
    % Call the base SMOTE function to generate ONLY synthetic samples
    [~, ~, X_syn] = smote(Xc, ratio, k);
    Y_syn = repmat(classes(c), size(X_syn, 1), 1);
    
    % Append the new synthetic data
    X_out = [X_out; X_syn];
    Y_out = [Y_out; Y_syn];
end
end