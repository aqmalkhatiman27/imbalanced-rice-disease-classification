function metrics = runSVMsuite(Xtr, Ytr, Xte, Yte, tag, varargin)
% Train L/Q/C SVMs. Includes optional 'Cost' and single-kernel 'Spec'
% parameters. Saves one final results file.

% --- Parse optional 'Cost' and 'Spec' name-value pair inputs ---
p = inputParser;
addParameter(p, 'Cost', []); % Default cost matrix is empty
addParameter(p, 'Spec', []); % Default kernel spec is empty
parse(p, varargin{:});
cost_matrix = p.Results.Cost;
userSpec    = p.Results.Spec;

% --- Feature Scaling ---
mu    = mean(Xtr, 1);
sigma = std(Xtr, 0, 1) + 1e-8;
Xtr   = (Xtr - mu) ./ sigma;
Xte   = (Xte - mu) ./ sigma;

% --- Kernel Specifications ---
if isempty(userSpec)
    % If no specific kernel is passed, default to all three
    spec = { ...
       struct('name','linear',   'order',1); ...
       struct('name','quadratic','order',2); ...
       struct('name','cubic',    'order',3) };
else
    % Otherwise, use only the single kernel spec that was passed in
    spec = { userSpec };
end

rng(20250622,'twister');
metrics = []; % Initialize empty output

% --- Train and Evaluate Each SVM Kernel ---
for k = 1:numel(spec)
    tpl = templateSVM('KernelFunction', 'polynomial', ...
                      'PolynomialOrder', spec{k}.order, ...
                      'KernelScale',     'auto', ...
                      'Standardize',     false);
    
    % Conditionally build the training command with or without cost
    if isempty(cost_matrix)
        mdl = fitcecoc(Xtr, Ytr, 'Learners', tpl, 'Coding', 'onevsall', 'Verbose', 0);
    else
        mdl = fitcecoc(Xtr, Ytr, 'Learners', tpl, 'Coding', 'onevsall', 'Cost', cost_matrix, 'Verbose', 0);
    end

    Ypred = predict(mdl, Xte);
    C     = confusionmat(Yte, Ypred);
    
    current_metric.kernel  = spec{k}.name;
    current_metric.acc     = mean(Ypred == Yte);
    current_metric.confMat = C;
    metrics = [metrics; current_metric]; % Append result to the metrics array
end

% --- Save the combined results once, outside the loop ---
outDir = fullfile(pwd, 'results');
if ~exist(outDir, 'dir'); mkdir(outDir); end
save(fullfile(outDir, sprintf('%s_results.mat', tag)), 'metrics');

end