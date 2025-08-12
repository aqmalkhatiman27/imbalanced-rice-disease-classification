function Xred = dtcwt_reduce(Xfull)
% Reduce Layer-1 vectors with a 2-level dual-tree complex wavelet transform.
% Uses Wavelet Toolbox's dualtree() / dtcwt() depending on release.
%
%  Xfull : [N × D]  single  — flattened conv-map rows
%  Xred  : [N × D/2] single — level-2 low-pass coefficients

[N,D] = size(Xfull);
J = 2;                                     % two levels (paper §3.2.3)

Xred = zeros(N, D/2, 'single');

for n = 1:N
    % ---- call Wavelet Toolbox ----
    % MATLAB R2023a+   :  dtcwt(…,Type="dual-tree")
    % Older releases   :  dualtree()
    try
        [A,~] = dtcwt(double(Xfull(n,:)),Level=J,Type="dual-tree");
    catch
        [A,~] = dualtree(double(Xfull(n,:)),'Level',J,'FilterLength',14);
    end

    % A is numeric (new toolbox) *or* cell array (old)
    if iscell(A)
        Xred(n,:) = single(A{J+1});        % old syntax
    else
        Xred(n,:) = single(A);             % new syntax returns LL directly
    end
end
end
