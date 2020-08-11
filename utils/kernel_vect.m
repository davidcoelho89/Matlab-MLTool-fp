function [kt] = kernel_vect(X,xt,PAR)

% --- Calculate the kernel vector of a data matrix and a sample ---
%
%   [Kt] = kernel_vect(X,xi,PAR)
%
%   Input:
%       X = matrix with samples                                 [p x N]
%       xt = specific data sample                               [p x 1]
%       PAR.
%           Ktype = kernel type                                 [cte]
%               1 -> Linear 
%               2 -> Gaussian (default)
%               3 -> Polynomial
%               4 -> Exponencial / Laplacian
%               5 -> Cauchy
%               6 -> Log
%               7 -> Sigmoid
%               8 -> Kmod
%           sig2n = kernel regularization parameter             [cte]
%           sigma   (gauss / exp / cauchy / log / kmod)         [cte]
%           gamma   (poly / log / Kmod)                         [cte]
%           alpha   (poly / sigmoid)                            [cte]
%           theta   (lin / poly / sigmoid)                      [cte]
%   Output:
%       kt = kernel vector, where kt(j) = K(X(:,j),xt)          [N x 1]

%% INITIALIZATIONS

% Get number of samples
[~,N] = size(X);

% Initialize Kernel Matrix
kt = zeros(N,1);

%% ALGORITHM

for j = 1:N,
    kt(j) = kernel_func(X(:,j),xt,PAR);
end

%% END