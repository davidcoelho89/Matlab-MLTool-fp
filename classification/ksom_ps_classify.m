function [OUT] = ksom_ps_classify(DATA,PAR)

% --- KSOM-PS Classify Function ---
%
%   [OUT] = ksom_ps_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                            [p x N]
%       PAR.
%           Cx = prototypes' attributes            	[p x Nk(1) x ... x Nk(Nd)]
%           Cy = prototypes' labels                 [Nc x Nk(1) x ... x Nk(Nd)]
%           Mred = matrix with M samples of training data   [p x m]
%           eig_vec = eigenvectors of reduced kernel matrix [M x m]
%           eig_val = eigenvalues of reduced kernel matrix  [1 x m]
%   Output:
%       OUT.
%           y_h = classifier's output                       [Nc x N]

%% INITIALIZATION

% Get Data
X = DATA.input;
[~,N] = size(X);

% Get auxiliary Variables
Mred = PAR.Mred;
[~,M] = size(Mred);
eig_vec = PAR.eig_vec;
eig_val = PAR.eig_val;

%% ALGORITHM

% Map samples to feature space

Mphi = zeros(M,N);

for n = 1:N,
    xn = X(:,n);
    for i = 1:M,
        sum_aux = 0;
        for m = 1:M,
            zm = Mred(:,m);
            k_xz = kernel_func(xn,zm,PAR);          % kernel between Xn an Zm
            sum_aux = sum_aux + eig_vec(m,i)*k_xz;  % calculate sum
        end
        Mphi(i,n) = sum_aux / sqrt(eig_val(i));     % atributte of mapped vector
    end
end

% Classify at approximated feature space, using non-kernelized distances

DATA.input = Mphi;
PAR.Ktype = 0;

[OUT] = prototypes_class(DATA,PAR);

%% END