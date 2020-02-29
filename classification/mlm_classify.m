function [OUT] = mlm_classify(DATA,PAR)

% --- MLM Classifier Test ---
%
%   [OUT] = mlm_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix       [p x N]
%       PAR.
%           B = Regression matrix           [K x K]
%           Rx = Input reference points   	[K x p]
%           Ty = Output reference points   	[K x Nc]
%   Output:
%       OUT.
%           y_h = classifier's output       [c x N]

%% INITIALIZATIONS

% Data Initialization
X = DATA.input';    % Input matrix [N x p]
[N,~] = size(X);    % Number of samples

% Get Parameters
B = PAR.B;          % Regression matrix 
Rx = PAR.Rx;        % Input reference points
Ty = PAR.Ty;        % Output reference points
[K,Nc] = size(Ty);  % Number of reference points and outputs

% Init outputs
y_h = zeros(N,Nc);  % One output for each sample

%% ALGORITHM

% Input distance matrix of reference points

Dx = zeros(N,K);                         	% Init Dx [N x K]

for n = 1:N
    xi = X(n,:);                         	% Get input sample
    for k = 1:K
        mk = Rx(k,:);                      	% Get input reference point
        Dx(n,k) = vectors_dist(xi,mk,PAR);	% Calculates distance
    end
end

% Output distance matrix of reference points

Dy_h = Dx*B;                                % [N x k]

% Options for fsolve algorithm

opts = optimoptions('fsolve','Algorithm','levenberg-marquardt', ...
          'Display', 'off', 'FunValCheck', 'on', 'TolFun', 10e-10);

% fsolve = solve non-linear equations system

for i = 1:N, 
    y_h(i,:) = fsolve(@(x)(sum((Ty - repmat(x,K,1)).^2,2) - Dy_h(i,:)'.^2).^2, ...
                      zeros(1,Nc), opts);
end

% Adjust y_h for [Nc x N] pattern

y_h = y_h';

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END