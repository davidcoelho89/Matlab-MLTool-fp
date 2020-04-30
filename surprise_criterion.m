function [SURout] = surprise_criterion(Dx,Dy,xt,yt,Kinv,HP)

% --- Apply the surprise criterion between a dictionary and a sample ---
%
%   [SURout] = surprise_criterion(Dx,xt,Kinv,HP)
%
%   Input:
%       Dx = dictionary prototypes' inputs                      [p x Nk]
%       Dy = dictionary prototypes' outputs                     [Nc x Nk]
%       xt = input of sample to be tested                       [p x 1]
%       yt = input of sample to be tested                       [p x 1]
%       HP.
%           v1 = Sparseness parameter 1                         [cte]
%           v2 = Sparseness parameter 2                         [cte]
%
%   Output:
%       ALDout.
%           result = if a sample fulfill the test               [0 or 1]
%           ktt = kernel function between sample and itself     [cte]
%           kt = kernel function between sample and dict prot   [Nk x 1]
%           at = ald coefficients                               [Nk x 1]
%           delta = constant compared with ald constant         [cte]

%% INITIALIZATIONS

[~,m] = size(Dx);               % Dictionary size
sig2n = HP.sig2n;           	% Kernel regularization parameter
v1 = HP.v1;                 	% Sparsification parameter
% v2 = HP.v2;                     % Sparseness parameter 2

%% ALGORITHM

% Calculate ktt (same as ALD)
ktt = kernel_func(xt,xt,HP);

% Calculate h(t) (same as k(t) from ALD)
ht = zeros(m,1);
for i = 1:m,
    ht(i) = kernel_func(Dx(:,i),xt,HP);
end

% Estimated variance ( sig2 = sig2n + ktt - ( ht' / Gt ) * ht )
% (same as ald coefficients and delta of ALD)
at = Kinv * ht;
sig2 = sig2n + ktt -  ht' * at;

% Estimated output ( y_h = ( ht' / Gt ) * Dy' ) (from GP)
y_h = (ht' * Kinv) * Dy';

% Distance between estimated output and real output
y_dist = (norm(y_h' - yt,2)^2);

% Surprise measure
Si = log(sqrt(sig2)) + (y_dist) / (2 * sig2);

% Calculate Criterion
result = (Si >= v1);

%% FILL OUTPUT STRUCTURE

SURout.result = result;
SURout.ktt = ktt;
SURout.kt = ht;
SURout.at = at;
SURout.Si = Si;

%% END