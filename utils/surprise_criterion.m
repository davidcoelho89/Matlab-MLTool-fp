function [SURout] = surprise_criterion(Dx,Dy,xt,yt,HP,Kinv)

% --- Apply the surprise criterion between a dictionary and a sample ---
%
%   [SURout] = surprise_criterion(Dx,xt,Kinv,HP)
%
%   Input:
%       Dx = dictionary prototypes' inputs                      [p x Q]
%       Dy = dictionary prototypes' outputs                     [Nc x Q]
%       xt = input of sample to be tested                       [p x 1]
%       yt = input of sample to be tested                       [p x 1]
%       HP.
%           v1 = Sparseness parameter 1                         [cte]
%           v2 = Sparseness parameter 2                         [cte]
%           sig2n = Kernel regularization parameter             [cte]
%       Kinv = inverse kernel matrix                            [Q x Q]
%   Output:
%       SURout.
%           result = if a sample fulfill the test               [0 or 1]
%           ktt = kernel function between sample and itself     [cte]
%           kt = kernel function between sample and dict prot   [Q x 1]
%           at = ald coefficients                               [Q x 1]
%           Si = Surprise measure                               [cte]
%           sig2 = estimated variance of input                  [cte]
%           y_dist = distance between outputs                   [cte]

%% INITIALIZATIONS

v1 = HP.v1;        	% Sparseness parameter 1
v2 = HP.v2;        	% Sparseness parameter 2
sig2n = HP.sig2n; 	% Kernel regularization parameter

%% ALGORITHM

% Calculate ktt (same as ALD)
ktt = kernel_func(xt,xt,HP);

% Calculate h(t) (same as k(t) from ALD)
ht = kernel_vect(Dx,xt,HP);

% Estimated variance ( sig2 = sig2n + ktt - ( ht' / Gt ) * ht )
% (same as ald coefficients and delta of ALD)
at = Kinv * ht;
sig2 = sig2n + ktt -  ht' * at;

% Estimated output 1: ( yh = ( ht' / Gt ) * Dy' ) (from GP)
% yh = (ht' * Kinv) * Dy';
% yh = yh';

% Estimated output 2: NN or KNN (from PBC)
DATA.input = xt;
OUT = prototypes_class(DATA,HP);
yh = OUT.y_h;

% Distance between estimated output and real output
y_dist = sqrt(sum((yt - yh).^2));

% Surprise measure
Si = log(sqrt(sig2)) + (y_dist) / (2 * sig2);

% Calculate Criterion
result = (Si >= v1 && Si <= v2);

%% FILL OUTPUT STRUCTURE

SURout.result = result;
SURout.ktt = ktt;
SURout.kt = ht;
SURout.at = at;
SURout.Si = Si;
SURout.sig2 = sig2;
SURout.y_dist = y_dist;

%% END