function [ALDout] = ald_criterion(Dx,xt,HP,Kinv)

% --- Apply the ald criterion between a dictionary and a sample ---
%
%   [ALDout] = ald_criterion(Dx,xt,Kinv,HP)
%
%   Input:
%       Dx = dictionary prototypes' inputs                      [p x Nk]
%       xt = sample to be tested                                [p x 1]
%       HP.
%           v1 = Sparseness parameter 1                         [cte]
%           sig2n = Kernel regularization parameter             [cte]
%       Kinv = inverse kernel matrix                            [Nk x Nk]
%   Output:
%       ALDout.
%           result = if a sample fulfill the test               [0 or 1]
%           ktt = kernel function between sample and itself     [cte]
%           kt = kernel function between sample and dict prot   [Nk x 1]
%           at = ald coefficients                               [Nk x 1]
%           delta = ald measure                                 [cte]

%% INITIALIZATIONS

[~,m] = size(Dx);               % Dictionary size

v1 = HP.v1;                 	% Sparsification parameter

sig2n = HP.sig2n;           	% Kernel regularization parameter

%% ALGORITHM

% Calculate ktt
ktt = kernel_func(xt,xt,HP);

% Calculate kt
kt = zeros(m,1);
for i = 1:m,
    kt(i) = kernel_func(Dx(:,i),xt,HP);
end

% Calculate ald coefficients
at = Kinv*kt;

% Calculate delta
delta = ktt - kt'*at;

% "Normalized delta"
delta = delta + sig2n;

% Calculate Criterion
result = (delta > v1);

%% FILL OUTPUT STRUCTURE

ALDout.result = result;
ALDout.ktt = ktt;
ALDout.kt = kt;
ALDout.at = at;
ALDout.delta = delta;

%% END