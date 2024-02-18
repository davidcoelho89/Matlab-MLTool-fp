function [NOVout] = novelty_criterion(Dx,xt,yt,HP)

% --- Apply the novelty criterion between a dictionary and a sample ---
%
%   [NOVout] = novelty_criterion(Dx,Dy,xt,yt,HP)
%
%   Input:
%       Dx = dictionary prototypes' inputs                      [p x Q]
%       xt = input of sample to be tested                       [p x 1]
%       yt = input of sample to be tested                       [p x 1]
%       HP.
%           v1 = Sparseness parameter 1                         [cte]
%           v2 = Sparseness parameter 2                         [cte]
%   Output:
%       NOVout.
%           result = if a sample fulfill the test               [0 or 1]
%           result1 = novelty test 1                            [cte]
%           result2 = novelty test 2                            [cte]
%           dist1 = novelty measure 1                           [cte]
%           dist2 = novelty measure 2                           [cte]

%% INITIALIZATIONS

v1 = HP.v1;                 	% Sparseness parameter 1
v2 = HP.v2;                     % Sparseness parameter 2

%% ALGORITHM

% OBS: NEW IMPLEMENTATION!

% Get model's output (prediction, winner, distances)

DATA.input = xt;
OUT = prototypes_class(DATA,HP);

% 1st part - measure distance between sample and nearest prototype 

win = prototypes_win(Dx,xt,HP);
dist1 = vectors_dist(Dx(:,win),xt,HP);
result1 = (dist1 > v1);

% 2nd part - method 1 - Measure distance from outputs

% yh = OUT.y_h;
% dist2 = sqrt(sum((yt - yh).^2));
% % dist2 = vectors_dist(yt,yh,HP);
% result2 = (dist2 > v2);

% 2nd part - method 2 - Verify if sample was misclassified

yh = OUT.y_h;
[~,yh_seq] = max(yh);
[~,yt_seq] = max(yt);
dist2 = vectors_dist(yt,yh,HP);
result2 = (yt_seq ~= yh_seq);

% Calculate Criterion

result = (result1 && result2);

%% FILL OUTPUT STRUCTURE

NOVout.result = result;
NOVout.result1 = result1;
NOVout.result2 = result2;
NOVout.dist1 = dist1;
NOVout.dist2 = dist2;

%% END