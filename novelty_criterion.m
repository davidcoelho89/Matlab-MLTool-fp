function [NOVout] = novelty_criterion(Dx,Dy,xt,yt,HP)

% --- Apply the novelty criterion between a dictionary and a sample ---
%
%   [NOVout] = novelty_criterion(Dx,Dy,xt,yt,HP)
%
%   Input:
%       Dx = dictionary prototypes' inputs                      [p x Nk]
%       Dy = dictionary prototypes' outputs                     [Nc x Nk]
%       xt = input of sample to be tested                       [p x 1]
%       yt = input of sample to be tested                       [p x 1]
%       HP.
%           v1 = Sparseness parameter 1                         [cte]
%           v2 = Sparseness parameter 2                         [cte]
%   Output:
%       ALDout.
%           result = if a sample fulfill the test               [0 or 1]
%           result1 = novelty test 1                            [cte]
%           result2 = novelty test 2                            [cte]
%           dist1 = novelty measure 1                           [cte]
%           dist2 = novelty measure 2                           [cte]
%           kt = kernel function between sample and dict prot   [Nk x 1]
%           at = ald coefficients                               [Nk x 1]
%           delta = constant compared with ald constant         [cte]

%% INITIALIZATIONS

v1 = HP.v1;                 	% Sparseness parameter 1
% v2 = HP.v2;                     % Sparseness parameter 2

%% ALGORITHM

% Apply prototypes' classification function

HP.Cx = Dx; HP.Cy = Dy;             % Get current dictionary
DATA.input = xt;                    % Get current input
OUT = prototypes_class(DATA,HP);    % Output of classification
yh = OUT.y_h;                       % Output Prediction

% Find nearest prototype to sample
win = OUT.win;

% Get distance between sample and nearest prototype 
dist1 = OUT.dist(win);

% First part of Criterion
result1 = (dist1 > v1);

% % Second part of Criterion (method 1 - expand dictionary if estimation
% % and real output are very diferrent from each other)
% dist2 = vectors_dist(yt,yh,PAR);
% result2 = (dist2 > v2);

% Second part of Criterion (method 2 - expand dictionary if the sample
% was misclassified)
[~,yh_seq] = max(yh);
[~,yt_seq] = max(yt);
dist2 = yh_seq;
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