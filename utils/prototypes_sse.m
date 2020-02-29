function [SSE] = prototypes_sse(C,DATA,PAR)

% --- Calculate the Sum of Squared errors of prototypes ---
%
%   [SSE] = prototypes_sse(C,DATA,PAR)
%
%   Input:
%       C = prototypes                              [p x Nk]
%       DATA.
%           input = input matrix                    [p x N]
%       PAR.
%           dist = Type of distance                 [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%   Output:
%       SSE = Sum of Squared Errors                 [cte]

%% INITIALIZATION

% Load Data

input = DATA.input;
[~,N] = size(input);

SSE = 0;    % accumulate value of squared error sum

%% ALGORITHM

for i = 1:N,
    xn = input(:,i);                  	% get a sample
    win = prototypes_win(C,xn,PAR);     % index of closest prototype
    Cx = C(:,win);                      % closest prototype
    SSE = SSE + sum((Cx - xn).^2);      % Sum of Squared Error
end

%% END