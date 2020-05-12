function [OUT] = dummy_classify(DATA,PAR)

% --- Dummy Classifier (output is the most likely a priori class ) ---
%
%   [OUT] = dummy_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix                   [p x N]
%       PAR.
%           Nc = number of classes from problem         [cte]
%           class = most likely a priori class          [cte]
%   Output:
%       OUT.
%           y_h = classifier's output                   [Nc x N]

%% INITIALIZATIONS

X = DATA.input;         % input matrix
[~,N] = size(X);        % Number of samples

class = PAR.class;      % Class of outputs
Nc = PAR.Nc;            % Number of classes from problem

y_h = -1*ones(Nc,N);    % Init predict output

%% ALGORITHM

% Function output
y_h(class,:) = 1;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END