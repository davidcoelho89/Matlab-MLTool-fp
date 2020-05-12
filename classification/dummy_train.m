function [PAR] = dummy_train(DATA)

% --- Dummy Classifier (output is the most likely a priori class ) ---
%
%   [PAR] = dummy_train(DATA)
%
%   Input:
%       DATA.
%           input = attributes matrix                   [p x N]
%           output = labels matrix                      [Nc x N]
%   Output:
%       PAR.
%           Nc = number of classes                      [cte]
%           class = most likely a priori class          [cte]

%% SET DEFAULT HYPERPARAMETERS

% Don't have hyperparameters

%% INITIALIZATIONS

% Get Data
Y = DATA.output;                % Output matrix
[Nc,~] = size(Y);               % Number of Classes

vector_priori = zeros(1,Nc);    % Hold number of samples from each class

%% ALGORITHM

[~,Ylbl] = max(Y);

for c = 1:Nc,
    vector_priori(c) = sum(Ylbl == c);
end

[~,class] = max(vector_priori);

%% FILL OUTPUT STRUCTURE

PAR.class = class;
PAR.Nc = Nc;

%% END