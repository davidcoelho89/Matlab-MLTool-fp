function [Yi] = elm_f_ativ(Ui,option)

% --- ELM Classifier activation function  ---
%
%   [Yi] = elm_f_ativ(Ui,option)
%
%   Input:
%       Ui = neuron activation
%       option = type of activation function
%           = 1: Sigmoidal -> output: [0 e 1]
%           = 2: Hyperbolic Tangent -> output: [-1 e +1]
%   Output:
%       Yi = output of non-linear function

%% ALGORITHM

[Yi] = mlp_f_ativ(Ui,option);

%% END