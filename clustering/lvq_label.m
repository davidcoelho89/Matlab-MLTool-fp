function [PARout] = lvq_label(DATA,OUT_CL)

% --- LVQ Labeling Function ---
%
%   [PARout] = lvq_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [c x N]
%       OUT_CL.
%           C = prototypes [p x Neu]
%           index = [1 x N]
%           SSE = [1 x Nep]
%   Output:
%       PARout.
%           C = prototypes [p x Neu]
%           label = class of each neuron [1 x Neu]
%           index = [1 x N]
%           SSE = [1 x Nep]

%% ALGORITHM

[PARout] = prototypes_label(DATA,OUT_CL);

%% END