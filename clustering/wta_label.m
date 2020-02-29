function [PARout] = wta_label(DATA,OUT_CL)

% --- WTA Labeling Function ---
%
%   [PARout] = wta_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix                    [p x Ntr]
%           output = output matrix                  [Nc x Ntr]
%       OUT_CL.
%           Cx = cluster prototypes             	[p x Nk]
%           dist = type of distance               	[cte]
%           ind = cluster index for each sample     [1 x Ntr]
%           lbl = type of labeling                  [cte]
%               1: Majority voting
%               2: Average distance
%               3: Minimum distance
%   Output:
%       PARout.
%       	Cx = clusters prototypes                [p x Nk]
%           Cy = class of each prototype            [Nc x Nk]

%% ALGORITHM

[PARout] = prototypes_label(DATA,OUT_CL);

%% END