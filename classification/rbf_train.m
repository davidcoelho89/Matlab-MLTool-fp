function [PARout] = rbf_train(DATA,PAR)

% --- RBF Classifier Training ---
%
%   [PARout] = rbf_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                                	[p x N]
%           output = labels                                   	[Nc x N]
%       PAR.
%           Nh = number of hidden neurons                      	[cte]
%           init = type of centroids initialization           	[cte]
%               1: Forgy Method (randomly choose k observations)
%               2: Vector Quantization (kmeans)
%           rad = type of radius / spread calculation         	[cte]
%               1: Equal for all RBFs
%               2: Each RBF has a different radius / spread
%           out = way to calculate output weights           	[cte]
%               1: OLS
%               2: LMS
%               3: RLS
%               4: PS
%           ativ = activation function type                  	[cte]
%               1: gaussian 
%               2: multiquadratic
%               3: inverse multiquadratic 
%   Output:
%       PARout.
%           W = Hidden layer weight Matrix                     	[p x Nh]
%           r = radius of each rbf                           	[1 x Nh]
%           M = Output layer weight Matrix                   	[Nc x Nh+1]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.Nh = 10;  	% number of hidden neurons
    PARaux.init = 2; 	% centroids initialization type
    PARaux.rad = 2;  	% radius / spread type
    PARaux.out = 1;    	% how to calculate output weights
    PARaux.ativ = 1;   	% activation function type
    PAR = PARaux;
else
    if (~(isfield(PAR,'Nh'))),
        PAR.Nh = 10;
    end
    if (~(isfield(PAR,'init'))),
        PAR.init = 2;
    end
    if (~(isfield(PAR,'rad'))),
        PAR.rad = 2;
    end
    if (~(isfield(PAR,'out'))),
        PAR.out = 1;
    end
    if (~(isfield(PAR,'ativ'))),
        PAR.ativ = 1;
    end
end

%% ALGORITHM

% First Step: Centroids Determination

W = rbf_f_cent(DATA,PAR);           % select centroids of each rbf

% Second Step: Radius / spread of each centroid

r = rbf_f_radius(W,PAR);            % select spread of each rbf

% Third Step: Output layer weight Matrix

M = rbf_f_outweig(DATA,PAR,W,r);    % calculate output weights matrix

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;
PARout.r = r;
PARout.M = M;

%% END