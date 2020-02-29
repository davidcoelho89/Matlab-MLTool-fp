function [M] = rbf_f_outweig(DATA,PAR,W,r)

% --- Calculates Output Matrix Weights ---
%
%   [M] = rbf_f_outweig(DATA,PAR,W,r)
%
%   Input:
%       DATA.
%           input = input matrix                        [p x N]
%           output = labels                             [Nc x N]
%       PAR.
%           Nh = number of hidden neurons (centroids)   [cte]
%           ativ = activation function type             [cte]
%               1: gaussian 
%               2: multiquadratic
%               3: inverse multiquadratic 
%           out = way to calculate output weights       [cte]
%               1: OLS
%               2: LMS
%               3: RLS
%               4: PS
%       W = centroids of radial basis function          [p x Nh]
%       r = spread of each centroid                     [1 x Nh]
%   Output:
%       M = Output layer weight Matrix                  [No x Nh+1]

%% INITIALIZATIONS

% Get Data

X = DATA.input;
Y = DATA.output;
[No,N] = size(Y);

% Get Parameters

Nh = PAR.Nh;
out_type = PAR.out;
ativ = PAR.ativ;

% Initialize Output Matrix Weights

M = zeros(No,Nh+1);

%% ALGORITHM

% Calculates hidden layer activations

Z = zeros(Nh+1,N);          	% Output of hidden layer

for t = 1:N
    x = X(:,t);                 % get sample
    z_vec = zeros(Nh,1);      	% init output of basis functions
    for i = 1:Nh,
        Ci = W(:,i);            % Get rbf center
        ui = sum((x - Ci).^2);  % Calculates distance
        ri = r(i);              % get radius
        z_vec(i) = rbf_f_ativ(ui,ri,ativ);
    end
    z_vec = z_vec/sum(z_vec);  	% normalize outputs
    Z(:,t) = [1;z_vec];        	% add bias to z vector
end

% Calculates

if (out_type == 1) % Through OLS
    % M = D*Z'*inv(Z*Z');     	% direct calculus of inverse
    % M = D/Z;                  % QR factorization
    M = Y*pinv(Z);          	% uses SVD to estimate M matrix

elseif (out_type == 2) % Through LMS (Adaline)
    % ToDo - all
    % Obs: "Output from Output Layer" belongs to real numbers)
    
else
    disp('Unknown initialization. Output Weights = 0.');
    
end

%% END