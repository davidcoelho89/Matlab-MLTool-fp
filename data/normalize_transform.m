function [DATAout] = normalize_transform(DATAin,PAR)

% --- Normalize the raw data ---
%
%   [DATAout] = normalize_transform(DATAin,PAR)
%
%   Input:
%       DATAin.
%           input = data matrix                     [p x N]
%       PAR.
%           Xmin = Minimum value of attributes      [p x 1]
%           Xmax = Maximum value of attributes      [p x 1]
%           Xmed = Mean value of attributes         [p x 1]
%           Xdp = Standard Deviation of attributes  [p x 1]
%           norm = how will be the normalization
%               0: input_out = input_in
%               1: normalize between [0, 1]
%               2: normalize between [-1, +1]
%               3: normalize by z-score transformation
%                  (empirical mean = 0 and standard deviation = 1)
%                  Xnorm = (X-Xmean)/(std)
%   Output:
%       DATAout.
%           input = normalized matrix               [p x N]

%% INITIALIZATIONS

% Get normalization option and data matrix [N x p]

normalization_option = PAR.norm;   
Xin = DATAin.input';      

% Get number of samples and attributes

[N,p] = size(Xin);

% Get min, max, mean and standard deviation measures

Xmin = PAR.Xmin;
Xmax = PAR.Xmax;
Xmed = PAR.Xmed;
Xstd = PAR.Xstd;

% initialize data

Xout = zeros(N,p); 

%% ALGORITHM

switch normalization_option
    case (0)
        Xout = Xin;
    case (1)    % normalize between [0 e 1]
        for i = 1:N
            for j = 1:p
                Xout(i,j) = (Xin(i,j) - Xmin(j))/(Xmax(j) - Xmin(j)); 
            end
        end
    case (2)    % normalize between [-1 e +1]
        for i = 1:N
            for j = 1:p
                Xout(i,j) = 2*(Xin(i,j) - Xmin(j))/(Xmax(j) - Xmin(j)) - 1; 
            end
        end
    case (3)    % normalize by z-score transform (by mean and std)
        for i = 1:N
            for j = 1:p
                Xout(i,j) = (Xin(i,j) - Xmed(j))/Xstd(j); 
            end
        end
    otherwise
        Xout = Xin;
        disp('Choose a correct option. Data was not normalized.')
end

Xout = Xout'; % transpose data for [p x N] pattern

%% FILL OUTPUT STRUCTURE

DATAout = DATAin;
DATAout.input = Xout;

%% END