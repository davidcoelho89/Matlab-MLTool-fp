function DATAout = normalizeTimeSeries_transform(DATAin,PAR)

% --- Normalize Time Series to a Specific Range ---
%
%	DATAout = normalizeTimeSeries_transform(DATAin,PAR)
%
%   Input:
%       DATAin.
%           input = Matrix with input time series     	[Nu x N]
%           output = Matrix with output time series    	[Ny x N]
%       PAR.
%           Umin = Minimum value of inputs              [Nu x 1]
%           Umax = Maximum value of inputs              [Nu x 1]
%           Umed = Mean value of inputs                 [Nu x 1]
%           Ustd = Standard Deviation of inputs         [Nu x 1]
%           Ymin = Minimum value of outputs             [Ny x 1]
%           Ymax = Maximum value of outputs             [Ny x 1]
%           Ymed = Mean value of outputs                [Ny x 1]
%           Ystd = Standard Deviation of outputs        [Ny x 1]
%           norm_type = how will be the normalization   [cte]
%               0: input_out = input_in
%               1: normalize between [0, 1]
%               2: normalize between [-1, +1]
%               3: normalize by z-score transformation (standardzation)
%                  (empirical mean = 0 and standard deviation = 1)
%                  Xnorm = (X-Xmean)/(std)
%               4: Xnorm = (X-Xmean)/(3*std)
%   Output:
%       DATAout.
%           input = Matrix with input time series     	[Nu x N]
%           output = Matrix with output time series   	[Ny x N]

%% INIT

% Get normalization option

norm_type = PAR.norm_type;

% Get data matrices

if(isfield(DATAin,'input'))
	U = DATAin.input;
	[Nu,N] = size(U);
    Unorm = zeros(Nu,N);
end

if(isfield(DATAin,'output'))
	Y = DATAin.output;
	[Ny,N] = size(Y);
    Ynorm = zeros(Ny,N);
end

% Get parameters

Umin = PAR.Umin;
Umax = PAR.Umax;
Umed = PAR.Umed;
Ustd = PAR.Ustd;
Ymin = PAR.Ymin;
Ymax = PAR.Ymax;
Ymed = PAR.Ymed;
Ystd = PAR.Ystd;

%% ALGORITHM

switch norm_type
	
	case (0)
        if(isfield(DATAin,'input'))
            Unorm = U;
        end
        if(isfield(DATAin,'output'))
            Ynorm = Y;
        end
	case (1)    % normalize between [0 e 1]
        if(isfield(DATAin,'input'))
            for i = 1:Nu
                for j = 1:N
                    Unorm(i,j) = (U(i,j) - Umin(i))/(Umax(i) - Umin(i));
                end
            end
        end
        if(isfield(DATAin,'output'))
            for i = 1:Ny
                for j = 1:N
                    Ynorm(i,j) = (Y(i,j) - Ymin(i))/(Ymax(i) - Ymin(i));
                end
            end
        end
	case (2)    % normalize between [-1 e +1]
        if(isfield(DATAin,'input'))
            for i = 1:Nu
                for j = 1:N
                    Unorm(i,j) = 2*(U(i,j) - Umin(i))/(Umax(i) - Umin(i)) - 1; 
                end
            end
        end
        if(isfield(DATAin,'output'))
            for i = 1:Ny
                for j = 1:N
                    Ynorm(i,j) = 2*(Y(i,j) - Ymin(i))/(Ymax(i) - Ymin(i)) - 1;
                end
            end
        end
	case (3) % normalize by z-score transform (by mean and std)
        if(isfield(DATAin,'input'))
            for i = 1:Nu
                for j = 1:N
                    Unorm(i,j) = (U(i,j) - Umed(i))/Ustd(i);
                end
            end
        end
        if(isfield(DATAin,'output'))
            for i = 1:Ny
                for j = 1:N
                    Ynorm(i,j) = (Y(i,j) - Ymed(i))/Ystd(i);
                end
            end
        end
	case (4) % normalize by z-score transform (by mean and 3*std)
        if(isfield(DATAin,'input'))
            for i = 1:Nu
                for j = 1:N
                    Unorm(i,j) = (U(i,j) - Umed(i))/(3*Ustd(i));
                end
            end
        end
        if(isfield(DATAin,'output'))
            for i = 1:Ny
                for j = 1:N
                    Ynorm(i,j) = (Y(i,j) - Ymed(i))/(3*Ystd(i));
                end
            end
        end
    otherwise
        disp('Choose a correct option. Data was not normalized.')
        if(isfield(DATAin,'input'))
            Unorm = U;
        end
        if(isfield(DATAin,'output'))
            Ynorm = Y;
        end
end

%% FILL OUTPUT STRUCTURE

DATAout = DATAin;
if(isfield(DATAin,'input'))
    DATAout.input = Unorm;
end
if(isfield(DATAin,'output'))
    DATAout.output = Ynorm;
end

%% END