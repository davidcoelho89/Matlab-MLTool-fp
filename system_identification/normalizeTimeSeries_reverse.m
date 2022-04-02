function DATAout = normalizeTimeSeries_reverse(DATAin,PAR)

% --- Reverse the Time Series amplitudes to the original values ---
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
	Unorm = DATAin.input;
	[Nu,N] = size(Unorm);
    U = zeros(Nu,N);
end

if(isfield(DATAin,'output'))
	Ynorm = DATAin.output;
	[Ny,N] = size(Ynorm);
    Y = zeros(Ny,N);
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
            U = Unorm;
        end
        if(isfield(DATAin,'output'))
            Y = Ynorm;
        end
	case (1)    % denormalize between [0 e 1]
        if(isfield(DATAin,'input'))
            for i = 1:Nu
                for j = 1:N
                    U(i,j) = Unorm(i,j)*(Umax(i) - Umin(i)) + Umin(i);
                end
            end
        end
        if(isfield(DATAin,'output'))
            for i = 1:Ny
                for j = 1:N
                    Y(i,j) = Ynorm(i,j)*(Ymax(i) - Ymin(i)) + Ymin(i);
                end
            end
        end
	case (2)    % denormalize between [-1 e +1]
        if(isfield(DATAin,'input'))
            for i = 1:Nu
                for j = 1:N
                    U(i,j) = 0.5*(Unorm(i,j) + 1)*(Umax(i) - Umin(i)) + Umin(i); 
                end
            end
        end
        if(isfield(DATAin,'output'))
            for i = 1:Ny
                for j = 1:N
                    Y(i,j) = 0.5*(Ynorm(i,j) + 1)*(Ymax(i) - Ymin(i)) + Ymin(i);
                end
            end
        end
	case (3) % denormalize by z-score transform (by mean and std)
        if(isfield(DATAin,'input'))
            for i = 1:Nu
                for j = 1:N
                    U(i,j) = Unorm(i,j)*Ustd(i) + Umed(i);
                end
            end
        end
        if(isfield(DATAin,'output'))
            for i = 1:Ny
                for j = 1:N
                    Y(i,j) = Ynorm(i,j)*Ystd(i) + Ymed(i);
                end
            end
        end
	case (4) % denormalize by z-score transform (by mean and 3*std)
        if(isfield(DATAin,'input'))
            for i = 1:Nu
                for j = 1:N
                    U(i,j) = Unorm(i,j)*Ustd(i)*3 + Umed(i);
                end
            end
        end
        if(isfield(DATAin,'output'))
            for i = 1:Ny
                for j = 1:N
                    Y(i,j) = Ynorm(i,j)*Ystd(i)*3 + Ymed(i);
                end
            end
        end
    otherwise
        disp('Choose a correct option. Data was not normalized.')
        if(isfield(DATAin,'input'))
            U = Unorm;
        end
        if(isfield(DATAin,'output'))
            Y = Ynorm;
        end        
end

%% FILL OUTPUT STRUCTURE

DATAout = DATAin;
if(isfield(DATAin,'input'))
    DATAout.input = U;
end
if(isfield(DATAin,'output'))
    DATAout.output = Y;
end

%% END