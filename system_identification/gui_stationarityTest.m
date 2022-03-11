function isStationary = stationarityTest_gui(x,nparts,tol)

% --- Verify if a signal is stationary ---
%
%   isStationary = stationarityTest_gui(x,nparts,tol)
%
%   Input:
%       x = Sampled signal                          [1 x Ns]
%       nparts = number of new vectors              [cte]
%       tol = limit used to indicate Stationarity	[cte]
%   Output:
%       isStationary = boolean value              	[0 or 1]

%% INITIALIZATIONS

Npoints = length(x);
parts_length = fix(Npoints/nparts);

%% ALGORITHM

% Divide signal in N parts
X = zeros(nparts,parts_length);
for i = 1:nparts
    X(i,:) = x( (i-1)*parts_length + 1 : (i)*parts_length );
end

mx_ti = mean(X);
var_ti = var(X,0,1);

M = mean(mx_ti);
V = mean(var_ti);

x_mean = mean(x);
sx = var(x,0);

if (abs(M - x_mean) < tol && abs(V - sx) < tol )
    isStationary = 1;
else
    isStationary = 0;
end

%% END