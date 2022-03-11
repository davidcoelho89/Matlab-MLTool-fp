function isStationary = stationarityAcfTest(x,nparts,tol,TAUmax)

% --- Verify AutoCorrelation Condition for Stationarity ---
%
%   isStationary = stationarityAcfTest(x,nparts,tol,TAUmax)
%
%   Input:
%       x = Sampled signal                                  [1 x Ns]
%       nparts = number of new vectors                      [cte]
%       tol = used to indicate AutoCorrelation stationarity	[cte]
%       TAUmax = maximum time lapse (lag)                   [cte]
%   Output:
%       isStationary = Verification                         [0 or 1]

%% INITIALIZATIONS

Npoints = length(x);
parts_length = fix(Npoints/nparts);

%% ALGORITHM

% Divide signal in N parts and calculate Autocovariance Vectors
X = zeros(nparts,parts_length);
Px = zeros(nparts,TAUmax+1);
for i = 1:nparts
    X(i,:) = x( (i-1)*parts_length + 1 : (i)*parts_length );
    Px(i,:) = acvcf(X(i,:),TAUmax);
end

% Verify if AutoCorrelation Functions are similar
diffs = zeros(nparts,nparts);
for i = 1:nparts
    pxi = Px(i,:);
    for j = i:nparts
        pxj = Px(j,:);
        diffs(i,j) = sum(abs(pxi-pxj));
        diffs(j,i) = diffs(i,j);
    end
end
diffs = diffs/(TAUmax+1);
max_diff = max(max(diffs));

disp(max_diff);

if (max_diff < tol)
    isStationary = 1;
else
    isStationary = 0;
end

%% END