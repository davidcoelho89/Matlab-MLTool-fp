function isStationary = stationarityTest(x,nparts,tol,TAUmax)

% --- Verify if a signal is stationary ---
%
%   isStationary = stationarityTest(x,nparts,tol,TAUmax)
%
%   Input:
%       x = Sampled signal                          [1 x Ns]
%       nparts = number of new vectors              [cte]
%       tol = limit used to indicate Stationarity	[cte]
%   Output:
%       isStationary = boolean value              	[0 or 1]

%% INITIALIZATIONS



%% ALGORITHM

isStationaryMean = stationarityMeanTest(x,nparts,tol);
isStationaryVar =  stationarityVarTest(x,nparts,tol);
isStationaryAcf =  stationarityAcfTest(x,nparts,tol,TAUmax);

disp(strcat(int2str(isStationaryMean),int2str(isStationaryVar),...
                    int2str(isStationaryAcf)));

if(isStationaryMean && isStationaryVar && isStationaryAcf)
    isStationary = 1;
else
    isStationary = 0;
end

%% END















