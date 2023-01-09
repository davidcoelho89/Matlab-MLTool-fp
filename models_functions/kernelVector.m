function [kt] = kernelVector(X,y,model)

%
%  --- HELP about kernelVector ---
%

%% INITIALIZATIONS

[~,N] = size(X);
kt = zeros(N,1);

%% ALGORITHM

for j = 1:N
    kt(j) = kernelFunction(X(:,j),y,model);
end

end