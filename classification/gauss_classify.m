function [OUT] = gauss_classify(DATA,PAR)

% --- Gaussian classifier test ---
%
%   [OUT] = gauss_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix                           [p x N]
%       PAR.
%           Ni = number of "a priori samples" per class         [Nc x 1]
%           mu_i = centroid of each class                       [Nc x p]
%           Ci = covariance matrix of each class                [Nc x p x p]
%           type = type of gaussian classifier                  [cte]
%               1: gi(x) = -0.5Qi(x) - 0.5ln(det(Ci)) + ln(p(Ci))
%               2: gi(x) = -0.5Qi(x) - 0.5ln(det(Ci))
%               3: gi(x) = -0.5Qi(x) (mahalanobis distance)
%            	   (covariance matrix is the pooled covariance matrix)
%               4: gi(x) = -0.5||x-mi||^2 (euclidean distance)
%   Output:
%       OUT.
%           y_h = classifier's output matrix                    [Nc x N]

%% INITIALIZATIONS

% Get input matrix
X = DATA.input;

% Get parameters
Ni = PAR.Ni;
mu_i = PAR.mu_i;
Ci = PAR.Ci;
type = PAR.type;

% Put input in [N x p] pattern
X = X';
[N,p] = size(X);

% Number of classes do the problem
[Nc,~] = size(Ni);

% Count all traning samples (a priori samples)
Ntr = sum(Ni);

% Initialize estimated output matrix
y_h = zeros(Nc,N);

%% ALGORITHM

if type == 1,  % Complete Classifier

% Inverse of Covariance Matrix
Ci_inv = cell(1,Nc);
for i = 1:Nc,
    Ci_inv{i} = pinv(Ci{i});
end

for i = 1:N,

    % Get sample
    xi = X(i,:);

    % initialize discriminant function
    gi = zeros(Nc,1);
    
    for c = 1:Nc,
        % mahalanobis distance for each class
        MDc = (xi - mu_i(c,:))*Ci_inv{c}*(xi - mu_i(c,:))';
        % discriminant function for each class
        gi(c) = - 0.5*MDc -0.5*log(det(Ci{c})) + log(Ni(c)/Ntr);
    end
    
    % Fill estimated output matrix for this sample
    y_h(:,i) = gi;
    
end

elseif type == 2, % Classifier without a Priori probability

% Covariance Matrix inverse
Ci_inv = cell(1,Nc);
for i = 1:Nc,
    Ci_inv{i} = pinv(Ci{i});
end

for i = 1:N,
    % Get sample
    xi = X(i,:);
    % init discriminant function
    gi = zeros(Nc,1);
    for c = 1:Nc,
        % mahalanobis distance for each class
        MDc = (xi - mu_i(c,:))*Ci_inv{c}*(xi - mu_i(c,:))';
        % discriminant function for each class
        gi(c) = - 0.5*MDc -0.5*log(det(Ci{c}));
    end
    
    % Fill estimated output matrix for this sample
    y_h(:,i) = gi;
    
end
    
elseif type == 3, % Classifier with pooled covariance matrix
    
% Pooled covariance Matrix
Ci_pooled = zeros(p,p);
for c = 1:Nc,
    Ci_pooled = Ci_pooled + Ci{c}*Ni(c)/Ntr;
end

% Pooled Covariance Matrix inverse
Ci_inv = pinv(Ci_pooled);

for i = 1:N,
    % Get sample
    xi = X(i,:);    
    % initialize discriminant function
    gi = zeros(Nc,1);
    for c = 1:Nc,
        % mahalanobis distance for each class
        MDc = (xi - mu_i(c,:))*Ci_inv*(xi - mu_i(c,:))';
        % discriminant function for each class
        gi(c) = - 0.5*MDc;
    end
    
    % Fill estimated output matrix for this sample
    y_h(:,i) = gi;
    
end

elseif type == 4, % Maximum likelihood Classifier
    
for i = 1:N,
    % Get sample
    xi = X(i,:);        
    % Initialize discriminant function
    gi = zeros(Nc,1);
    for c = 1:Nc,
        % Mahalanobis distance
        MDc = (xi - mu_i(c,:))*(xi - mu_i(c,:))';
        % Discriminant function
        gi(c) = - 0.5*MDc;
    end
    
    % Fill estimated output matrix for this sample
    y_h(:,i) = gi;

end
    
else % invalid option
    
    disp('type a valid option: 1, 2, 3, 4');

end

% Adjust outputs between [-1 and +1] - ToDo
if ((type == 1)||(type == 2)||(type == 3)||(type == 4)),

end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END