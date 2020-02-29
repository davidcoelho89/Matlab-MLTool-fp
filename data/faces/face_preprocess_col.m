function [DATA] = face_preprocess_col(OPTION)

% --- Create Data Matrix from YALE Faces DataSet ---
%
%   DATA = face_preprocess_col(OPTION)
% 
%   Input:
%       OPTION.
%           prob2 = new size of image       [cte]
%   Output:
%       DATA.
%           input = input matrix            [p x N]
%           output = output matrix          [1 x N] (sequential: 1, 2...)
%           rot = mantain original labels   [1 x N]

%% SET DEFAULT HYPERPARAMETERS

if (~(isfield(OPTION,'prob2'))),
    OPTION.prob2 = 30;
end

%% INITIALIZATIONS

expression = {'.centerlight' '.glasses' '.happy' '.leftlight' '.noglasses' ...
              '.normal' '.rightlight' '.sad' '.sleepy' '.surprised' '.wink'};

Nind = 15;                      % No of people (classes)
Nexp = length(expression);      % No of expressions
Nsamp = Nind * Nexp;            % No of samples
img_size = OPTION.prob2;        % New size of images

X = zeros(img_size^2,Nsamp);	% Accumulator of vectorized images
d = -1*ones(Nind, Nsamp);       % Accumulator of labels
sample = 0;                     % Update sample 

%% ALGORITHM

for i = 1:Nind,
    
    for j = 1:Nexp,
        
        % Update sample
        sample = sample + 1;
        
        % Update label matrix
        d(i,sample) = 1;
        
        % Build files' name
        if i < 10,
            name = strcat('subject0',int2str(i),expression{j});   
        else
            name = strcat('subject',int2str(i),expression{j});
        end
        
        % Read Image
        img = imread(name);
        % Resize image
        Ar = imresize(img,[img_size img_size]);
        % Converte image to double precision
        A = im2double(Ar);
        % Convert matrix into column vector
        A = A(:);
        
        % Update Matrix of Vectorized Images
        X(:,sample) = A;
        
    end 
end

[~,lbl] = max(d);

%% FILL OUTPUT STRUCTURE

DATA.input = X;
DATA.output = lbl;
DATA.lbl = 1:165;

%% END