%% Machine Learning ToolBox

% Images Pre-processing
% Author: David Nascimento Coelho
% Last Update: 2019/07/15

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% YALE DATA BASE A - GENERATE X AND D

expression = {'.centerlight' '.glasses' '.happy' '.leftlight' '.noglasses' ...
              '.normal' '.rightlight' '.sad' '.sleepy' '.surprised' '.wink'};

Nind = 15;                  % No of people (classes)
Nexp = length(expression);	% No of expressions
Nsamp = Nind * Nexp;        % No of samples
im_size = 30;               % New size of images

X = zeros(im_size^2,Nsamp);	% Accumulator of vectorized images
D = -1*ones(Nind, Nsamp);	% Accumulator of labels
sample = 0;                 % Update sample 

for i = 1:Nind,
    
    display(i);
    
    for j = 1:Nexp,
        
        % Update sample
        sample = sample + 1;
        
        % Update label matrix
        D(i,sample) = 1;
        
        % Build files' name
        if i < 10,
            name = strcat('subject0',int2str(i),expression{j});   
        else
            name = strcat('subject',int2str(i),expression{j});
        end
        % Read image
        img = imread(name);     
        % Show image
        imshow(img);
        pause;
        % Resize image
        Ar = imresize(img,[im_size im_size]);
        % Add noise to image
        An = imnoise(Ar,'salt & pepper',0.05);
        % Save resized image
        name_resized = strcat(name,'_resized');
        imwrite(An,name_resized,'JPG');
        % Converte image to double precision
        A = im2double(An);
        % Convert matrix into column vector
        A = A(:);
        
        % Update Matrix of Vectorized Images
        X(:,sample) = A;
        
    end 
end

%% LENA IMAGE - PCA, COMPRESS, RECONSTRUCT

% Define block
block = [8 8];

% Read Image
A = imread('data_img_lena_pb.jpg','JPEG'); 

% Image initial size
img_size = size(A);

% Verify image
figure;
imshow(A);

% Add noise
% An = A; 
% An = imnoise(A,'gaussian');
An = imnoise(A,'salt & pepper',0.05);

% Verify image
figure;
imshow(An);

% Convert for double precision
B = im2double(An);

% Transform image into a vector of submatrix
C = im2col(B,block,'distinct');

% PCA from covariance matrix
Cx = cov(C');
[V,L] = pcacov(Cx);  

% Calculate Explained variance
p = length(L);
Ev = zeros(1,p);
Ev(1) = L(1);
for i = 2:p,
    Ev(i) = Ev(i-1) + L(i);
end
Ev = Ev/sum(L);

% Plot Explained Variance
figure; 
plot(Ev); 
grid

% Find number of Principal Components
tol = 0.95;                 % tolerance
q = length(find(Ev<=tol));  % q greater eigenvalues

% Transform Matrix
M = V(:,1:q);

% Transformed Img-Matrix
D = M'*C;

% Reconstructed Img-Matrix
E = M*D;

% Transform from matrix to img
Ar = col2im(E,block,img_size,'distinct'); 

% Show reconstructed Image
figure; 
imshow(Ar);

%% END