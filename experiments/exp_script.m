%% Test Script

clear;
clc;
format long e;

%% Add a path to find functions

% addpath('Func/');

%% Seed for aleatory numbers

% rng(0);

%% Pause loop

% for i = 1:10,
%     disp(i);
%     pause;
% end

%% Empty Fields in Functions

% s = input('Enter a number\n');
% if isempty(s)
%     s = 3;
% end

% output = function1(x,y,z);

% var = function1(10,20);

% if nargin == 2,	% if the number of inputs equals 2
%   z = 5;          % make the third value, z, equal to default value 5.
% end

% Omit One value

% testNargin(1,[],3)

%% Permute and Reshape Matrices

% if (length(size(C)) == 3),
%     C = permute(C,[2 3 1]);
%     C = reshape(C,[],size(C,3),1);
%     C = C';
% end

%% Cells, Matrices and Structures

% % Initializations
% [Nr,~] = size(cell_1);
% [Nc,~] = size(cell_1{1,1});
% sum = zeros(Nc);
% 
% % Algorithm
% for r = 1:Nr,
%     sum = sum + cell_1{r};
% end

% % Structs Test
% 
% labels = cell(2,1);
% labels(1) = {'ALG-1'};
% labels{2} = 'ALG-2';

%% Write to Excel or Text Software

% load patients.mat
% T = table(LastName,Age,Weight,Smoker);
% T(1:5,:)
% 
% filename = 'patientdata.xlsx';
% writetable(T,filename,'Sheet',1,'Range','D1');
% writetable(T,filename,'Sheet','MyNewSheet','WriteVariableNames',false);
% 
% A = magic(5);
% C = {'Time', 'Temp'; 12 98; 13 'x'; 14 97};
% 
% filename = 'testdata.xlsx';
% writematrix(A,filename,'Sheet',1,'Range','E1:I5');
% writecell(C,filename,'Sheet','Temperatures','Range','B2');
% 
% filename = 'testdata.xlsx';
% A = {'Time','Temperature'; 12,98; 13,99; 14,97};
% sheet = 'testsheet';
% xlRange = 'E1';
% xlswrite(filename,A,sheet,xlRange);

%% Get Number of samples, attributes, classes

% Nc = length(unique(DATA.output));	% get number of classes
% [p,N] = size(DATA.input);        	% get number of attributes and samples

%% Cross Validation - Hyperparameters

% ELMcv.Nh = 10:30;
%     
% MLPcv.Nh = 2:20;
% 
% RBFcv.Nh = 2:20;
% 
% SVMcv.lambda = [0.5 5 10 15 25 50 100 250 500 1000];
% SVMcv.sigma = [0.01 0.05 0.1 0.5 1 5 10 50 100 500];
%     
% LSSVMcv.lambda = 2.^linspace(-5,20,26);
% LSSVMcv.sigma = 2.^linspace(-10,10,21);
% 
% MLMcv.K = 2:15;
%    
% KNNcv.K = 1:10;
% 
% KMcv.Nk = 2:20;
% 
% WTAcv.Nk = 2:20;
% 
% LVQcv.Nk = 2:20;

%% Hyperparameters Optimization Functions

% With Grid Search Method and Cross Validation
% [HP] = grid_search_cv(DATAtr,HPcv,@gauss_train,@gauss_classify,CVp);

% With Optimization Method and Cross Validation
% [HP] = hp_optm_cv(DATAtr,HPcv,@gauss_train,@gauss_classify,CVp);

%% Image table - Write to Excel

% filename = 'facesdata.xlsx';
% 
% expressions = {'CenterLight';'Glasses';'Happy';'LeftLight';'NoGlasses';...
%               'Normal';'RightLight';'Sad';'Sleepy';'Surprised';'Wink'};
% 
% Nind = 15;                      % No of people (classes)
% Nexp = length(expressions);   	% No of expressions
% Nsamp = Nind * Nexp;            % No of samples
% [~,Nts] = size(DATA_acc{1}.DATAts.input);   % No of test samples
% 
% X = DATA.input;
% 
% Matrizes = cell(OPT.Nr,1);
% Nprot = zeros(OPT.Nr,1);
% 
% for r = 1:OPT.Nr,
%     % Get Prototypes, Number of Prototypes, and Classes
%     PAR = PAR_acc{r};
%     Cx = PAR.Cx;
%     Cy = PAR.Cy;
%     [~,Nprot(r)] = size(Cx);
%     [~,Prototype_index] = max(Cy);
%     % Get Test data
%     DATAts = DATA_acc{r}.DATAts;
%     Xts = DATAts.input;
%     % Get Test Statistics
%     [~,Y] = max(STATS_ts_acc{r}.Y);
%     [~,Yh] = max(STATS_ts_acc{r}.Yh);
%     % Fill Matrix
%     Mpos = -1*ones(Nexp,Nind);
%     for i = 1:Nsamp,
%         % Get Sample
%         xn = X(:,i);
%         % Get Choosen Prototypes
%         for j = 1:Nprot(r),
%             cx = Cx(:,j);
%             if (norm(cx-xn,2) == 0),
%                 expression = mod(i,Nexp);
%                 if(expression == 0),
%                     expression = Nexp;
%                 end
%                 Mpos(expression,Prototype_index(j)) = 1;
%                 break;
%             end
%         end
%         % Get errors and hits
%         for j = 1:Nts,
%             xts = Xts(:,j);
%             if(norm(xts-xn,2) == 0),
%                 expression = mod(i,Nexp);
%                 if(expression == 0),
%                     expression = Nexp;
%                 end
%                 if (Y(j) == Yh(j))
%                     Mpos(expression,Y(j)) = 2;
%                 else
%                     Mpos(expression,Y(j)) = -2;
%                 end
%                 break;
%             end
%         end
%     end
%     % Accumulate Matrix
%     Matrizes{r} = Mpos;
% end
% 
% % Save Information Sheet (Sheet 0) at File 
% 
% sheet = 'Sheet1';
% 
% A = {'OPT','Value','HP','Value','Acc Train','Acc Test','Nprot'};
% xlRange = 'A1';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = {'prob';'prob2';'norm';'lbl';'Nr';'hold';'ptrn';'file'};
% xlRange = 'A2';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = {OPT.prob;OPT.prob2;OPT.norm;OPT.lbl;OPT.Nr;OPT.hold;OPT.ptrn;OPT.file};
% xlRange = 'B2';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = {'Dm';'Ss';'Ps';'Us';'v1';'v2';'K';'Ktype';'sigma'};
% xlRange = 'C2';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = {PAR.Dm;PAR.Ss;PAR.Ps;PAR.Us;PAR.v1;PAR.v2;PAR.K;PAR.Ktype;PAR.sigma};
% xlRange = 'D2';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = nSTATS_tr.acc';
% xlRange = 'E2';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = nSTATS_ts.acc';
% xlRange = 'F2';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = Nprot;
% xlRange = 'G2';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = {'label'};
% xlRange = 'A12';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = {'-1';'1';'-2';'2'};
% xlRange = 'A13';
% xlswrite(filename,A,sheet,xlRange);
% 
% A = {'Train and Not Selected';'Train and Selected';...
%      'Test and Error';'Test and Hit'};
% xlRange = 'B13';
% xlswrite(filename,A,sheet,xlRange);
% 
% % Save Remaining Sheets (With matrices)
% 
% for r = 1:OPT.Nr,
%     
%     sheet = r+1;
% 
%     A = {'Expression / Person'};
%     xlRange = 'A1';
%     xlswrite(filename,A,sheet,xlRange);
% 
%     A = 1:15;
%     xlRange = 'B1';
%     xlswrite(filename,A,sheet,xlRange);
% 
%     A = Matrizes{r};
%     xlRange = 'B2';
%     xlswrite(filename,A,sheet,xlRange);
% 
%     A = expressions;
%     xlRange = 'A2';
%     xlswrite(filename,A,sheet,xlRange);
%     
%     A = {'Prot Selected';'Prot not selected';...
%          'Correctly Classified';'Misclassified'};
%     xlRange = 'A13';
%     xlswrite(filename,A,sheet,xlRange);
%     
%     A = {'=COUNTIF(B2:B12;1)'};
%     xlRange = 'B13';
%     xlswrite(filename,A,sheet,xlRange);
% 
%     A = {'=COUNTIF(B2:B12;-1)'};
%     xlRange = 'B14';
%     xlswrite(filename,A,sheet,xlRange);
% 
%     A = {'=COUNTIF(B2:B12;+2)'};
%     xlRange = 'B15';
%     xlswrite(filename,A,sheet,xlRange);
% 
%     A = {'=COUNTIF(B2:B12;-2)'};
%     xlRange = 'B16';
%     xlswrite(filename,A,sheet,xlRange);
% end

%% Scatter Matrices and Separability Measurements

% % Get data structure
% 
% OPT.prob = 6;
% OPT.norm = 3;
% OPT.lbl = 1;
% 
% DATA = data_class_loading(OPT);
% 
% PARnorm = normalize_fit(DATA,OPT);
% DATA = normalize_transform(DATA,PARnorm);
% 
% DATA = label_encode(DATA,OPT);
% 
% % Get input and output (Number of samples, attributes and classes)
% 
% X = DATA.input;
% [p,N] = size(X);
% [~,Y] = max(DATA.output);
% Nc = length(unique(Y));
% m = mean(X,2);
% 
% % Calculate Total Scatter Matrix
% 
% M = repmat(m,1,N);
% St = (X - M) * (X - M)' / N;    % Total covariance matrix
% 
% % Measures of individual classes
% 
% Xi = cell(Nc,1);    % Samples from each class
% mi = cell(Nc,1);    % Mean of each class
% Ni = cell(Nc,1);    % Number of samples from each class
% Pi = cell(Nc,1);    % A priori probability of each class
% Si = cell(Nc,1);    % Covariacen matrix of each class
% 
% % Initialize within and between scatter matrices
% 
% Sw = zeros(p,p);
% Sb = zeros(p,p);
% 
% % Calculate Individula Measures and Scatter Matrices
% 
% for j = 1:Nc,
%     Xi{j} = X(:,(Y == j));
%     mi{j} = mean(Xi{j},2);
%     Ni{j} = length(find(Y == j));
%     Mi = repmat(mi{j},1,Ni{j});
%     Pi{j} = Ni{j} / N;
%     Si{j} = (Xi{j} - Mi) * (Xi{j} - Mi)' / Ni{j};
%     Sw = Sw + Pi{j} * Si{j};
%     Sb = Sb + Pi{j} * (mi{j} - m) * (mi{j} - m)';
% end
% 
% % Total Scatter should be the sum of between scatter and within scatter.
% 
% Stest = St - (Sw + Sb);
% 
% % Separability measures
% 
% Ja = det(Sw\Sb);        % For multiclass problems
% Jb = trace(Sw\Sb);      % For multiclass problems
% Nk = 3;                 % Number of clusters (hyperparameter)
% Jc = trace(Sw\Sb)/ Nk;  % For number of clusters decision

%% SUBPLOT

% subplot(2,2,1), plot(Validation.FPE_index,'x-'), ...
% title('FPE'), xlabel('K'), ylabel('Critério')
% subplot(2,2,2), plot(Validation.AIC_index,'x-'), ...
% title('AIC'), xlabel('K'), ylabel('Critério')
% subplot(2,2,3), plot(Validation.BIC_index,'x-'), ... 
% title('BIC'), xlabel('K'), ylabel('Critério')
% subplot(2,2,4), plot(Validation.MDL_index,'x-'), ...
% title('MDL'), xlabel('K'), ylabel('Critério')

%% Distance Between Samples (for novelty criterion)

% X = DATA.input;
% [~,N] = size(X);
% 
% Mdist = zeros(N,N);
% 
% for j = 1:N,
%     for i = j:N,
%         Mdist(j,i) = vectors_dist(X(:,j),X(:,i),HP);
%         Mdist(i,j) = Mdist(j,i);
%     end
% end
% 
% min_value = min(min(Mdist));
% max_value = max(max(Mdist));
% mean_value = mean(mean(Mdist));

%% Plot Classes With different colors

% colors = {'r*','b*','k*'};
% figure;
% hold on
% for j = 1:3,
%     plot(Xlda(1,(Y == j)),Xlda(2,(Y == j)),colors{j});
% end
% hold off

%% Kernel Mahalanobis Distance

% close;
% clear;
% clc;
% format long e;
% 
% OPT.prob = 6;
% OPT.norm = 3;
% OPT.lbl = 1;
% OPT.Nr = 02;
% OPT.hold = 2;
% OPT.ptrn = 0.7;
% 
% DATA = data_class_loading(OPT);
% DATA = normalize(DATA,OPT);
% DATA = label_encode(DATA,OPT);
% 
% HP.sig2n = 0.001;
% sig2n = HP.sig2n;
% HP.Ktype = 2;
% HP.sigma = 2;
% 
% r = 1;
% 
% DATA_acc{r} = hold_out(DATA,OPT);
% DATAtr = DATA_acc{r}.DATAtr;
% DATAts = DATA_acc{r}.DATAts;
% 
% % TRAINING
% 
% Xtr = DATAtr.input;
% Ytr = DATAtr.output;
% [Nc,~] = size(Ytr);
% [~,Y_seq] = max(Ytr);
% 
% Xtr_c = cell(Nc,1);         % Hold Training Data points per class
% n_c = cell(Nc,1);           % Hold Number of samples per class
% H_c = cell(Nc,1);           % Hold H matrix per class
% 
% Km = cell(Nc,1);            % kernel matrix
% Kinv = cell(Nc,1);          % inverse kernel matrix
% Km_t = cell(Nc,1);          % "tilde" -> "centered"
% Kinv_t = cell(Nc,1);        % "tilde" -> "centered"
% Km_reg = cell(Nc,1);        % regularized kernel matrix
% Kinv_reg = cell(Nc,1);      % regularized inverse kernel matrix
% Km_reg_t = cell(Nc,1);      % "tilde" -> "centered"
% Kinv_reg_t = cell(Nc,1);    % "tilde" -> "centered"
% 
% for c = 1:Nc,
%     % Get samples of class
%     n_c{c} = sum(Y_seq == c);
%     Xtr_c{c} = Xtr(:,(Y_seq == c));
%     % Calculate H matrix of class
%     V1 = ones(n_c{c},1);
%     Ident = eye(n_c{c});
%     H_c{c} = (Ident - (1/n_c{c})*(V1*V1'));
%     % Calculate kernel matrix
%     Km{c} = kernel_mat(Xtr_c{c},HP);
%     Kinv{c} = pinv(Km{c});
%     % Calculate Regularized matrix
%     Km_reg{c} = kernel_mat(Xtr_c{c},HP) + (n_c{c} - 1)*sig2n*eye(n_c{c});
%     Kinv_reg{c} = pinv(Km_reg{c});
%     % Calculate Centered kernel matrix
%     Km_t{c} = H_c{c}*Km{c}*H_c{c};
%     Kinv_t{c} = pinv(Km_t{c});
%     % Calculate Regularizerd Centered kernel matrix
% 	Km_reg_t{c} = H_c{c}*Km_reg{c}*H_c{c};
%     Kinv_reg_t{c} = pinv(Km_reg_t{c});
% end
% 
% % TEST
% 
% Xts = DATAts.input;
% Yts = DATAts.output;
% [Nc,Nts] = size(Yts);
% 
% % TEST 1 and 2
% 
% % Initialize estimated output matrix
% y_h1 = zeros(Nc,Nts);
% y_h2 = zeros(Nc,Nts);
% 
% for j = 1:Nts,
%     % get sample
%     xi = Xts(:,j);
%     % Init discriminant
%     gic = zeros(Nc,1);
%     grc = zeros(Nc,1);
%     for c = 1:Nc,
%         % Get Matrices
%         H = H_c{c};
%         Km_c = Km{c};
%         Kinv_t_c = Kinv_t{c};
%         Kinv_reg_t_c = Kinv_reg_t{c};
%         % training samples from class
%         nc = n_c{c};
%         Xc = Xtr_c{c};
%         % Calculate kx
%         kx = zeros(nc,1);
%         for i = 1:nc,
%             kx(i) = kernel_func(Xc(:,i),xi,HP);
%         end
%         % Calculate kx_t (centered)
%         kx_t = H*(kx - (1/nc)*Km_c*ones(nc,1));
%         % Calculate kxx
%         kxx = kernel_func(xi,xi,HP);
%         % Calculate kxx_t
%         kxx_t = kxx - (2/nc)*ones(1,nc)*kx ...
%                     + (1/(nc^2))*ones(1,nc)*Km_c*ones(nc,1);
%         % Kernel Mahalanobis Distance 1
%         KMDic = nc*kx_t'*Kinv_t_c*Kinv_t_c*kx_t;
%         % Kernel Mahalanobis Distance 2
%         KMDrc = (1/sig2n)*(kxx_t - kx_t'*Kinv_reg_t_c*kx_t);
%         % Calculate Eigenvalues of Km_c
%         [~,L] = eig((1/nc)*Km_c);
%         L = diag(L);
%         % Discriminant Function 1
%         gic(c) = -0.5 * (  KMDic + log(prod(L)) );
%         % Discriminant Function 2
%         grc(c) = -0.5 * ( KMDrc + log(prod(L)) );
%     end
%     % fill estimated output for this sample
%     y_h1(:,j) = gic;
%     y_h2(:,j) = grc;
% end
% 
% % Confusion matrix and accuracy (1 and 2)
% 
% Mconf1 = zeros(Nc,Nc);
% Mconf2 = zeros(Nc,Nc);
% for j = 1:Nts,
%     yi = Yts(:,j);  	% get actual label
%     [~,iY] = max(yi);
%     yh1i = y_h1(:,j);	% get estimated label 1
%     [~,iY_h1] = max(yh1i);
%     yh2i = y_h2(:,j);   % get estimated label 2
%     [~,iY_h2] = max(yh2i);
%     Mconf1(iY,iY_h1) = Mconf1(iY,iY_h1) + 1;
%     Mconf2(iY,iY_h2) = Mconf2(iY,iY_h2) + 1;
% end
% 
% acc1 = trace(Mconf1)/Nts;
% acc2 = trace(Mconf2)/Nts;

%% Kernel Ridge Regression

% close;
% clear;
% clc;
% format long e;
% 
% OPT.prob = 6;
% OPT.norm = 3;
% OPT.lbl = 1;
% OPT.Nr = 02;
% OPT.hold = 2;
% OPT.ptrn = 0.7;
% 
% DATA = data_class_loading(OPT);
% DATA = normalize(DATA,OPT);
% DATA = label_encode(DATA,OPT);
% 
% PAR.sig2n = 0.001;
% PAR.Ktype = 2;
% PAR.sigma = 2;
% 
% r = 1;
% 
% DATA_acc{r} = hold_out(DATA,OPT);
% DATAtr = DATA_acc{r}.DATAtr;
% DATAts = DATA_acc{r}.DATAts;
% 
% % TRAINING
% 
% Xtr = DATAtr.input;
% Ytr = DATAtr.output;
% [Nc,Ntr] = size(Ytr);
% 
% alphas = cell(Nc,1);
% 
% Km = kernel_mat(Xtr,PAR);
% Kinv = pinv(Km);
% 
% for c = 1:Nc,
%     alphas{c} = Kinv*Ytr(c,:)';
% end
% 
% % TEST
% 
% Xts = DATAts.input;
% Yts = DATAts.output;
% [Nc,Nts] = size(Yts);
% 
% % Initialize estimated output matrix
% y_h = zeros(Nc,Nts);
% 
% for j = 1:Nts,
%     % Get sample
%     x = Xts(:,j);
%     % Calculate Kt
%     kx = zeros(Ntr,1);
%     for i = 1:Ntr,
%         xj = Xtr(:,i);
%         kx(i) = kernel_func(x,xj,PAR);
%     end
%     % Calculate Outputs
%    	for c = 1:Nc,
%         alpha = alphas{c};
%         y_h(c,j) = alpha'*kx;
%     end
% end
% 
% % Confusion matrix and accuracy
% 
% Mconf = zeros(Nc,Nc);
% for j = 1:Nts,
%     yi = Yts(:,j);  	% get actual label
%     yh1i = y_h(:,j);	% get estimated label
%     [~,iY] = max(yi);
%     [~,iY_h1] = max(yh1i);
%      Mconf(iY,iY_h1) = Mconf(iY,iY_h1) + 1;
% end
% 
% acc = trace(Mconf)/Nts;

%% MNIST Digits Database (load data and see figures)

% clear; clc;
% 
% img_tr = loadMNISTImages('train-images.idx3-ubyte');
% img_ts = loadMNISTImages('t10k-images.idx3-ubyte');
% 
% lbls_tr = loadMNISTLabels('train-labels.idx1-ubyte')';
% lbls_ts = loadMNISTLabels('t10k-labels.idx1-ubyte')';
% 
% x1 = img_ts(:,3);
% x2 = reshape(x1,[28,28]);
% imshow(x2);
% 
% load('mnist.mat')
% 
% mnist_data = [img_tr, img_ts];
% mnist_lbl = [lbls_tr, lbls_ts];
% 
% clear img_tr img_ts lbls_tr lbls_ts
% 
% save('mnist.mat');

%% Pairplot and Boxplot Tests

% % Clean variables, workspace and figures
% 
% clear; clc; close all;
% 
% % Pairplot of iris dataset (3 ways of doing)
% 
% OPT.prob = 6;
% DATA = data_class_loading(OPT);
% label = {'SL','SW','PL','PW'};
% 
% plot_data_pairplot(DATA);
% plot_data_pairplot(DATA,label);
% plot_data_pairplot(DATA,label,'histogram');
% 
% % Boxplot of Comparison of results
% 
% x1 = [85.3 92.4 76.7 93.2 88.1]';
% x2 = [88.3 92.4 92.7 93.2 88.1]';
% x3 = [58.3 52.4 52.7 43.2 48.1]';
% 
% X = [x1,x2,x3];
% classifier_names = {'Classifier 1','Classifier 2','Classifier 3'};
% figure; boxplot(X,'labels',classifier_names)
% ylabel('Accuracy')
% title('Classifers Results')

%% Weighted Knn

% K = 5;
% Nc = 3;
% lbls_near = [[-1;-1;1], [-1;1;-1], [-1;1;-1] [1;-1;-1] [1;-1;-1]];
% votes = zeros(1,Nc);
% for k = 1:K,
% 	[~,class] = max(lbls_near(:,k));
% 	votes(class) = votes(class) + 1;
% end

%% Recursive Calculate Average and Variance

% x = [2,4,5,7,13,2.5,8,4.5];
% % x = [4,6,12,9];
% N = length(x);
% 
% % Calculate with function
% 
% Xmean = mean(x);
% Xvar = var(x,1);
% Xstd = std(x,1);
% 
% % Calculate recursively
% 
% Xmean_rec = x(1);
% Xvar_rec = 0;
% Xstd_rec = 0;
% 
% for t = 2:N,
%     Xmean_rec = ((t-1)/t)*Xmean_rec + x(t)/t;
%     Xvar_rec = ((t-1)/t)*Xvar_rec + (1/(t-1))*(x(t)-Xmean_rec)^2;
%     Xstd_rec = sqrt(Xvar_rec);
% end

%% Bar graph

% x = categorical({'C','E'});
% y = [1 3];
% bar(x,y,0.5)

%% Eigenfaces and FisherFaces Test

% ToDo - All

%% Optimization Functions - ACO

% ToDo - All

%% Mixture of Gaussians (MOG)

% ToDo - All

%% Decision Trees - Info Theory Based

% ToDo - All

%% Naive Bayes Classifier

% model = fitcnb(X,Y)

%% Kc House Data

% lambda = 0.01;
% 
% y = kchousedata(:,1);
% 
% X = kchousedata(:,2:end);
% [N,p] = size(X);
% X = [ones(N,1),X];
% 
% beta = (X'*X + lambda*eye(p+1))\X'*y;
% 
% yh = X*beta;
% 
% MSE = sqrt((1/N) * (y-yh)'*(y-yh));

%% KNN e MDC

% Minimum Distance to Centroid (MDC) = Nearest Prototype Classifier (NPC)

% xlswrite('classe1.xlsx',Classe1)
% xlswrite('classe2.xlsx',Classe2)

%% Pi with Monte Carlo Simulation!

% % Number of samples
% N = 100;
% 
% x = -1 + 2.*rand(2,N);
% 
% % Number of points inside circle
% np = 0;
% for i = 1:N,
%     if (x(1,i)^2+x(2,i)^2 < 1)
%         np = np +1;
%     end
% end
% 
% % Estimated pi
% pi_h = 4*np/N;
% error = pi - pi_h;

%% Random Walk

% % Number of steps
% N = 20;
% 
% % Positions
% x = zeros(N+1,1);
% y = zeros(N+1,1);
% 
% % Walk
% for i = 2:N+1,
%     step = rand(1,1);
%     if (step <= 0.25),      % Left
%         x(i) = x(i-1) - 1;
%         y(i) = y(i-1);
%     elseif (step <= 0.5),   % Right
%         x(i) = x(i-1) + 1;
%         y(i) = y(i-1);
%     elseif (step <= 0.75),  % Up
%         x(i) = x(i-1);
%         y(i) = y(i-1) + 1;
%     else                    % Down
%         x(i) = x(i-1);
%         y(i) = y(i-1) - 1;
%     end
% end
% 
% % See walk
% figure,
% plot(x,y,'b-')
% axis([-N/2 N/2 -N/2 N/2])

%% END