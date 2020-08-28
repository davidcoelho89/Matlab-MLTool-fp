%% test script

clear;
clc;
format long e;

%% Get Number of samples, attributes, classes

% Nc = length(unique(DATA.output));	% get number of classes
% [p,N] = size(DATA.input);        	% get number of attributes and samples

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

%% Inverse Test - update one element (Sherman-Morrison 1950)

% % Original matrix and its inverse
% 
% a = [2.384 1.238 0.861 2.413; ...
%      0.648 1.113 0.761 0.137; ...
%      1.119 0.643 3.172 1.139; ...
%      0.745 2.137 1.268 0.542];
% 
% [n,~] = size(a);
% 
% b = inv(a);
% 
% % Create perturbed matrix
% 
% R = 2;
% S = 4;
% delta_ars = 0.4;
% 
% A = a;
% A(R,S) = A(R,S) + delta_ars;
% 
% % Calculate inverse of A iteratively
% 
% B1 = zeros(n,n);
% 
% for r = 1:n,
%     for j = 1:n,
%         if(r == S),
%             B1(S,j) = b(S,j) / (1 + b(S,R)*delta_ars);
%         elseif (j == R),
%             B1(r,R) = b(r,R) / (1 + b(S,R)*delta_ars);
%         else
%             B1(r,j) = b(r,j) - ...
%                      (b(r,R)*b(S,j)*delta_ars)/(1 + b(S,R)*delta_ars);
%         end
%     end
% end
% 
% % Calculate inverse of A directly
% 
% B2 = inv(A);

%% Inverse Test - Remove one line and one column (Juarez-Ruiz 2016)

% % Original matrix
% 
% A = [1  4 6; ...
%      2 -1 3; ...
%      3  2 5];
% 
% [n,~] = size(A);
% 
% % Inverse matrix
% 
% A_inv = pinv(A);
% 
% % Get line and column to be removed
% 
% p = 2;
% q = 3;
% 
% % Calculate inverse of Apq directly
% 
% Apq = A;
% Apq(p,:) = [];
% Apq(:,q) = [];
% 
% Apq_inv_batch = pinv(Apq);
% 
% % Calculate inverse of Apq iteratively
% 
% ep = zeros(n,1);
% ep(p) = 1;
% 
% u = A(:,q) - ep;
% 
% eq = zeros(n,1);
% eq(q) = 1;
% 
% v = eq;
% 
% Apq_inv_online = A_inv + (A_inv * u)*(v' * A_inv) / (1 - v' * A_inv * u);
% Apq_inv_online(q,:) = [];
% Apq_inv_online(:,p) = [];

%% Inverse Test - Add one line and one column (Vaerenbergh 2014)

% close;                              % Close all windows
% clear;                              % Clear all variables
% clc;                                % Clear command window
% format long e;                      % Output data style (float)
% 
% OPT.prob = 6;                       % Which problem will be solved
% OPT.norm = 3;                       % Normalization definition
% OPT.lbl = 1;                        % Labeling definition
% 
% DATA = data_class_loading(OPT);     % Load Data Set
% DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
% DATA = label_encode(DATA,OPT);      % adjust labels for the problem
% 
% PAR.Ktype = 2;
% PAR.sigma = 2;
% PAR.sig2n = 0.001;
% 
% c = PAR.sig2n;
% 
% X = DATA.input;
% [~,N] = size(X);
% 
% Km1 = [];
% Kinv1 = [];
% Km2 = [];
% Kinv2 = [];
% Dx = [];
% 
% for j = 1:N,
%     % Get new sample and size of dictionary
%     xt = X(:,j);
%     [~,m] = size(Dx);
%     % Calculate ktt
%     ktt = kernel_func(xt,xt,PAR);
%     % Calculate Inverse Kernel Matrix Recursively
%     if(m == 0),
%         Km2 = ktt + c;
%         Kinv2 = 1/Km2;
%     else
%         kx = zeros(m,1);
%         for i = 1:m,
%             kx(i) = kernel_func(Dx(:,i),xt,PAR);
%         end
%         Km2_aux = [Km2, kx; kx', ktt+c];
%         Km2 = Km2_aux;
%         at = Kinv2*kx;
%         gamma = ktt - kx'*at + c;
%         Kinv2 = (1/gamma)* [gamma*Kinv2 + at*at', -at; -at', 1];
%    end
%     % Update Dictionary
%     Dx_aux = [Dx,xt];
%     Dx = Dx_aux;
%     % Calculate Inverse Kernel Matrix Directly
%     Km1 = kernel_mat(Dx,PAR);
%     Kinv1 = pinv(Km1);
% end

%% Inverse Test - Update one line and one column ()

% ToDo - All
% Now: remove "old line and old column" and add "new line and new column"

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

% OPT.prob = 6;
% OPT.norm = 3;
% OPT.lbl = 1;
% 
% DATA = data_class_loading(OPT);
% DATA = normalize(DATA,OPT);
% DATA = label_encode(DATA,OPT);
% 
% X = DATA.input;
% [p,N] = size(X);
% [~,Y] = max(DATA.output);
% Nc = length(unique(Y));
% 
% m = mean(X,2);
% M = repmat(m,1,N);
% 
% Xi = cell(Nc,1);
% mi = cell(Nc,1);
% Ni = cell(Nc,1);
% Pi = cell(Nc,1);
% Si = cell(Nc,1);
% 
% Sw = zeros(p,p);
% Sb = zeros(p,p);
% St = (X - M) * (X - M)' / N;
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
% Stest = St - (Sw + Sb);
% 
% % Separability measures
% 
% Ja = det(Sw\Sb);        % For multiclass problems
% Jb = trace(Sw\Sb);      % For multiclass problems
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

%% Pairplot Test

clear; clc;

OPT.prob = 6;
DATA = data_class_loading(OPT);
label = {'SL','SW','PL','PW'};

figure; plot_data_pairplot(DATA);
figure; plot_data_pairplot(DATA,label);
figure; plot_data_pairplot(DATA,label,'histogram');

%% Results Analysis - Streaming Data - HPO: 0

% Analysis measures

Analysis = zeros(8,OPT.Nr);

Accs = zeros(1,OPT.Nr);
Nprots = zeros(1,OPT.Nr);
v1s = zeros(1,OPT.Nr);
Ktypes = zeros(1,OPT.Nr);
sigmas = zeros(1,OPT.Nr);
alphas = zeros(1,OPT.Nr);
thetas = zeros(1,OPT.Nr);
gammas = zeros(1,OPT.Nr);

% Get measures

for r = 1:OPT.Nr,
    
    acc_vect = accuracy_vector_acc{r};
    Accs(r) = acc_vect(end);
    
    param = PAR_acc{r};
    [~,Nprots(r)] = size(param.Cx);
    v1s(r) = param.v1;
    Ktypes = param.Ktype;
    sigmas(r) = param.sigma;
    alphas(r) = param.alpha;
    thetas(r) = param.theta;
    gammas(r) = param.gamma;
end

% Hold analysis

Analysis(1,:) = Accs;
Analysis(2,:) = Nprots;
Analysis(3,:) = v1s;
Analysis(4,:) = Ktypes;
Analysis(5,:) = sigmas;
Analysis(6,:) = alphas;
Analysis(7,:) = thetas;
Analysis(8,:) = gammas;

% Plot Graphics

plot(Nprots,Accs,'k.');

% Verify best accuracies

% indexes = find(Accs == max(Accs)),
% v1s(indexes),
% Nprots(indexes),
% Ktypes(indexes),
% sigmas(indexes),
% alphas(indexes),
% thetas(indexes),
% gammas(indexes),

%% Results Analysis - Streaming Data - HPO: 1

% clear;
% clc;

% Dm - 2 Hpo - 1, Norm - 3, Class - nn, Ss - 1, Us - 1, Ps - 2

% load('chess_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_lin_1nn.mat');
% load('chess_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_gau_1nn.mat');
% load('chess_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_pol_1nn.mat');
% load('chess_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_cau_1nn.mat');
% load('chess_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_sig_1nn.mat');
% load('coverType_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_lin_1nn.mat');
% load('coverType_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_gau_1nn.mat');
% load('coverType_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_pol_1nn.mat');
% load('coverType_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_cau_1nn.mat');
% load('coverType_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_sig_1nn.mat');
% load('hyper_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_lin_nn.mat');
% load('hyper_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_gau_nn.mat');
% load('hyper_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_pol_nn.mat');
% load('hyper_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_cau_nn.mat');
% load('hyper_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_sig_nn.mat');
% load('outdoor_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_lin_1nn.mat');
% load('outdoor_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_gau_1nn.mat');
% load('outdoor_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_pol_1nn.mat');
% load('outdoor_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_cau_1nn.mat');
% load('outdoor_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_sig_1nn.mat');
% load('poker_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_lin_1nn.mat');
% load('poker_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_gau_1nn.mat');
% load('poker_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_pol_1nn.mat');
% load('poker_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_cau_1nn.mat');
% load('poker_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_sig_1nn.mat');
% load('rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_lin_nn.mat');
% load('rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_gau_nn.mat');
% load('rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_pol_nn.mat');
% load('rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_cau_nn.mat');
% load('rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_sig_nn.mat');
% load('rbfMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_lin_1nn.mat');
% load('rbfMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_gau_1nn.mat');
% load('rbfMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_pol_1nn.mat');
% load('rbfMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_cau_1nn.mat');
% load('rbfMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_sig_1nn.mat');
% load('rialto_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_lin_nn.mat');
% load('rialto_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_gau_nn.mat');
% load('rialto_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_pol_nn.mat');
% load('rialto_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_cau_nn.mat');
% load('rialto_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_sig_nn.mat');
% load('sea_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_lin_nn.mat');
% load('sea_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_gau_nn.mat');
% load('sea_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_pol_nn.mat');
% load('sea_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_cau_nn.mat');
% load('sea_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_sig_nn.mat');
% load('squaresMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_lin_1nn.mat');
% load('squaresMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_gau_1nn.mat');
% load('squaresMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_pol_1nn.mat');
% load('squaresMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_cau_1nn.mat');
% load('squaresMov_isk2nn_hpo1_norm0_Dm2_Ss1_Us1_Ps2_sig_1nn.mat');
% load('weather_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_lin_nn.mat');
% load('weather_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_gau_nn.mat');
% load('weather_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_pol_nn.mat');
% load('weather_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_cau_nn.mat');
% load('weather_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_sig_nn.mat');

x = 1:Nttt;

% Data and Prototypes
figure;
hold on 
plot(DATAttt.input(1,:),DATAttt.input(2,:),'r.');
plot(PAR.Cx(1,:),PAR.Cx(2,:),'k*');
title('Data and Prototypes')
hold off

% Number of samples per class
figure;
colors = lines(Nc);
hold on
for c = 1:Nc,
    plot(x,samples_per_class(c,:),'Color',colors(c,:));
end
title('Number of Samples Per Class')
hold off

% Number of Prototypes (Total and per class)
figure;
colors = lines(Nc+1);
hold on
for c = 1:Nc+1,
    plot(x,prot_per_class(c,:),'Color',colors(c,:));
end
title('Number of Prototypes')
hold off

% Number of hits x number of errors
figure;
hold on
plot(x,no_of_errors,'r-');
plot(x,no_of_correct,'b-');
title('number of hits and errors')
hold off

% Percentage of Correct Classified
figure;
hold on
plot(x,accuracy_vector,'r-');
title('percentage of correct classified')
axis([-1 length(x) -0.1 1.1])
hold off

Ktype = PAR.Ktype;
Acc = accuracy_vector(end);
Err = 1 - Acc;
[~,Nprot] = size(PAR.Cx);
v1 = PAR.v1;
sigma = PAR.sigma;
alpha = PAR.alpha;
theta = PAR.theta;
gamma = PAR.gamma;

vetor = [Ktype,Acc,Err,Nprot,v1,sigma,alpha,theta,gamma];
% display(vetor);

% OUTttt.y_h = predict_vector;
% STATS = class_stats_1turn(DATAttt,OUTttt);

%% Results Analysis - Stationary Data - HPO: 1

[~,A] = max(DATA.output);
edges = unique(A);
counts1 = histc(A(:), edges);

[~,A] = max(DATAhpo.output);
edges = unique(A);
counts2 = histc(A(:), edges);

DATA = DATAhpo;
HPgs = HP_gs;
f_train = @isk2nn_train;
f_class = @isk2nn_classify;
PSp = GSp;
nargin = 5;

[~,test] = max(PAR.Cy);
he = sum(test == 1);

%% Weighted Knn

K = 5;
Nc = 3;
lbls_near = [[-1;-1;1], [-1;1;-1], [-1;1;-1] [1;-1;-1] [1;-1;-1]];
votes = zeros(1,Nc);
for k = 1:K,
	[~,class] = max(lbls_near(:,k));
	votes(class) = votes(class) + 1;
end

%% Recursive Calculate Average and Variance

x = [2,4,5,7,13,2.5,8,4.5];
% x = [4,6,12,9];
N = length(x);

% Calculate with function

Xmean = mean(x);
Xvar = var(x,1);
Xstd = std(x,1);

% Calculate recursively

Xmean_rec = x(1);
Xvar_rec = 0;
Xstd_rec = 0;

for t = 2:N,
    Xmean_rec = ((t-1)/t)*Xmean_rec + x(t)/t;
    Xvar_rec = ((t-1)/t)*Xvar_rec + (1/(t-1))*(x(t)-Xmean_rec)^2;
    Xstd_rec = sqrt(Xvar_rec);
end

%% Bar graph

x = categorical({'C','E'});
y = [1 3];
bar(x,y,0.5)

%% Eigenfaces and FisherFaces Test

% ToDo - All

%% Optimization Functions - ACO

% ToDo - All

%% Mixture of Gaussians (MOG)

% ToDo - All

%% Decision Trees - Info Theory Based

% ToDo - All

%% END