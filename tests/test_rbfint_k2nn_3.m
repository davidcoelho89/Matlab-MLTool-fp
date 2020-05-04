%% Machine Learning ToolBox

% RBF Interchanging and isk2nn classifier
% Author: David Nascimento Coelho
% Last Update: 2020/02/23

format long e;  % Output data style (float)

%% KERNEL = LINEAR, HPO-1 NORM-3 DM-2 SS-1 US-1 PS-2 NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',28,'prob2',1,'norm',3,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file', ...
              'rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_lin_nn.mat');
          
HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-10,10,21);
HP_gs.v2 = 0.9;        
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.Ps = 2;          
HP_gs.min_score = -10; 
HP_gs.max_prot = Inf;  
HP_gs.min_prot = 1;
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 1;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2;
HP_gs.gamma = 2;       
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_Streaming_5;

%% KERNEL = GAUSSIAN, HPO-1 NORM-3 DM-2 SS-1 US-1 PS-2 NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',28,'prob2',1,'norm',3,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file', ...
              'rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_gau_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-4,3,8);
HP_gs.v2 = 0.9;        
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.Ps = 2;          
HP_gs.min_score = -10; 
HP_gs.max_prot = Inf;  
HP_gs.min_prot = 1;
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 2;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2.^linspace(-10,9,20);
HP_gs.gamma = 2;       
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_Streaming_5;

%% KERNEL = POLYNOMIAL, HPO-1 NORM-3 DM-2 SS-1 US-1 PS-2 NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',28,'prob2',1,'norm',3,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file', ...
              'rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_pol_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-13,6,20);
HP_gs.v2 = 0.9;        
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.Ps = 2;
HP_gs.min_score = -10; 
HP_gs.max_prot = Inf;  
HP_gs.min_prot = 1;
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 3;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2;
HP_gs.gamma = [2,2.2,2.4,2.6,2.8,3];
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_Streaming_5;

%% KERNEL = CAUCHY, HPO-1 NORM-3 DM-2 SS-1 US-1 PS-2 NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',28,'prob2',1,'norm',3,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file', ...
              'rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_cau_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-4,3,8);
HP_gs.v2 = 0.9;        
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.Ps = 2;
HP_gs.min_score = -10; 
HP_gs.max_prot = Inf;  
HP_gs.min_prot = 1;
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 5;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2.^linspace(-10,9,20);
HP_gs.gamma = 2;
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_Streaming_5;

%% KERNEL = SIGMOID, HPO-1 NORM-3 DM-2 SS-1 US-1 PS-2 NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',28,'prob2',1,'norm',3,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file', ...
              'rbfint_isk2nn_hpo1_norm3_Dm2_Ss1_Us1_Ps2_sig_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-13,6,20);
HP_gs.v2 = 0.9;        
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.Ps = 2;          
HP_gs.min_score = -10; 
HP_gs.max_prot = Inf;  
HP_gs.min_prot = 1;
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 7;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2;
HP_gs.gamma = 2;
HP_gs.alpha = 2.^linspace(-8,2,11);       
HP_gs.theta = 0.1;       

test_Streaming_5;

%% END