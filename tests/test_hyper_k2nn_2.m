%% Machine Learning ToolBox

% Rotating Hyperplane and k2nn classifier
% Author: David Nascimento Coelho
% Last Update: 2020/02/23

format long e;  % Output data style (float)

%% LINEAR S/ NORM NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',26,'prob2',1,'norm',0,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file','hyper_k2nn_lin2_snorm_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-10,10,21);
HP_gs.v2 = 0.9;        
HP_gs.Ps = 1;          
HP_gs.min_score = -10; 
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.max_prot = Inf;  
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 1;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2;
HP_gs.gamma = 2;       
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_Streaming_3;

%% GAUSSIAN S/ NORM NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',26,'prob2',1,'norm',0,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file','hyper_k2nn_g2_snorm_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-4,3,8);
HP_gs.v2 = 0.9;        
HP_gs.Ps = 1;          
HP_gs.min_score = -10; 
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.max_prot = Inf;  
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 2;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2.^linspace(-10,9,20);
HP_gs.gamma = 2;       
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_Streaming_3;

%% POLY S/ NORM NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',26,'prob2',1,'norm',0,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file','hyper_k2nn_p2_snorm_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-13,6,20);
HP_gs.v2 = 0.9;        
HP_gs.Ps = 1;          
HP_gs.min_score = -10; 
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.max_prot = Inf;  
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 3;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2;
HP_gs.gamma = [2,3];
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_Streaming_3;

%% CAUCHY S/ NORM NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',26,'prob2',1,'norm',0,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file','hyper_k2nn_c2_snorm_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-4,3,8);
HP_gs.v2 = 0.9;        
HP_gs.Ps = 1;          
HP_gs.min_score = -10; 
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.max_prot = Inf;  
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 5;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2.^linspace(-10,9,20);
HP_gs.gamma = 2;
HP_gs.alpha = 1;       
HP_gs.theta = 1;       

test_Streaming_3;

%% SIGMOID S/ NORM NN

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

OPT =  struct('prob',26,'prob2',1,'norm',0,'lbl',1,'Nr',1, ...
              'hold',2,'ptrn',0.7,'file','hyper_k2nn_s2_snorm_nn.mat');

HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 2.^linspace(-13,6,20);
HP_gs.v2 = 0.9;        
HP_gs.Ps = 1;          
HP_gs.min_score = -10; 
HP_gs.Us = 1;          
HP_gs.eta = 0.01;      
HP_gs.max_prot = Inf;  
HP_gs.Von = 0;         
HP_gs.K = 1;           
HP_gs.Ktype = 7;       
HP_gs.sig2n = 0.001;   
HP_gs.sigma = 2;
HP_gs.gamma = 2;
HP_gs.alpha = 2.^linspace(-8,2,11);       
HP_gs.theta = 0.1;       

test_Streaming_3;

%% END