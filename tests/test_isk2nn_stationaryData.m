%% Machine Learning ToolBox

% isK2nn Model testing in various stationary datasets
% Author: David Nascimento Coelho
% Last Update: 2020/04/08

%% Datasets:

% # code: # samples / # attributes / # classes
% Brief Description

norm = 3;   % Normalization
lbl = 1;    % Type of labeling. 1: from sequential to [-1 and +1]
Nr = 10;    % Number of repetitions of each algorithm
hold = 2;   % Hold out method
ptrn = 0.7;	% Percentage of samples for training

%% Iris

% % 06: 150 / 04 / 03
% % Just For debug
% 
% OPT =  struct('prob',06,'prob2',2,'norm',norm,'lbl',lbl, ...
%               'Nr',Nr,'hold',hold,'ptrn',ptrn);
% test_isk2nn_pipeline_stationary_1data_1Ss_Nkernel(OPT);

%% Motor Failure

% 07: 504 / 06 / 02
% Motor Failure: Short-circuit

OPT =  struct('prob',07,'prob2',2,'norm',norm,'lbl',lbl, ...
              'Nr',Nr,'hold',hold,'ptrn',ptrn);
test_isk2nn_pipeline_stationary_1data_1Ss_Nkernel(OPT);

%% Vertebral Column

% % 10: xx / xx / xx
% % Images of Vertebral Columns in order to find deseases
% 
% OPT =  struct('prob',10,'prob2',2,'norm',norm,'lbl',lbl, ...
%               'Nr',Nr,'hold',hold,'ptrn',ptrn);
% test_isk2nn_pipeline_stationary_1data_1Ss_Nkernel(OPT);

%% Cervical Cancer

% % 19: 917 / 20 / 02
% % Image of Pap-Smear Cells used to detect Cervical Cancer
% 
% OPT =  struct('prob',19,'prob2',2,'norm',norm,'lbl',lbl, ...
%               'Nr',Nr,'hold',hold,'ptrn',ptrn);
% test_isk2nn_pipeline_stationary_1data_1Ss_Nkernel(OPT);

%% Wall-Following

% % 22: 5456 / 02 / 02
% % An avoiding Wall Robot.
% 
% OPT =  struct('prob',22,'prob2',1,'norm',norm,'lbl',lbl, ...
%               'Nr',Nr,'hold',hold,'ptrn',ptrn);
% test_isk2nn_pipeline_stationary_1data_1Ss_Nkernel(OPT);

%% END