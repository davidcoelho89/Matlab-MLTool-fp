%% Machine Learning ToolBox

% isK2nn Model testing in various streaming datasets
% Author: David Nascimento Coelho
% Last Update: 2020/04/08

%% Datasets:

% # code: # samples / # attributes / # classes
% Brief Description

norm = 0;   % Normalization
lbl = 1;    % Type of labeling. 1: from sequential to [-1 and +1]

%% Sea Concepts

% % 25: 200k / 03 / 02
% % label noise (10%)
% % f1 + f2 = b; b is changing each 5000 samples.
% % Abrupt drift
% 
% OPT =  struct('prob',25,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Rotating Hyperplane

% 26: 200k / 10 / 02. 
% Moving Hyperplane. 
% Gradual Drift.

OPT =  struct('prob',26,'prob2',1,'norm',norm,'lbl',lbl);
test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% RBF Moving

% 27: 200k / 10 / 05. 
% Moving RBFs. Different Mean. 
% Gradual drift.

OPT =  struct('prob',27,'prob2',1,'norm',norm,'lbl',lbl);
test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% RBF Interchange

% 28: 200k / 02 / 15. 
% Interchanging RBFs. Change Means. Abrupt drift.

OPT =  struct('prob',28,'prob2',1,'norm',norm,'lbl',lbl);
test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Moving Squares

% % 29: 200k / 02 / 04. 
% % Moving Squares. Gradual/Incremental drift.
% 
% OPT =  struct('prob',29,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Transient Chessboard

% % 30: 200k / 02 / 08. 
% % Virtual Reocurring drifts.
% 
% OPT =  struct('prob',30,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Mixed Drift

% % 31: 600k / 02 / 15. 
% % Various drifts.
% 
% OPT =  struct('prob',31,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% LED

% % 32: 200k / 24 / 10
% % Atributes = 0 or 1. Represents a 7 segments display.
% % 17 Irrelevant Attributes. Which attribute is irrelevant: changes.
% % Incremental Drift.
% 
% OPT =  struct('prob',32,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Weather

% % 33: 18159 / 08 / 02
% % Virtual Drift
% 
% OPT =  struct('prob',33,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Electricity

% % 34: 45312 / 08 / 02
% % Real Drift
% 
% OPT =  struct('prob',34,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Cover Type

% % 35: 581012 / 54 / 07
% % Real Drift
% 
% OPT =  struct('prob',35,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Poker Hand

% % 36: 829201 / 10 / 10
% % Virtual Drift
% 
% OPT =  struct('prob',36,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Outdoor

% % 37: 4000 / 21 / 40
% % Virtual Drift
% 
% OPT =  struct('prob',37,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Rialto

% % 38: 82250 / 27 / 10
% % Virtual Drift
% 
% OPT =  struct('prob',38,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% Spam

% % 39: 
% % Real Drift
% 
% OPT =  struct('prob',39,'prob2',1,'norm',norm,'lbl',lbl);
% test_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);

%% END