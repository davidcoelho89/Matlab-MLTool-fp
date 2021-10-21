%% Machine Learning ToolBox

% isk2nn with various streaming datasets
% Author: David Nascimento Coelho
% Last Update: 2021/08/24

%% Labeling and Normalization Choice

norm = 0;   % Normalization. 0: Don't normalize. 3: z-score normalization.
lbl = 1;    % Type of labeling. 1: from sequential to [-1 and +1]

%% Datasets Choice

% Vector with chosen datasets

datasets = 30; % just Chessboard
% datasets = 38; % just Rialto Dataset
% datasets = [28,29,30,33,34,37];
% datasets = [28,29,30,33,34,37,38];

%% Datasets List

% # code: # samples / # attributes / # classes
% Brief Description

% Sea Concepts              => 25: 200k / 03 / 02
% label noise (10%)
% f1 + f2 = b; b is changing each 5000 samples.
% Abrupt drift

% Rotating Hyperplane       => 26: 200k / 10 / 02. 
% Moving Hyperplane. 
% Gradual Drift.

% RBF Moving                => 27: 200k / 10 / 05. 
% Moving RBFs. Different Mean. 
% Gradual drift.

% RBF Interchange           => 28: 200k / 02 / 15. 
% Interchanging RBFs. Change Means. Abrupt drift.

% Moving Squares            => 29: 200k / 02 / 04. 
% Moving Squares. Gradual/Incremental drift.

% Transient Chessboard      => 30: 200k / 02 / 08. 
% Virtual Reocurring drifts.

% Mixed Drift               => 31: 600k / 02 / 15. 
% Various drifts.

% LED                       => 32: 200k / 24 / 10
% Atributes = 0 or 1. Represents a 7 segments display.
% 17 Irrelevant Attributes. Which attribute is irrelevant: changes.
% Incremental Drift.

% Weather                   => 33: 18159 / 08 / 02
% Virtual Drift

% Electricity               => 34: 45312 / 08 / 02
% Real Drift

% Cover Type                => 35: 581012 / 54 / 07
% Real Drift

% Poker Hand                => 36: 829201 / 10 / 10
% Virtual Drift

% Outdoor                   => 37: 4000 / 21 / 40
% Virtual Drift

% Rialto                    => 38: 82250 / 27 / 10
% Virtual Drift

% Spam                      => 39: 
% Real Drift

%% Run algorithm at datasets

if any(datasets == 25)
    OPT = struct('prob',25,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 26)
    OPT = struct('prob',26,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 27)
    OPT = struct('prob',27,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 28)
    OPT = struct('prob',28,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 29)
    OPT = struct('prob',29,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);    
end

if any(datasets == 30)
    OPT = struct('prob',30,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 31)
    OPT = struct('prob',31,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 32)
    OPT = struct('prob',32,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 33)
    OPT = struct('prob',33,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 34)
    OPT = struct('prob',34,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 36)
    OPT = struct('prob',36,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 37)
    OPT = struct('prob',37,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 38)
    OPT = struct('prob',38,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 35)
    OPT = struct('prob',35,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

if any(datasets == 39)
    OPT = struct('prob',39,'prob2',1,'norm',norm,'lbl',lbl);
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT);
end

%% FINISHED!

% load handel
% sound(y,Fs)

%% END