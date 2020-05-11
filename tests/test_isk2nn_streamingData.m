%% Machine Learning ToolBox

% isK2nn Model testing in various streaming datasets
% Author: David Nascimento Coelho
% Last Update: 2020/04/08

%% Datasets:

% # code: # samples / # attributes / # classes
% Brief Description

%% Iris

% 06: 150 / 04 / 03
% Just For debug

%% LED

% 32: xx / 22 / 10
% Atributes = 0 or 1. Represents a 7 segments display.
% 15 Irrelevant Attributes. Which attribute is irrelevant: changes.
% Incremental Drift.

%% Sea Concepts

test_sea_isk2nn;

% 25: 200k x 03 x 02
% label noise (10%)
% f1 + f2 = b; b is changing each 5000 samples.
% Abrupt drift

%% Rotating Hyperplane

% 200k x 10 x 02. Moving Hyperplane. Gradual Drift.

test_hyper_isk2nn;

%% RBF Moving

% 200k x 02 x 05. Moving RBFs. Different Mean. Gradual drift.

%% RBF Interchange

% 200k x 02 x 15. Interchanging RBFs. Change Means. Abrupt drift.

test_rbfint_isk2nn;

%% Moving Squares

% ---- x -- x 04. Moving Squares. Gradual/Incremental drift.

%% Transient Chessboard

% ---- x -- x 08. Virtual Reocurring drifts.

%% Mixed Drift

% ---- x -- x 15. Various drifts.

%% Weather

test_weather_isk2nn;

%% Electricity



%% Cover Type



%% Poker Hand



%% Outdoor



%% Rialto

test_rialto_isk2nn;

%% Spam



%% END