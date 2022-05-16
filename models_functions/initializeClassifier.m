function [classifier] = initializeClassifier(classifier_name)

% --- Initialize a Classifier ---
%
%   classifier = initializeClassifier(classifier_name)
%
%   Input:
%       classifier_name = which classifier will be used
%           'lms'

%% ALGORITHM

classifierString = strcat(classifier_name,'Classifier');

classifier = feval(str2func(classifierString));

%% END