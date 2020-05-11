function [frame] = plot_class_boundary(DATA,PAR,class_test,OPTION)

% --- Save current frame of Boundary for Any Classifiers' functions ---
%
%   [frame] = plot_class_boundary(DATA,PAR,class_test,OPTION)
%
%   Input:
%       DATA.
%           input = input matrix                [p x N]
%           output = output matrix              [c x N]
%       PAR = parameters structure              [struct]
%       class_test = classifier test function   [handle]
%       OPTION.
%           p1 = first attribute                [cte]
%           p2 = second attribute               [cte]
%   Output:
%       frame = struct containing 'cdata' and 'colormap'

%% INITIALIZATION

% Get chosen attributes
if(nargin == 3),
    p1 = 1;
    p2 = 2;
else
    p1 = OPTION.p1;
    p2 = OPTION.p2;
end

% choose grid length
grid_len = 100;

% Get input data
X = DATA.input;
[p,~] = size(X);

% Get max and min values of attributes
x1_min = min(X(p1,:));
x1_max = max(X(p1,:));
x2_min = min(X(p2,:));
x2_max = max(X(p2,:));

% Init auxiliary variables
hp = zeros(2,grid_len^2);	% 2d points of hyperplane
xn = zeros(p,1);           	% pattern to be used in function
count = 0;                  % count number of points from hyperplane

%% ALGORITHM

% Define grid
xaxis_values = linspace(x1_min,x1_max,grid_len);
yaxis_values = linspace(x2_min,x2_max,grid_len);

% Get points of hyperplane (horizontal)
for i = 1:grid_len,
    
    % set p1 attribute value
    xn(p1) = xaxis_values(i);
    
    % Get class of first point
    xn(p2) = yaxis_values(1);
    DATAts.input = xn;
    OUTts = class_test(DATAts,PAR);
    [~,class_prev] = max(OUTts.y_h);

    for j = 2:grid_len,
        xn(p2) = yaxis_values(j);         	% set p2 attribute value
        DATAts.input = xn;                  % build data in struct
        OUTts = class_test(DATAts,PAR);     % get classifier output
        [~,class] = max(OUTts.y_h);         % get current class
        if (class ~= class_prev)
            count = count + 1;
            hp(:,count) = xn;               % hold point
        end
        class_prev = class;                 % update previous class
    end
end

% Begin new figure

%cla;       % clear current axis
figure;
hold on

% Define figure properties

s1 = 'Attribute ';  s2 = int2str(p1);    s3 = int2str(p2);
xlabel(strcat(s1,s2));
ylabel(strcat(s1,s3));
title('Model Boundary and Data')

% Plot Hyperplane points

hyperplane = hp(:,1:count);  % just get points that were saved
plot(hyperplane(1,:),hyperplane(2,:),'k.')
axis([x1_min - 0.1 , x1_max + 0.1 , x2_min - 0.1 , x2_max + 0.1])

% Plot Data points
plot(X(p1,:),X(p2,:),'r.')
hold off

% Get frame
frame = getframe;

%% END