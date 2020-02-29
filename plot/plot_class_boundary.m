function [frame] = plot_class_boundary(DATA,PAR,class_test)

% --- Save current frame for Nonlinear Classifiers functions ---
%
%   [frame] = plot_class_boundary(DATA,PAR,class_test)
%
%   Input:
%       DATA.
%           input = input matrix                [p x N]
%           output = output matrix              [c x N]
%       PAR = parameters structure              [struct]
%       class_test = classifier test function   [handle]
%   Output:
%       frame = struct containing 'cdata' and 'colormap'

%% INITIALIZATION

% chosen attributes
p1 = 1;
p2 = 2;

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
x = zeros(p,1);             % pattern to be used in function
count = 0;                  % count number of points from hyperplane

%% ALGORITHM
    
% Define grid
x_p1 = linspace(x1_min,x1_max,grid_len);
x_p2 = linspace(x2_min,x2_max,grid_len);

% Get previous classifier output
x(p1) = x_p1(1);
x(p2) = x_p2(1);
DATAts.input = x;
OUTts = class_test(DATAts,PAR);
[~,class_prev] = max(OUTts.y_h);

% Get points of hyperplane
for i = 1:grid_len,
    x(p1) = x_p1(i);                        % set p1 attribute value
    for j = 1:grid_len,
        x(p2) = x_p2(j);                    % set p2 attribute value
        DATAts.input = x;                   % build data in struct
        OUTts = class_test(DATAts,PAR);     % get classifier output
        [~,class] = max(OUTts.y_h);         % get class
        if (class ~= class_prev)
            count = count + 1;
            hp(:,count) = [x(p1);x(p2)];    % hold point
        end
        class_prev = class;                 % update previous class
    end
end

% Clear current axis
cla;

% Plot Hyperplane points
hyperplane = hp(:,1:count);  % just get points that were saved

plot(hyperplane(1,:),hyperplane(2,:),'k.')
axis([x1_min-0.1 , x1_max+0.1 , x2_min - 0.1 , x2_max + 0.1])
hold on

% Plot Data points
plot(X(p1,:),X(p2,:),'r.')
hold off

% Get frame
frame = getframe;

%% END