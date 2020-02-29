function [frame] = plot_lin_boundary(DATA,PAR)

% --- Save current frame for Linear Classifiers functions ---
%
%   [frame] = plot_lin_boundary(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix            [p x N]
%           output = output matrix          [c x N]
%       PAR.
%           W = weight's matrix             [c x p+1]
%   Output:
%       frame = struct containing 'cdata' and 'colormap'

%% INITIALIZATION

% chosen attributes
p1 = 1;
p2 = 2;

% Get the 2 attributes of input data
x = DATA.input([p1 p2],:);

% Get max and min values of attributes
x1_min = min(x(1,:));
x1_max = max(x(1,:));
x2_min = min(x(2,:));
x2_max = max(x(2,:));

% chosen neuron
neu = 1;

% Get neuron and bias
w = PAR.W(neu, [p1+1 p2+1]);
b = PAR.W(neu, 1);

%% ALGORITHM


% Axis of hyperplane
h_x1 = linspace(x1_min,x1_max,10);
h_x2 = -(b + w(1).*h_x1)./(w(2));

% Clear current axis
cla;

% Plot Hyperplane
plot(h_x1,h_x2,'k-')
axis([x1_min-0.1 , x1_max+0.1 , x2_min - 0.1 , x2_max + 0.1])
hold on

% Plot points
plot(x(1,:),x(2,:),'r.')
hold off

% Get frame
frame = getframe;

%% END