function [frame] = get_frame_hyperplane_lin(DATA,W,OPTION)

% --- Get current frame for Linear Classifiers functions ---
%
%   [frame] = get_frame_hyperplane_lin(DATA,W,OPTION)
%
%   Input:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [c x N]
%       W = weight's matrix [c x p+1]
%       OPTION.
%           p1 = first attribute         	[cte]
%           p2 = second attribute          	[cte]
%   Output:
%       frame = struct containing 'cdata' and 'colormap'

%% INITIALIZATION

% chosen attributes
if(nargin == 2),
    p1 = 1;
    p2 = 2;
else
    p1 = OPTION.p1;
    p2 = OPTION.p2;
end

% Get the 2 attributes of input data
x = DATA.input([p1 p2],:);

% Get max and min values of attributes
x1_min = min(x(1,:));
x1_max = max(x(1,:));
x2_min = min(x(2,:));
x2_max = max(x(2,:));

% chosen neuron
prot = 1;

% Get neuron and bias
w = W(prot, [p1+1 p2+1]);
b = W(prot, 1);

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