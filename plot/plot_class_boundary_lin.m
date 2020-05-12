function [] = plot_class_boundary_lin(DATA,PAR,OPTION)

% --- Plot Boundary for Linear Classifiers functions ---
%
%   [] = plot_class_boundary_lin(DATA,PAR,OPTION)
%
%   Input:
%       DATA.
%           input = input matrix            [p x N]
%           output = output matrix          [c x N]
%       PAR.
%           W = weight's matrix             [c x p+1]
%       OPTION.
%           p1 = first attribute         	[cte]
%           p2 = second attribute          	[cte]
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATION

% Get chosen attributes
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

% chosen prototype
prot = 1;

% Get neuron and bias
w = PAR.W(prot, [p1+1 p2+1]);
b = PAR.W(prot, 1);

%% ALGORITHM

% Begin Figure
figure;
hold on

% Define figure properties
s1 = 'Attribute ';  s2 = int2str(p1);    s3 = int2str(p2);
xlabel(strcat(s1,s2));
ylabel(strcat(s1,s3));
title('Model Boundary and Data')

% Axis of hyperplane
h_x1 = linspace(x1_min,x1_max,10);
h_x2 = -(b + w(1).*h_x1)./(w(2));

% Plot Hyperplane
plot(h_x1,h_x2,'k-')
axis([x1_min-0.1 , x1_max+0.1 , x2_min - 0.1 , x2_max + 0.1])

% Plot points
plot(x(1,:),x(2,:),'r.')

% Finish Figure
hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END