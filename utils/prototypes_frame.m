function [frame] = prototypes_frame(Cx,DATA)

% --- Save current frame for clustering function ---
%
%   [frame] = prototypes_frame(C,DATA)
%
%   Input:
%       Cx = prototypes              	[p x Nk]
%       DATA.
%           input = input matrix        [p x N]
%   Output:
%       frame = struct containing 'cdata' and 'colormap'

%% INITIALIZATION

input = DATA.input;

%% ALGORITHM

% Plot Clusters
cla;
hold on
if(~isempty(Cx)),
    plot(Cx(1,:),Cx(2,:),'k*');
end
plot(input(1,:),input(2,:),'r.');
hold off
frame = getframe;

%% END