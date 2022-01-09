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
if(~isempty(Cx))
    plot(Cx(1,:),Cx(2,:),'k*');
end
plot(input(1,:),input(2,:),'r.');
if(isfield(DATA,'Xmin'))
    axis([DATA.Xmin(1)*0.9 DATA.Xmax(1)*1.1 ...
          DATA.Xmin(2)*0.9 DATA.Xmax(2)*1.1]);
end
hold off
frame = getframe;

%% END