function [PARout] = k2nn_score_updt(DATAn,PAR,OUTn)

% --- Update Score of each prototype from Dictionary ---
%
%   [PARout] = k2nn_score_updt(DATA,PAR,OUT)
%
%   Input:
%       DATAn.
%           xt = attributes of sample                        	[p x 1]
%           yt = class of sample                             	[Nc x 1]
%       PAR.
%           Cx = Attributes of input dictionary                 [p x Nk]
%           Cy = Classes of input dictionary                    [Nc x Nk]
%           score = used for prunning method                    [1 x Nk]
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method
%       OUT.
%           y_h = classifier's output                           [Nc x 1]
%           win = closest prototype to each sample              [1 x 1]
%   Output: 
%       PARout.
%           Cx = Attributes of output dictionary                [p x Nk]
%           Cy = Classes of output dictionary                   [Nc x Nk]
%           score = updated for prunning method                 [1 x Nk]

%% INITIALIZATIONS

% Get Data output
yt = DATAn.output;

% Get Hyperparameters
Dy = PAR.Cy;        % Classes of dictionary
score = PAR.score;  % Score of each prototype from dictionary
Ps = PAR.Ps;        % Pruning Strategy

% Get predicted output
yh = OUTn.y_h;
win = OUTn.win;

% Get problem parameters
[~,Nk] = size(Dy);   % hold dictionary size

% Init Outputs
score_out = zeros(1,Nk);

%% ALGORITHM

if(Ps == 1),
    
    % Get current data class, predicted class and prototypes classes
    [~,yt_class] = max(yt);
    [~,yh_class] = max(yh);
    [~,Dy_class] = max(Dy);
    
    % number of elements, in the dictionary, of the same class as yh
    mc = sum(Dy_class == yt_class); 
    
    % Do not update dictinary score
    if (mc == 0),
        score_out = score;

    % Update score
    else
        for k = 1:Nk,
            % if it was a hit
            if (yt_class == yh_class),
                if (k == win),
                    score_out(k) = score(k) + 1;
                elseif (Dy_class(k) == yh_class)
                    score_out(k) = score(k) - 0.1;
                else
                    score_out(k) = score(k);
                end
            % if it was an error
            else
                if (k == win),
                    score_out(k) = score(k) - 1;
                else
                    score_out(k) = score(k);
                end
            end
        end
    end
    
end

%% FILL OUTPUT STRUCTURE

PARout = PAR;               % get all the parameters
PARout.score = score_out;   % updated score

%% END