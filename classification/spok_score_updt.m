function [PAR] = spok_score_updt(DATAn,OUTn,HP)

% --- Update Score of each prototype from Dictionary ---
%
%   [PAR] = spok_score_updt(DATAn,OUTn,HP)
%
%   Input:
%       DATAn.
%           input = attributes of sample                     	[p x 1]
%           output = class of sample                            [Nc x 1]
%       HP.
%           Cx = Attributes of input dictionary                 [p x Nk]
%           Cy = Classes of input dictionary                    [Nc x Nk]
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method 1
%               = 2 -> score-based method 2
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]
%       OUT.
%           y_h = classifier's output                           [Nc x 1]
%           win = closest prototype to each sample              [1 x 1]
%           dist = distance from sample to each prototype       [Nk x 1]
%           near_ind = indexes for nearest prototypes           [K x N]
%   Output: 
%       PAR.
%           Cx = Attributes of output dictionary                [p x Nk]
%           Cy = Classes of  output dictionary                  [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]

%% INITIALIZATIONS

% Get Data output
yt = DATAn.output;

% Get Hyperparameters
Ps = HP.Ps;                         % Pruning Strategy
score = HP.score;                   % Score of each prototype
class_history = HP.class_history;	% Verify if last time that was chosen,
                                    % it classified correctly.

% Get Parameters
Cy = HP.Cy;                         % Classes of dictionary

% Get predicted output
yh = OUTn.y_h;
win = OUTn.win;
near_ind = OUTn.near_ind;
[K,~] = size(near_ind);

% Get problem parameters
[~,Nk] = size(Cy);                  % hold dictionary size

% Init Outputs
score_out = score;
class_hist_out = class_history;

%% ALGORITHM

if(Ps == 0)
    
    % Does nothing
    
else
    
    % Get current data class, predicted class and prototypes classes
    [~,yt_class] = max(yt);
    [~,yh_class] = max(yh);
    [~,Dy_class] = max(Cy);
    
    % number of elements, in the dictionary, of the same class as yt
    mc = sum(Dy_class == yt_class); 
    
    % if there are no prototypes from yt class
    if (mc == 0)

        % Does nothing
        
    % Update all scores
    elseif (Ps == 1)
        
        for k = 1:Nk
            % if it was a hit
            if (yt_class == yh_class)
                if (k == win)
                    score_out(k) = score(k) + 1;
                elseif (Dy_class(k) == yh_class)
                    score_out(k) = score(k) - 0.1;
                else
                    score_out(k) = score(k);
                end
            % if it was an error
            else
                if (k == win)
                    score_out(k) = score(k) - 1;
                else
                    score_out(k) = score(k);
                end
            end
        end
        
    % Update score of winner
    elseif (Ps == 2)
        
        if (K == 1) % nn strategy

            if(Dy_class(win) == yt_class)
                % Update score of winner
                if((score(win) < 0) && (class_history(win) == 1))
                    score_out(win) = score(win) + 1;
                end
                % Update class_history
                class_hist_out(win) = 1;
            else
                % Update score of winner
                if (class_history(win) == -1)
                    score_out(win) = score(win) - 1;
                end
                % Update class_history
                class_hist_out(win) = -1;
            end
            
        else % knn strategy
            
            for k = 1:K
                
                % get class of prototype
                c = Dy_class(near_ind(k));
                
                % if it was a hit
                if (yt_class == yh_class)
                    % prototype has the same class as sample?
                    if (c == yt_class)
                        % Update score
                        if((score(k) < 0) && (class_history(k) == 1))
                            score_out(k) = score(k) + 1;
                        end
                        % Update class_history
                        class_hist_out(k) = 1;
                        % Stop search
                        break;
                    else
                        continue;
                    end
                    
                % if it was an error
                else
                    % prototype and sample are from different classes?
                    if (c ~= yt_class)
                        % Update score
                        if (class_history(k) == -1)
                            score_out(k) = score(k) - 1;
                        end
                        % Update class_history
                        class_hist_out(k) = -1;
                        % Stop search
                        break;
                    else
                        continue;
                    end
                end
                
            end % end for k = 1:K

        end % end if (K == 1)
        
    end
    
end

%% FILL OUTPUT STRUCTURE

PAR = HP;                        	% Get all the parameters
PAR.score = score_out;            	% Updated score
PAR.class_history = class_hist_out;	% Update classification history

%% END