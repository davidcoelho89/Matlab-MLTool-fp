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
%           Cx = Attributes of input dictionary                 [p x Q]
%           Cy = Classes of input dictionary                    [Nc x Q]
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method 1
%               = 2 -> score-based method 2
%           score = used for prunning method                    [1 x Q]
%           class_history = used for prunning method           	[1 x Q]
%           times_selected = used for prunning method           [1 x Q]
%       OUT.
%           y_h = classifier's output                           [Nc x 1]
%           win = closest prototype to each sample              [1 x 1]
%           dist = distance from sample to each prototype       [Q x 1]
%           near_ind = indexes for nearest prototypes           [K x N]
%   Output: 
%       PAR.
%           Cx = Attributes of output dictionary                [p x Q]
%           Cy = Classes of  output dictionary                  [Nc x Q]
%           Km = Kernel matrix of dictionary                    [Q x Q]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Q x Q]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Q]
%           class_history = used for prunning method           	[1 x Q]
%           times_selected = used for prunning method           [1 x Q]

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
winnner = OUTn.win;
nearIndex = OUTn.near_ind;
[K,~] = size(nearIndex);

% Get problem parameters
[~,Q] = size(Cy);                   % hold dictionary size

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
    Qc = sum(Dy_class == yt_class); 
    
    % if there are no prototypes from yt class
    if (Qc == 0)

        % Does nothing
        
    % Update all scores
    elseif (Ps == 1)
        
        for q = 1:Q
            % if it was a hit
            if (yt_class == yh_class)
                if (q == winnner)
                    score_out(q) = score(q) + 1;
                elseif (Dy_class(q) == yh_class)
                    score_out(q) = score(q) - 0.1;
                else
                    score_out(q) = score(q);
                end
            % if it was an error
            else
                if (q == winnner)
                    score_out(q) = score(q) - 1;
                else
                    score_out(q) = score(q);
                end
            end
        end
        
    % Update score of winner
    elseif (Ps == 2)
        
        if (K == 1) % nn strategy

            if(Dy_class(winnner) == yt_class)
                % Update score of winner
                if((score(winnner) < 0) && (class_history(winnner) == 1))
                    score_out(winnner) = score(winnner) + 1;
                end
                % Update class_history
                class_hist_out(winnner) = 1;
            else
                % Update score of winner
                if (class_history(winnner) == -1)
                    score_out(winnner) = score(winnner) - 1;
                end
                % Update class_history
                class_hist_out(winnner) = -1;
            end
            
        else % knn strategy
            
            for k = 1:K
                
                % get index
                index = nearIndex(k);
                
                % get class of prototype
                c = Dy_class(index);
                
                % if it was a hit
                if (yt_class == yh_class)
                    % prototype has the same class as sample?
                    if (c == yt_class)
                        % Update score
                        if((score(index) < 0) && (class_history(index) == 1))
                            score_out(index) = score(index) + 1;
                        end
                        % Update class_history
                        class_hist_out(index) = 1;
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
                        if (class_history(index) == -1)
                            score_out(index) = score(index) - 1;
                        end
                        % Update class_history
                        class_hist_out(index) = -1;
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