classdef spokClassifier < prototypeBasedClassifier
    %
    % --- SParse Online adptive Kernel Classifier ---
    %
    % Properties (Hyperparameters):
    %
    %    - number_of_epochs = <integer>
    %        if > 1, "shows the data" more than once to the algorithm
    %    - is_stationary = [0 or 1]
    %         Allow to shuffle data and run more than one epoch
    %    - design_method
    %        Define how the dictionary will be built
    %        = 'single_dictionary'
    %        = 'one_dicitionary_per_class'
    %    - sparsification_strategy
    %        Define how the prototypes will be selected
    %        = 'ald'
    %        = 'coherence'
    %        = 'novelty'
    %        = 'surprise'
    %    - v1 = sparsification coefficient 1 <real>
    %    - v2 = sparsification coefficient 2 <real> 
    %    - update_strategy
    %        Define how the prototypes will be updated
    %        = 'none'
    %        = 'wta' (lms, unsupervised)
    %        = 'lvq' (supervised)
    %    - update_rate = <real>
    %        Values ranging from 0 to 1
    %    - pruning_strategy = define how the prototypes will be prunned
    %        = 'none' (do not remove prototypes)
    %        = 'drift_based'
    %        = 'error_score_based'
    %    - min_score = minimum score allowed for a prototype
    %        Used for the prunning methods
    %    - max_prototypes = max # of model's prototypes ("Budget")
    %    - min_prototypes = min # of model's prototypes ("restriction")
    %    - video_enabled = [0 or 1]
    %    - distance_measure = [cte]
    %    	= 0: Dot product
    %       = inf: Chebyshev distance
    %       = -inf: Minimum Minkowski distance
    %       = 1: Manhattam (city-block) distance
    %       = 2: Euclidean distance
    %       >= 3: Minkowski distance
    %    - nearest_neighbors = number of nearest neighbors
    %    - knn_aproximation = how the output will be generated
    %        = 'majority_voting'
    %        = 'weighted_knn'
    %    - kernel_type = which kernel will be used
    %        = 'none' (it is not a kernel based model)
    %    - regularization = kernel regularization parameter
    %        Avoid numerical problems when inverting kernel matrices
    %    - sigma = kernel hyperparameter ( see kernel_func() ) 
    %    - alpha = kernel hyperparameter ( see kernel_func() )
    %    - theta = kernel hyperparameter ( see kernel_func() )
    %    - gamma = kernel hyperparameter ( see kernel_func() )
    %
    % Properties (Parameters)
    %
    %    - Cx = Clusters' centroids (prototypes) [p x Nk]
    %    - Cy = Clusters' labels
    %    - Km = Kernel Matrix of Entire Dictionary
    %    - Kmc = Kernel Matrix for each class (cell)
    %    - Kinv = Inverse Kernel Matrix of Entire Dictionary
    %    - Kinvc = Inverse Kernel Matrix for each class (cell)
    %    - score = Score used for prunning method
    %    - classification_history = Used for prunning method
    %    - times_selected = Used for prunning method
    %    - video = frame structure (can be played with 'video function')
    %              Size: [1 x Nep.Ntr]
    %    - Yh = Hold all predictions
    %    - yh = Hold last prediction
    %
    % Methods:
    %
    %	- spokClassifier()        % Constructor
    %	- partial_fit(self,x,y)   % Training Function (1 instance)
    %	- fit(self,X,Y)           % Training Function (N instances)
    %	- predict(self,X)         % Prediction Function

    % Hyperparameters
    properties
        number_of_epochs = 1;
        is_stationary = 0;
        design_method = 'one_dicitionary_per_class';
        sparsification_strategy = 'ald';
        v1 = 0.1;
        v2 = 0.9;
        update_strategy = 'lms';
        update_rate = 0.1;
        pruning_strategy = 'error_score_based';
        min_score = -10;
        max_prototypes = 600;
        min_prototypes = 2;
        video_enabled = 0;
        regularization = 0.001;
        sigma = 2;
        alpha = 1;
        theta = 1;
        gamma = 2;
    end

    % Parameters
    properties (GetAccess = public, SetAccess = protected)
        
        Km = [];
        Kmc = [];
        Kinv = [];
        Kinvc = [];
        score = [];
        classification_history = [];
        times_selected = [];
        video = [];
        
    end

    methods

        % Constructor
        function self = spokClassifier()
            % Set the hyperparameters after initializing!
        end

        % Training Function (1 instance)
        function self = partial_fit(self,x,y)

            number_of_classes = length(y);

            [~,number_of_prototypes_old] = size(self.Cx);

            if (number_of_prototypes_old == 0)
                % Make a guess (yh = [1 -1 ... -1 -1]' : first class)
                self.yh = -1*ones(number_of_classes,1);
                self.yh(1) = 1;
                % Add sample to dictionary
                self = self.dictionaryGrow(x,y);
            else
                % Predict Output
                self = self.partial_predict(x);

                % Update number of times a prototype has been the winner
                self.times_selected(self.winner) = self.times_selected(self.winner) + 1;

                % Growing Strategy
                self = self.dictionaryGrow(x,y);

                % Update Strategy
                [~,number_of_prototypes_new] = size(self.Cx);
                if(number_of_prototypes_new == number_of_prototypes_old)
                    self = self.dictionaryUpdate(x,y);
                else
                    % For debug. Display dictionary size when it grows.
                    % disp(number_of_prototypes_new);
                end

                % Prunning Strategy
                self = self.updateScore(x,y);
                self = self.dictionaryPrune(x,y);

            end

        end

        % Training Function (N instances)
        function self = fit(self,X,Y)

            % Problem Initialization
            Xs = X;                  % Don't shuffle original data
            Ys = Y;                  % Don't shuffle original data
            [Nc,N] = size(Y);        % Total of classes and samples
            self.Yh = -1*ones(Nc,N); % Initialize outputs
            iteration = 0;           % Initialize number of iterations

            self.video = struct('cdata',cell(1,N*self.number_of_epochs),...
                                'colormap', cell(1,N*self.number_of_epochs));

            for epoch = 1:self.number_of_epochs

                for n = 1:N

                    if(self.video_enabled)
                        iteration = iteration + 1;
                        self.video(iteration) = prototypes_frame(self.Cx,X);
                    end

                    self = self.partial_fit(Xs(:,n),Ys(:,n));
                    self.Yh(:,n) = self.yh;

                end

                % Shuffle Data
                if(self.is_stationary)
                    I = randperm(N);
                    Xs = Xs(:,I);
                    Ys = Ys(:,I);
                end

                % Hold last classification labels
                if(epoch == self.number_of_epochs)
                    self = self.predict(X);
                end

            end % end epoch

        end % end fit

        function self = dictionaryGrow(self,x,y)
            
            [~,m] = size(self.Cx);     % Dictionary size
            [~,c] = max(y);            % Class of sample (Sequential encoding)
            [~,Cy_seq] = max(self.Cy); % Classes of dictionary (Sequential)
            mc = sum(Cy_seq == c);     % Number of prototypes from samples' class
            
            % Add first element to dictionary (total or from class)
            if (m == 0 || (Dm == 2 && mc == 0))
                self = self.addSample(x,y);
            else
                % Dont add if number of prototypes is too high
                if (m < self.max_prototypes)
                    
                    if(strcmp(self.design_method,'single_dictionary'))
                        Dx = self.Cx;
                        Dy = self.Cy;
                        Kinverse = self.Kinv;
                    elseif(strcmp(self.design_method,'one_dicitionary_per_class'))
                        Dx = self.Cx(:,Cy_seq == c);
                        Dy = self.Cy(:,Cy_seq == c);
                        Kinverse = self.Kinvc{c};
                    end
                    
                    % Get criterion result
                    if(strcmp(self.sparsification_strategy,'ald'))
                        OUTcrit = aldCriterion(self,Dx,x,Kinverse);
                    elseif(strcmp(self.sparsification_strategy,'coherence'))
                        OUTcrit = coherenceCriterion(self,Dx,x);
                    elseif(strcmp(self.sparsification_strategy,'novelty'))
                        OUTcrit = noveltyCriterion(self,Dx,Dy,x,y);
                    elseif(strcmp(self.sparsification_strategy,'surprise'))
                        OUTcrit = surpriseCriterion(self,Dx,Dy,x,y,Kinverse);
                    else % use ald as default
                        self.sparsification_strategy = 'ald';
                        OUTcrit = aldCriterion(self,Dx,x,Kinverse);
                    end
                    
                    % Expand or not Dictionary
                    if(OUTcrit.result == 1)
                        self = self.addSample(x,y);
                    end                 
                    
                end
                
            end
            
        end

        function self = dictionaryUpdate(self,x,y)
            
            if(~strcmp(self.update_strategy,'none'))
                
                % Get sequential class of sample
                [~,yt_seq] = max(y);

                % Find nearest prototype from whole dictionary
                if(strcmp(self.design_method,'single_dictionary'))
                    winner = self.findWinnerPrototype(self.Cx,x,self);
                % Find nearest prototype from class conditional dictionary
                elseif(strcmp(self.design_method,'one_dicitionary_per_class'))
                    [~,Dy_seq] = max(self.Cy);
                    Dx_c = self.Cx(:,Dy_seq == yt_seq);
                    win_c = self.findWinnerPrototype(Dx_c,x,self);
                    winner = self.findWinnerPrototype(self.Cx,Dx_c(:,win_c),self);
                end
                
                % Find nearest prototype output
                y_new = self.Cy(:,winner);
                [~,y_new_seq] = max(y_new);
    
                % Update Closest prototype (new one)
                if(strcmp(self.update_strategy,'wta'))
                    x_new = self.Cx(:,win) + self.update_rate*(x - self.Cx(:,win));
                elseif(strcmp(self.update_strategy,'lvq'))
                    if(yt_seq == y_new_seq)
                        x_new = self.Cx(:,win) + self.update_rate*(x - self.Cx(:,win));
                    else
                        x_new = self.Cx(:,win) - self.update_rate*(x - self.Cx(:,win));
                    end
                elseif(strcmp(self.update_strategy,'wta_der'))
                    x_new = self.Cx(:,win) + self.update_rate*...
                                             kernelDerivative(x,self.Cx(:,win),self);
                elseif(strcmp(self.update_strategy,'lvq_der'))
                    if(yt_seq == y_new_seq)
                        x_new = self.Cx(:,win) + self.update_rate*...
                                             kernelDerivative(x,self.Cx(:,win),self);
                    else
                        x_new = self.Cx(:,win) - self.update_rate*...
                                             kernelDerivative(x,self.Cx(:,win),self);
                    end
                end
                
                % Hold varibles used for prunning
                score_aux = self.score(winner);
                class_hist_aux = self.classification_history(winner);
                times_selected_aux = self.times_selected(winner);
                
                % Remove "old" prototype and add "updated" one from dictionary
                self = self.removeSample(winner);
                self = self.addSample(x_new,y_new);
                
                % Get variables for prunning
                self.score(end) = score_aux;
                self.classification_history(end) = class_hist_aux;
                self.times_selected(end) = times_selected_aux;
                                
            end
            
        end

        function self = dictionaryPrune(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

        function self = addSample(self,x,y)
            
            % Initializations
            ktt = kernelFunction(x,x,self); % kernel of sample and itself
            Nc = length(y);                 % number of classes
            [~,c] = max(y);                 % class of sample
            [~,m] = size(self.Cx);       	% # of prototypes in the dict
            [~,Dy_seq] = max(self.Cy);      % Sequential classes of dict
            
            % Add sample to dictionary
            self.Cx = [self.Cx,x];
            self.Cy = [self.Cy,y];
            
            % Add variables used to prunning
            self.score = [self.score,0];
            self.classification_history = [self.classification_history,0];
            self.times_selected = [self.times_selected,0];
            
            % Update Kernel Matrices
            
            if (m == 0)

                % Init Kernel matrix and its inverse for each class
                self.Kmc = cell(Nc,1);
                self.Kmc{c} = ktt + self.regularization;
                self.Kinvc = cell(Nc,1);
                self.Kinvc{c} = 1/self.Kmc{c};
                
                % Init Kernel matrix and its inverse for dataset
                self.Km = ktt + self.regularization;
                self.Kinv = 1/self.Km;
                
            else
                
                % Get number of prototypes from samples' class
                mc = sum(Dy_seq == c);
                
                % Init kernel matrix and its inverse of samples' class
                if (mc == 0)
                    self.Kmc{c} = ktt + self.regularization;
                    self.Kinvc{c} = 1/self.Kmc{c};
                    
                % Update kernel matrix and its inverse of samples' class
                else
                    % Get auxiliary variables
                    Cx_c = self.Cx(:,Dy_seq == c); % Inputs from class c
                    kt_c = kernelVector(Cx_c,x,self);
                    at_c = self.Kinvc{c}*kt_c;
                    delta_c = (ktt - kt_c'*at_c) + self.regularization;
                    % Update Kernel matrix
                    self.Kmc{c} = [self.Kmc{c}, kt_c; ...
                                   kt_c', ktt + self.regularization];
                    % Update Inverse Kernel matrix
                    self.Kinvc{c} = (1/delta_c)*...
                                  [delta_c*self.Kinvc{c} + at_c*at_c',-at_c; -at_c', 1];
                end
                
                % Get auxiliary variables
                kt = kernelVector(self.Cx,x,model);
                at = self.Kinv * kt;
                delta = (ktt - kt'*at) + self.regularization;
                
                % Update kernel matrix and its inverse for dataset
                self.Km = [self.Km, kt; kt', ktt + self.regularization];
                self.Kinv = (1/delta)*[delta*self.Kinv + at*at', -at; -at', 1];
                
            end
            
            
        end

        function self = removeSample(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

        function self = updateScore(self,x,y)
            % ToDo - All
            self = self + x + y;
        end

    end % end methods
    
    methods (Static)
        
        function ALDout = aldCriterion(self,Dx,x,Kinverse)
            
            % Calculate auxiliary variables
            ktt = kernelFunction(x,x,self);
            kt = kernelVector(Dx,x,model);
            
            % Calculate ald coefficients
            at = Kinverse * kt;
            
            % Calculate "Normalized delta"
            delta = (ktt - kt'*at);
            delta = delta + self.regularization;
            
            % Calculate Criterion (boolean)
            result = (delta > v1);
            
            % Hold results
            ALDout.result = result;
            ALDout.ktt = ktt;
            ALDout.kt = kt;
            ALDout.at = at;
            ALDout.delta = delta;            
            
        end
        
        function COHout = coherenceCriterion(self,Dx,x)
            % ToDo - all
            COHout = self + Dx + x;
        end
        
        function NOVout = noveltyCriterion(self,Dx,Dy,x,y)
            % ToDo - all
            NOVout = self + Dx + Dy + x + y;
        end
        
        function SURout = surpriseCriterion(self,Dx,Dy,x,y,Kinverse)
            % ToDo - all
            SURout = self + Dx + Dy + x + y + Kinverse;
        end
        
    end

end














