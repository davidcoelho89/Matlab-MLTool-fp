classdef spokClassifier
   % 
   % SParse Online adptive Kernel Classifier
   %
   % Properties (Hyperparameters):
   %
   %    number_of_epochs = <integer>
   %        if > 1, "shows the data" more than once to the algorithm
   %    is_stationary = [0 or 1]
   %        
   %    design_method = how the dictionary will be built
   %        = 'one_dicitionary_per_class'
   %        = 'single_dictionary'
   %    sparsification_strategy = define how the prototypes will be selected
   %        = 'ald'
   %        = 'coherence'
   %        = 'novelty'
   %        = 'surprise'
   %    v1 = 
   %    v2 = 
   %    update_strategy = define how the prototypes will be updated
   %        = 'none'
   %        = 'wta' (lms, unsupervised)
   %        = 'lvq' (supervised)
   %    update_rate = <real>
   %        [0 to 1]
   %    pruning_strategy = define how the prototypes will be prunned
   %        = 'none'
   %        = 'drift_based'
   %        = 'hits_and_error'
   %    min_score = 
   %    max_prototypes = 
   %    min_prototypes = 
   %    video_enabled = 
   %    nearest_neighbors = 
   %    knn_aproximation = 
   %    kernel_type = 
   %    regularization = 
   %    sigma = 
   %    alpha = 
   %    theta = 
   %    gamma = 
   %
   % Properties (Parameters)
   %
   %    Cx = Clusters' centroids (prototypes)
   %    Cy = Clusters' labels 
   %    Km = Kernel Matrix of Entire Dictionary 
   %    Kinv = Kernel Matrix for each class (cell) 
   %    Kinvc = Inverse Kernel Matrix for each class (cell)
   %    score = Used for prunning method
   %    classification_history = Used for prunning method
   %    times_selected = Used for prunning method
   %    video = frame structure (can be played with 'video function')
   %    Yh = Hold all predictions
   %
   % Methods:
   %
   %	spokClassifier()        % Constructor
   %	partial_fit(self,x,y)	% Training Function (1 instance)
   %	fit(self,X,Y)           % Training Function (N instances)
   %	Yh = predict(self,X)    % Prediction Function
   
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
        max_prototypes = 600;  % ("Budget")
        min_prototypes = 2;    % ("restriction")
        video_enabled = 0;
        nearest_neighbors = 1; % KNN
        knn_aproximation = 'majority_voting';
        kernel_type = 'gaussian';
        regularization = 0.001;
        sigma = 2;
        alpha = 1;
        theta = 1;
        gamma = 2;
   end
   
   % Parameters
   properties (GetAccess = public, SetAccess = protected)
       Cx = [];     % Clusters' centroids (prototypes)
       Cy = [];     % Clusters' labels 
       Km = [];     % Kernel Matrix of Entire Dictionary  
       Kmc = [];    % Kernel Matrix for each class (cell) 
       Kinv = [];   % Inverse Kernel Matrix of Dictionary
       Kinvc = [];  % Inverse Kernel Matrix for each class (cell)
       score = [];          % Used for prunning method
       class_history = [];  % Used for prunning method
       times_selected = []; % Used for prunning method
       video = []; % frame structure (can be played with 'video function')
       Yh = [];    % Hold all predictions
   end
   
   methods

       % Constructor
       function self = spokClassifier()
           % Set the hyperparameters after initializing!
       end
       
       % Training Function (1 instance)
       function self = partial_fit(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
       % Training Function (N instances)
       function self = fit(self,X,Y)
           
           for ep = 1:self.number_of_epochs
               
               % Problem Initialization
               [Nc,N] = size(Y);        % Total of classes and samples
               self.Yh = -1*ones(Nc,N); % Initialize outputs
               
               % Shuffle Data
               if(self.is_stationary)
                   I = randperm(N);
                   X = X(:,I);
                   Y = Y(:,I);
               end
               
               for n = 1:N
                   
                   xn = X(:,n);
                   yn = Y(:,n);
                   
                   [~,number_of_prototypes] = size(self.Cx);
                   
                   if(number_of_prototypes == 0)
                       self.Yh(1,n) = 1; % Make a guess (yh=1: first class)
                       self = self.dictionaryGrow(self,xn,yn);
                       continue; % call next sample
                   end
                   
                   
                   
               end
               
               
           end
           
           % ToDo - All
           self = self + X + Y;
       end
       
       % Prediction Function
       function Yh = predict(self,X)
           % ToDo - All
           Yh = prototypesClassify(self,X);
       end
       
       function self = dictionaryGrow(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
       function self = dictionaryUpdate(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
       function self = dictionaryPrune(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
       function self = addSample(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
       function self = removeSample(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
       function self = updateScore(self,x,y)
           % ToDo - All
           self = self + x + y;
       end
       
   end
    
end