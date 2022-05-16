classdef classificationExperiment

    properties
        number_of_realizations = 5;
        classifier_name = 'lms';
        dataset_name = 'iris';
        label_encoding = 'bipolar';
        normalization = 'zscore';
        split_method = 'random';
        percentage_for_training = 0.7;
        filename = [];

        hp_optm_method = 'random';
        hp_optm_max_interations = 10;
        hp_optm_cost_function = 'error';
        hp_optm_weighting_factor = 0.5;
        hp_optm_struct = [];
        
        dataset = [];
        classifier = [];
        classification_stats_tr = [];
        classification_stats_ts = [];
    end
    
    properties (GetAccess = public, SetAccess = protected)
        
        data_acc = [];
        classifier_acc = [];
        stats_tr_acc = [];
        stats_ts_acc = [];
        stats_tr_all = [];
        stats_ts_all = [];
        
    end
    
    methods
        
        function self = classificationExperiment()
            % Set the hyperparameters after initializing!
        end
        
        function self = run(self)
            
            self.data_acc = cell(self.number_of_realizations,1);
            self.classifier_acc = cell(self.number_of_realizations,1);
            self.stats_tr_acc = cell(self.number_of_realizations,1);
            self.stats_ts_acc = cell(self.number_of_realizations,1);
            
            if(isempty(self.classifier))
                self.classifier = initializeClassifier(self.classifier_name);
            end
            
            if(isempty(self.dataset))
                self.dataset = loadClassificationDataset(self.dataset_name);
            end
            
            self.dataset = encodeLabels(self.dataset,self.label_encoding);

            normalizer = dataNormalizer();
            normalizer.normalization = self.normalization;
            
            statsGen1turn = classificationStatistics1turn();

            self.classification_stats_tr = classificationStatisticsNturns();
            self.classification_stats_ts = classificationStatisticsNturns(); 

            for r = 1:self.number_of_realizations
                
                disp('Turn and Time');
                disp(r);
                display(datestr(now));
                
                datasets = splitDataset(self.dataset, ...
                                        self.split_method, ...
                                        self.percentage_for_training);
                
                normalizer = normalizer.fit(datasets.data_tr.input);
                datasets.data_tr.input = normalizer.transform(datasets.data_tr.input);
                datasets.data_ts.input = normalizer.transform(datasets.data_ts.input);
                
                self.data_acc{r} = datasets;

%                 % ToDo - hyperOpt class!
                
                self.classifier_acc{r} = self.classifier.fit(datasets.data_tr.input, ...
                                                             datasets.data_tr.output);

                Yh_tr = self.classifier_acc{r}.predict(datasets.data_tr.input);
                Yh_ts = self.classifier_acc{r}.predict(datasets.data_ts.input);
                
                stats_tr = statsGen1turn.calculate_all(datasets.data_tr.output,Yh_tr);
                stats_ts = statsGen1turn.calculate_all(datasets.data_ts.output,Yh_ts);

                self.stats_tr_acc{r} = stats_tr;
                self.stats_ts_acc{r} = stats_ts;
                
                self.classification_stats_tr = ...
                      self.classification_stats_tr.addResult(stats_tr);

                self.classification_stats_ts = ...
                      self.classification_stats_ts.addResult(stats_ts);

            end % end for loop (realizations)
            
            self.classification_stats_tr = self.classification_stats_tr.calculate();
            self.classification_stats_ts = self.classification_stats_ts.calculate();

        end % end run()
        
        function show_results(self)
            
            disp(self.classification_stats_ts.acc_vect);
            
        end % end show_results()
        
        function build_filename(self)
            
            disp(self.filename);
            
        end % end build_filename()
        
    end
    
end


















