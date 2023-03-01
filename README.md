# MLPE (Metric Lattice for Performance Estimation)

 *MLPE takes a binary ML model for which classification thresholds have already been determined and provides clinicians with patient-specific model performance estimates.* 

To use MPLE on an ML model, load in a single dataframe with train and test data used to build an ML model, with demographic attributes and feature data such that:

      -Predicted labels are last column of dataframe
      -Labels are penultimate column of dataframe
      -Demographic attributes are pandas type Object (string). Numerical demographic data should be binned.
      -Feature data is everything else, and should be processed as it was to train the ML model
      -Ensure there are no NaN values in feature data (replace with choice of median, mean, etc.)
      -NaN is ok in demographic categories
      -Training data is defined by having NaN predicted labels
      -Test data has is defined by having non-NaN predicted labels

**prepare_example_dataset.py pulls synthetic data from the Synthea COVID-19 dataset and will produce example data that fit these criteria.**

## About the package

The MLPE fit() function re-parameterizes data using metric learning on provided demographic information, and a lattice of model specificity or sensitivity across feature space is computed. It also alerts the researcher if the train/test split varies significantly in its distribution across the new metric feature space. 

The MLPE predict() function returns a confidence interval on model specificity or sensitivity based on patient data.

The MLPE transform() function projects patient data into the learned metric space, returning the new data features.

The MLPE feedback() function provides the researcher with demographic intersections which may have especially poor model performance, in order to alert the researcher to patient subpopulations which may benefit from additional training data or revised classification thresholds. 


## Parameters 

Most-used parameters are listed below. The Synthea COVID-19 dataset was able to be run on default parameters using an Apple M1 chip on 8 GB of RAM. 

### fit()
    
    train_and_test_data: pandas DataFrame
        formatted as explained above. Example can be made by running prepare_example_dataset.py
        
    remove_outliers_thresh: float between 0 and 1, default=0.01
        for training metric learning, within each demogrpahic attribute, 
        records including categories representing less than remove_outliers_thresh of individuals are removed
        
    balanced_pairs: int, default=5000
        number of pairs (both positive and negative) to be used for metric learning
        
    metric_learn_max_proj: int, default=100000
        maximum number of projections used in metric learning. If transform() returns feature data unaltered, 
        this parameter must be increased.
        
    performance_metric: {'sensitivity','specificity','precision','accuracy'}, default='sensitivity' 
        ML model performance metric on which to build the lattice
        
    desired_points_in_lattice: int, default=100000
        number of lattice points across which to calculate performance_metric
    
    r_multiple: float or int, default=1
        n-ball radius as a multiple of lattice point width
    
    csv: boolean, default=True
        whether to output mmc_L and lattice_scores outputs as csvs into the current directory 
        
    
### predict()
    
    patient_feature_data: pandas DataFrame
        patient feature data only, processesed the same as train_and_test_data
        
    information_source: {'self','csv'}, default='self' 
        source of fit() output. If 'self', takes in lattice and L matrix directly from fit(). 
        If 'csv', automatically reads from lattice_structure.csv, lattice_scores.csv, and mmc_L.csv in the current directory. 
        'path_structure', 'suffix_structure',  'path_scores', 'suffix_scores', 'path_L' and 'suffix_L' parameters can 
        additionally be used to state the path and file name of these inputs.
        
   
### transform()
      
    patient_feature_data: pandas DataFrame
        patient feature data only, processesed the same as train_and_test_data
        
    information_source: {'self','csv'}, default='self' 
        source of fit() output. If 'self', takes in lattice and L matrix directly from fit(). 
        If 'csv', automatically reads from lattice_structure.csv and mmc_L.csv in the current directory. 
        'path_lattice', 'suffix_lattice', 'path_L' and 'suffix_L' parameters can additionally be used to 
        state the path and file name of these inputs.
        
  
## Returns

### fit()
    
    mmc_L: pandas DataFrame
        L matrix to transform patient data into learned metric space. Also output to csv if csv=True.
    
    lattice_scores: pandas DataFrame
        lattice of model performance across learned metric space. Also output to csv if csv=True.
        
    lattice_structure: pandas DataFrame
        structural information needed to recreate the performance lattice in predict(). Also output to csv if csv=True.
    
    
### predict()

    confidence_intervals: pandas DataFrame
        confidence interval(s) on the desired metric of model performance 
        
        
### transform()
    
    transformed_features: pandas DataFrame
        patient features transformed into learned metric space. Can be used for clustering 
        and examinations of latent/lurking variables and hidden stratification.
        
        
# Contact
If you have a query regarding use of this tool or would like to collaborate on an improved version of MLPE, please contact us at kellensandvik(at)gmail.com or jesseaviv(at)gmail.com.
