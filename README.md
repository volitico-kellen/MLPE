# MLPE (Metric Lattice for Performance Estimation)

 *MLPE takes a binary ML model for which classification thresholds have already been determined and provides clinicians with patient-specific model performance estimates for use at prediction time.* 

To use MLPE on an ML model, load in a single dataframe with train and test data used to build the ML model, with demographic attributes and feature data such that:

      -Predicted labels are last column of dataframe
      -Labels are penultimate column of dataframe
      -Demographic attributes are pandas type Object (string). Numerical demographic data should be binned. 
        <25 bins or categories per deomgraphic attribute recommended
      -Feature data is everything else, and should be processed as it was to train the ML model
      -Ensure there are no NaN values in feature data (replace with choice of median, mean, etc.)
      -NaN is ok in demographic categories
      -Training data is defined by having NaN predicted labels
      -Test data has is defined by having non-NaN predicted labels

**prepare_example_dataset.py pulls synthetic data from the Synthea COVID-19 dataset and will produce example data that fit these criteria.**

Methods and motivation:
https://drive.google.com/file/d/1WcJXqrhoPy9FAtU3KtaKqPlRug7AVsQj/view?usp=sharing

Demo: https://youtu.be/_9Jkyv-FBuw

## Example Use
Example use of the full method can be found in example_usage.ipynb.

      >> mlpe = mititgate_disparity.MLPE()
      >> MLPE.fit(data)

# About the package

The MLPE fit() function re-parameterizes data using metric learning on provided demographic information, and a lattice of model specificity or sensitivity across feature space is computed. It also alerts the researcher if the train/test split varies significantly in its distribution across the new metric feature space. 

The MLPE predict() function returns a confidence interval on model specificity or sensitivity based on patient data. Can be used to evaluate ML model performance at prediction time.

The MLPE transform() function projects patient data into the learned metric space, returning the new data features. Can be used for clustering and examinations of latent/lurking variables and hidden stratification.

The MLPE feedback() function provides the researcher with demographic intersections which may have especially poor model performance, in order to alert the researcher to patient subpopulations which may benefit from additional training data or revised classification thresholds. 


# Parameters 

Most-used parameters are listed below. The Synthea COVID-19 dataset was able to be run on default parameters using an Apple M1 chip on 8 GB of RAM. 

### fit()
    
    train_and_test_data: pandas DataFrame
        formatted as explained above. Example can be made by running prepare_example_dataset.py
        
    remove_outliers_thresh: float between 0 and 1, default=0.01
        for training metric learning, within each demogrpahic attribute, 
        records including categories representing less than remove_outliers_thresh of individuals are removed
        
    n_balanced_pairs: int, default=5000
        number of pairs (both positive and negative) to be used for metric learning
        
    metric_learn_max_proj: int, default=100000
        maximum number of projections used in metric learning. If MLPE transform() returns feature data unaltered, 
        this parameter must be increased.
        
    performance_metric: {'sensitivity','specificity','precision','accuracy'}, default='sensitivity' 
        ML model performance metric on which to build the lattice
        
    desired_points_in_lattice: int, default=5000
        number of desired lattice points across which to calculate performance_metric. 
        The true number of lattice points n will be desired_points_in_lattice/2 < n <= desired_points_in_lattice
    
    r_multiple: float or int, default=1.3
        n-ball radius as a multiple of lattice point width, for calculation of model performance
    
    csv: bool, default=True
        whether to output mmc_L, lattice_structure, and lattice_scores outputs as csvs into the current directory.
        'path' and 'suffix' parameters can additionally be used to state altered path and file names of these inputs.
        
    
### predict()
    
    patient_feature_data: pandas DataFrame
        patient feature data only, processesed the same as train_and_test_data (do not MLPE transform() prior to predict)
        
    information_source: {'self','csv'}, default='self' 
        source of fit() output. If 'self', takes in lattice and L matrix directly from fit(). 
        If 'csv', automatically reads from lattice_structure.csv, lattice_scores.csv, and mmc_L.csv in the current directory. 
        'path' and 'suffix' parameters can additionally be used to state altered path and file names of these inputs.
        
   
### transform()
      
    patient_feature_data: pandas DataFrame
        patient feature data only, processesed the same as train_and_test_data
        
    information_source: {'self','csv'}, default='self' 
        source of fit() output. If 'self', takes in lattice and L matrix directly from fit(). 
        If 'csv', automatically reads from mmc_L.csv in the current directory. 
        'path' and 'suffix' parameters can additionally be used to state altered path and file names of these inputs.
        

### feedback()
      
    Note: Can only be run after running fit()
    
    information_source: {'self','csv'}, default='self' 
        source of fit() output. If 'self', takes in lattice and L matrix directly from fit(). 
        If 'csv', automatically reads from lattice_structure.csv, lattice_scores.csv, and mmc_L.csv in the current directory. 
        'path' and 'suffix' parameters can additionally be used to state altered path and file names of these inputs.
    
    level: int, default=2
        the number of demographic categories to enumerate
    
    sort_by: {'widest','lowest'}, default='widest'
        how result DataFrame is sorted, decreasing by width of the CI or increasing by mean
        
### in measure_disparity.py: measure_disparity()
    
    patient_info: pandas DataFrame
        patient demographic attributes only, formatted/binned same as train_and_test_data
        
    low_CI_scores: array-like of shape (n_samples,)
        the 5th percentile outputs of MLPE predict() on the chosen set of patients
        
    level: int, default=1
        the number of demographic categories to enumerate
        
  
# Returns

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
        
        
### feedback()
    
    subgroup_summary: pandas DataFrame
        summary statistics of model training set demographic subgroup estimated performance
        
### in measure_disparity.py: measure_disparity()

    class_summary: pandas DataFrame
        summary statistic of estimated model performance for each demographic class
        
# Troubleshooting

## MMC
### Metric learning eigenvalues do not converge:
try re-running fit(). New data pairs will be chosen, which often solves the issue. This may take a few tries

### Metric learning does not converge in the default number of iters (1000):
the max iters can be increased with the parameter mmc_max_iter

### Metric learning does not learn a new metric space:
metric_learn_max_proj parameter must be increased
      
## Pair Selection
### No or too few pairs are able to be selected:
pair selection removes patient records from metric learning traning if they contain a demographic attribute represented by less than remove_outliers_thresh. Try decreasing remove_outliers_thresh or using wider bins for demographic data

## Lattice
### Lattice confidence intervals are all too wide:
try adjusting r_multiple parameter and desired_points_in_lattice. Increasing r_multiple includes more datapoints per lattice point

# Contact
If you have a query regarding use of this tool, a bug to report, or would like to collaborate on an improved version of MLPE, please contact us at kellensandvik(at)gmail.com or jesseaviv(at)gmail.com
