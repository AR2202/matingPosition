# Short User Guide

## fly mating video analysis scripts

This is a collection of python scripts for the analysis of tracked fly mating videos.
The tracking is done using Deep Lab Cut. The Deep Lab Cut model was trained to label
points on the flies' head, abdomen and shoulders. These scripts are for calculating
the mating angle and tilting index. Furthermore, a machine learning model is trained
on these features to recognize normal vs. abnormal mating positions

### Requirements

* python 3
* the following python libraries:
* numpy
* scipy
* scikit-learn
* matplotlib
* pandas
* joblib
* pytest (for tests only)

### Usage

All of the descriptions below assume you have trained a Deep Lab Cut model to recognize the head, abdomen of both flies and shoulders of the male flies and applied that model to your videos. The outpul csv file of the model is the input data to these scripts.

* [calculating the mating angle and tilting index](#matingangle)
* [Using a script to calculate the median mating angles and tilting indices in a video](#medianma)
* [Removing frames close to the wall](#removewall)
* [Training a machine learning model to classify frames](#ml)

<a name=#matingangle></a>

#### calculating the mating angle and tilting index

To calculate the mating angle for each row of your csv file, use the function unfiltered_outputs() from the mating_angles_model2 module:

`mating_angles,wing_dist_male,abd_dist,head_dist=unfiltered_outputs(path_to_csv)`

mating_angles will be an array of your mating angles. This function also outputs wing distance in the male, distance of male to female abdomen, and distance of male to female head.

Alternatively, you can filter your data to only include those frames that exceed a minimum likelihood (the P argument) of the labelling that was done by Deep Lab Cut. To use filtering, use the filtered_outputs() function:

`mating_angles,wing_dist_male,abd_dist,head_dist=filtered_outputs(path_to_csv,P)`

To calculate the tilting index, use the function tilting_index from the mating_angles_model2 module:

`tilting_ind=tilting_index(wing_distance_male,copulationstartframe)`

tilting_ind will be an array of your tilting indices. The input arguments are the male wing distance, which can be obtained from the function unfiltered_outputs or filtered_outputs (see above) and the copulationstartframe= the start of your flies' copulation.

<a name=#medianma></a>

#### Using a script to calculate the median mating angles and tilting indices in a video

The sample script  model2_mating_angles can be used to calculate median mating angles and tilting indices. The paths are examples and have to be changed to the actual paths to the data. At the top of the file, change the line:
`sys.path.append("path")`
to the path of the mating_angles_model2.py file

Additional lines might need to be commented out if you don't have labelled data. You can also change some lines to print, plot or save the mating_angles and tilting_indices.

<a name=#removewall></a>

#### Removing frames close to the wall

The filtered_outputs() and unfiltered_outputs() functions in the mating_angles_model2.py module take the optional keyword arguments removeWall and minWallDist. Defaults are removeWall=False and minWallDist=3. removeWall specifies whether frames where flies are on the side wall should be removed. minWallDist is the minimum distance to the wall (in pixels) that flies should have if frames are to be kept.

<a name=#ml></a>

#### Training a machine learning model to classify frames into 'normal' or 'abnormal' mating positions

* [Creating training data](#traindata)
* [Training the model](#train)
* [Training the model from a script](#trainscript)
* [Loading a pretrained model](#loadmodel)
* [Evaluating a pretrained model](#evalmodel)
* [Applying a pretrained model to new data](#applymodel)
* [Using a script to apply a model](#applyscript)

<a name=#traindata></a>

#### Creating training data

Label some frames in your videos as in the example_labels.csv file. The first row should contain column headers. The columns should be:

* column 1: copulation start frame for that video (one row only)
* column 2: normal position frame numbers, pairs of consecutive rows will be interpreted as start and end frame of a normal period
* column 3: abnormal position frame numbers, pairs of consecutive rows will be interpreted as start and end frame of a normal period

<a name=#train></a>

#### Training the model

Use the function learning_pipeline() from the mating_angles_learn_model2.py module:

`models=learning_pipeline(list_of_paths_to_training,list_of_paths_to_labels,featurelist=features)`

featurelist ist an optional argument, you can specify the features you want to use for training. The default is:

`features=["angles_b_scaled","head_dist_scaled","abd_dist_scaled","tilting_index_scaled"]`

Select subsets of those features as you wish.

<a name=#trainscript></a>

#### Training the model from a script

The script model2_mating_position.py gives an example of how to train the model. The paths have to be changed.

<a name=#loadmodel></a>

#### Loading a pretrained model

Use the load_pretrained() function from the mating_angles_learn_model2.py module:

`models=load_pretrained(path_to_model)`

<a name=#evalmodel></a>

#### Evaluating a pretrained model on new data

* create new labelled test data as outlined above
* Use the evaluate_pretrained() function from the mating_angles_learn_model2.py module: 

    `scores=evalulate_pretrained(list_of_paths_to_test_csv,list_of_paths_to_test,featurelist=features)`

<a name=#applymodel></a>

#### Applying the pretrained model to new data

`models=load_pretrained(path_to_model)`

`data=prepare_training_data(path_to_data_csv,copstartframe=copulationstartframe,featurelist=features)`

`predictions_data1,fraction1=apply_pretrained(models,data,startframe=copulationstartframe)`

The keyword arguments are optional.

<a name=#applyscript></a>

#### Using a script to evaluate and apply an existing model

The script model2_test_apply.py gives an example of how to train the model. The paths have to be changed.
