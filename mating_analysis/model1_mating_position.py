import sys
sys.path.append("/Users/annika/Documents/git_repos/2-photon/plotting")
import mating_angles_learn
from mating_angles_learn import learning_pipeline,apply_pretrained,prepare_training_data,evalulate_pretrained

#paths to data - change if model is to be trained/applied on new data
path_to_training='/Volumes/LaCie/Projects/Matthew/behaviour/R1_Exp2_Chamber1_CompressEndDLC_resnet50_crf15_tstMar12shuffle1_300000filtered.csv' 
path_to_labels='/Volumes/LaCie/Projects/Matthew/behaviour/Exp2_chamber1.csv'  
path_to_test='/Volumes/LaCie/Projects/Matthew/behaviour/Exp2_chamber1_test.csv'  
path_to_Test_csv='/Volumes/LaCie/Projects/Matthew/behaviour/R1_Exp2_Chamber1_CompressEndDLC_resnet50_crf15_tstMar12shuffle1_300000filtered.csv' 
path_to_data1='/Volumes/LaCie/Projects/Matthew/behaviour/R1_G10Ctrl3_Chamber3_CompressEndDLC_resnet50_crf15_tstMar12shuffle1_300000filtered.csv' 
path_to_data2='/Volumes/LaCie/Projects/Matthew/behaviour/R1_Exp2_Chamber1_CompressEndDLC_resnet50_crf15_tstMar12shuffle1_300000filtered.csv' 

features=["angles_w_scaled","angles_b_scaled","head_dist_scaled","abd_dist_scaled","tilting_index_scaled"]
#training the model
models=learning_pipeline(path_to_training,path_to_labels,featurelist=features)  
#applying the pretrained model 
#if model is not to be trained again, the pretrained model can be loaded from the file with load_pretrained()
data1=prepare_training_data(path_to_data1,copstartframe=1300,featurelist=features)
data2=prepare_training_data(path_to_data2,copstartframe=562,featurelist=features)
predictions_data1,fraction1=apply_pretrained(models,data1,startframe=1300) 
predictions_data2,fraction2=apply_pretrained(models,data2,startframe=562) 
print("Fraction of abnormal copulation in group 1: {}".format(fraction1))
print("Fraction of abnormal copulation in group 2: {}".format(fraction2))

#model evaluation
print("Evaluating model on new data...")
scores=evalulate_pretrained(path_to_Test_csv,path_to_test,featurelist=features)


