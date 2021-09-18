import mating_angles_learn_model2
from mating_angles_learn_model2 import apply_pretrained, prepare_training_data, evalulate_pretrained, load_pretrained


def test_apply(path_to_data1,
               path_to_data2,
               path_to_test,
               path_to_Test_csv,
               path_to_test_2,
               path_to_test2_csv):
    """Apply Model to new data and calculate the fraction of abnormal copulation"""
    features = ["angles_b_scaled",
                "head_dist_scaled",
                "abd_dist_scaled",
                "tilting_index_scaled"]

    models = load_pretrained()

    # applying the pretrained model
    data1 = prepare_training_data(
        path_to_data1, copstartframe=562, featurelist=features)
    data2 = prepare_training_data(
        path_to_data2, copstartframe=1936, featurelist=features)
    predictions_data1, fraction1 = apply_pretrained(
        models, data1, startframe=562)
    predictions_data2, fraction2 = apply_pretrained(
        models, data2, startframe=1936)

    print("Fraction of abnormal copulation in group 1: {}".format(fraction1))
    print("Fraction of abnormal copulation in group 2: {}".format(fraction2))

    # model evaluation
    print("Evaluating model on new data...")
    scores = evalulate_pretrained(
        path_to_Test_csv, path_to_test, featurelist=features)
    print("Evaluating model on new data...")
    scores = evalulate_pretrained(
        path_to_test2_csv, path_to_test2, featurelist=features)


# paths to data - change if model is to be trained/applied on new data
path_to_test2_csv = '/Volumes/LaCie/Projects/Matthew/behaviour/R1_Exp2_Chamber1DLC_resnet50_Model3Apr24shuffle1_300000filtered.csv'
path_to_test2 = '/Volumes/LaCie/Projects/Matthew/behaviour/Exp2_chamber1.csv'
path_to_test = '/Volumes/LaCie/Projects/Matthew/behaviour/Ch3_chamber3.csv'
path_to_Test_csv = '/Volumes/LaCie/Projects/Matthew/behaviour/R1_G10Ctrl3_Chamber3DLC_resnet50_Model3Apr24shuffle1_300000filtered.csv'
path_to_data1 = '/Volumes/LaCie/Projects/Matthew/behaviour/R1_Exp2_Chamber1DLC_resnet50_Model3Apr24shuffle1_300000filtered.csv'
path_to_data2 = '/Volumes/LaCie/Projects/Matthew/behaviour/R1_G10Ctrl3_Chamber3DLC_resnet50_Model3Apr24shuffle1_300000filtered.csv'

if __name__ == "__main__":
    test_apply(path_to_data1,
               path_to_data2,
               path_to_test,
               path_to_Test_csv,
               path_to_test2,
               path_to_test2_csv)
