import pytest
import math
import pandas as pd
import numpy as np
import sys
import mating_angles_model2
from mating_angles_model2 import signDeltax2, mating_angle_from_angles
from mating_angles_model2 import mating_angle_from_body_axis
from mating_angles_model2 import mating_angle_from_body_axis_pd_df
from mating_angles_model2 import filter_by_likelihood_body, load_csv_file
from mating_angles_model2 import unfiltered_outputs, filtered_outputs
from mating_angles_model2 import centroids, centroid_distance, tilting_index
from mating_angles_model2 import tilting_index_all_frames
from mating_angles_model2 import angle_from_cos, mating_angle_from_cos_pd_df
from mating_angles_learn_model2 import scale_filtered, scale_unfiltered
from mating_angles_learn_model2 import prepare_training_data
sys.path.append("./mating_analysis")


# this file contains the tests for the mating_angles_model2
#
# path to test data
testdatapath = "./pytests/testdata_pytest/exampledata.csv"


def test_signDeltax2():
    assert signDeltax2(3) == 1
    assert signDeltax2(-2) == -1
    assert signDeltax2(0) == 0


def test_mating_angle_from_angles():
    assert mating_angle_from_angles(1.5, 0.75, 1) == 0.75
    assert mating_angle_from_angles(-1.5, -0.75, 1) == 0.75
    assert mating_angle_from_angles(0.75, 1.5, 1) == 0.75
    assert mating_angle_from_angles(-0.75, 0.75, 1) == 1.5

    assert mating_angle_from_angles(-0.75, 0.75, -1) == math.pi-1.5
    assert mating_angle_from_angles(-0.75, 0.75, 0) == math.pi-1.5


def test_mating_angle_from_body_axis():
    assert mating_angle_from_body_axis(2, 1, 1, 0, -1, 2, 0, 1) == math.pi/2


def test_angle_from_cos():
    assert angle_from_cos(2, 1, 1, 0, -1, 2, 0, 1) == math.pi/2
    assert angle_from_cos(2, 1, 1, 0, 0, 1, -1, 2) == math.pi/2
    assert angle_from_cos(2, 0, 1, 0, 0, 1, -1, 2) < math.pi/2
    assert angle_from_cos(2, 0, 1, 0, -1, 2, 0, 1) > math.pi/2


def test_zero_division_mating_angle_from_body_axis():
    with pytest.raises(ZeroDivisionError):
        mating_angle_from_body_axis(2, 0, 1, 0, -0, 2, 0, 1)


def test_zero_division_angle_from_cos():
    assert angle_from_cos(2, 0, 1, 0, 0, 2, 0, 1) == math.pi/2


def test_mating_angle_from_body_axis_pd_df():
    datadict = {"FemaleHeadX": [2], "FemaleHeadY": [1],
                "FemaleAbdomenX": [1], "FemaleAbdomenY": [0],
                "MaleHeadX": [-1], "MaleHeadY": [2],
                "MaleAbdomenX": [0], "MaleAbdomenY": [1]}
    df = pd.DataFrame(datadict)
    angles = df.apply(mating_angle_from_body_axis_pd_df, axis=1)
    assert angles[0] == math.pi/2


def test_mating_angle_from_cos_pd_df():
    datadict = {"FemaleHeadX": [2], "FemaleHeadY": [1],
                "FemaleAbdomenX": [1], "FemaleAbdomenY": [0],
                "MaleHeadX": [-1], "MaleHeadY": [2],
                "MaleAbdomenX": [0], "MaleAbdomenY": [1]}
    df = pd.DataFrame(datadict)
    angles = df.apply(mating_angle_from_cos_pd_df, axis=1)
    assert angles[0] == math.pi/2


def test_filter_by_likelihood_body():
    testdata = load_csv_file(testdatapath)
    filtered_data, rows = filter_by_likelihood_body(testdata, 0.8)
    filtered_data_female_A = filtered_data[filtered_data.FemaleAbdomenP > 0.8]
    filtered_data_female_H = filtered_data[filtered_data.FemaleHeadP > 0.8]
    filtered_data_male_A = filtered_data[filtered_data.MaleAbdomenP > 0.8]
    filtered_data_male_H = filtered_data[filtered_data.MaleHeadP > 0.8]
    assert len(rows) == len(filtered_data)
    assert len(filtered_data_female_A) == len(filtered_data)
    assert len(filtered_data_female_H) == len(filtered_data)
    assert len(filtered_data_male_A) == len(filtered_data)
    assert len(filtered_data_male_H) == len(filtered_data)
    assert isinstance(rows, np.ndarray), "rows should be a numpy array"


def test_filtered_outputs():
    angles_b, wing_dist_male, abd_dist, head_dist, rownumbers =\
        unfiltered_outputs(testdatapath)
    angles_b_filter0, wing_dist_male_filter0, abd_dist_filter0, head_dist_filter0, rownumbers_filter0 =\
        filtered_outputs(testdatapath, 0)
    angles_b_filter08, wing_dist_male_filter08, abd_dist_filter08, head_dist_filter08, rownumbers_filter08 =\
        filtered_outputs(testdatapath, 0.8)

    assert len(angles_b) == len(angles_b_filter0), "filtering by P=0 should \
        not change data"
    assert angles_b[1] == angles_b_filter0[1], "filtering by P=0 should not \
        change data"
    assert angles_b[10000] == angles_b_filter0[10000], "filtering by P=0 \
        should not change data"
    assert angles_b[rownumbers_filter08[6000]] ==\
        angles_b_filter08[rownumbers_filter08[6000]], "data at corresponding \
            indices should match"
    assert len(rownumbers_filter08) ==\
        len(wing_dist_male_filter08), "length of row numbers should match \
            length of data"
    assert len(angles_b) !=\
        len(angles_b_filter08), "filtering by P=0.8 should change data"
    assert rownumbers_filter08[6000] !=\
        6000, "there are rownumbers missing in the filtered dataset"
    assert rownumbers_filter08[0] ==\
        0, "rownumbers should be 0 based"
    assert isinstance(
        rownumbers, np.ndarray), "rownumbers should be a numpy array"
    assert isinstance(rownumbers_filter08,
                      np.ndarray), "rownumbers should be a numpy array"


def test_removeWall_option():
    angles_b, wing_dist_male, abd_dist, head_dist, rownumbers =\
        unfiltered_outputs(testdatapath)
    angles_b_r, wing_dist_male_r, abd_dist_R, head_dist_r, rownumbers_r =\
        unfiltered_outputs(testdatapath, removeWall=True, minWallDist=50)
    assert len(angles_b) !=\
        len(angles_b_r), "filtering by distance to wall should remove data"
    assert len(rownumbers_r) > 0
    assert len(rownumbers) == 0
    assert len(rownumbers_r) == len(angles_b_r)
    assert len(angles_b_r) != len(angles_b)
    assert 0 in rownumbers_r
    assert 12231 not in rownumbers_r
    assert 6000 in rownumbers_r
    assert isinstance(
        rownumbers, np.ndarray), "rownumbers should be a numpy array"
    assert isinstance(
        rownumbers_r, np.ndarray), "rownumbers_r should be a numpy array"


def test_centroids():
    data = load_csv_file(testdatapath)
    centroidx, centroidy, d = centroids(data)
    assert d > 550
    assert d < 600
    assert centroidx > 300
    assert centroidy > 300
    assert centroidx < 400
    assert centroidy < 400


def test_centroid_distance():
    data = load_csv_file(testdatapath)
    centroidx, centroidy, d = centroids(data)

    distanceToCentroid = data.apply(lambda df: centroid_distance(df, centroidx,
                                    centroidy), axis=1)
    rownumbers = np.where(distanceToCentroid < ((d/2)-50))[0]
    assert distanceToCentroid[13673] > 250, "frame 4257 should have a larger \
        than 200 px distance from the centroid"
    assert distanceToCentroid[13673] > ((d/2)-50), "frame 4257 should have a \
        larger than 200 px distance from the centroid"
    assert distanceToCentroid[8815] < ((d/2)-50), "frame 4257 should have a \
        larger than 200 px distance from the centroid"
    assert d/2 < 300, "the radius of the chamber should be smaller than 300 px"
    assert 12231 not in rownumbers, "frame 4257 should not be in rownumbers"
    assert 8815 in rownumbers, "frame 19168 should  be in rownumbers"
    assert isinstance(rownumbers, np.ndarray), "rows should be numpy array"


def test_tilting_index():
    malewingdist = np.array([2, 2, 2, 1, 1, 1])
    malewingdist.reshape(1, 6)
    copstartframe = 3
    tilting_ind = tilting_index(malewingdist, copstartframe)
    assert tilting_ind[0] == 0.5
    assert tilting_ind[1] == 0.5


def test_tilting_index_with0():
    malewingdist = np.array([2, 2, 2, 2, 2, 2])
    malewingdist.reshape(1, 6)
    copstartframe = 0
    tilting_ind = tilting_index(malewingdist, copstartframe)
    assert tilting_ind[0] == 1


def test_tilting_index_all_frames():
    malewingdist = np.array([1, 1, 1, 2, 2, 2])
    malewingdist.reshape(1, 6)
    copstartframe = 3
    tilting_ind = tilting_index_all_frames(malewingdist, copstartframe)
    assert tilting_ind[0] == 1


def test_tilting_index_all_frames_df():
    angles_b, wing_dist_male, abd_dist, head_dist, rownumbers =\
        unfiltered_outputs(testdatapath)
    copstartframe = 3
    tilting_ind = tilting_index_all_frames(wing_dist_male, copstartframe)
    assert round(tilting_ind[1]) == 1
    assert isinstance(rownumbers, np.ndarray), "rownumbers should be np.array"


def test_tilting_index_all_frames_df_copstart0():
    angles_b, wing_dist_male, abd_dist, head_dist, rownumbers =\
        unfiltered_outputs(testdatapath)
    copstartframe = 0
    tilting_ind = tilting_index_all_frames(wing_dist_male, copstartframe)
    assert round(tilting_ind[1]) == 1
    assert isinstance(rownumbers, np.ndarray), "rownumbers should be np.array"


def test_tilting_index_df_copstart0():
    angles_b, wing_dist_male, abd_dist, head_dist, rownumbers =\
        unfiltered_outputs(testdatapath)
    copstartframe = 0
    tilting_ind = tilting_index(wing_dist_male, copstartframe)
    assert round(tilting_ind[1]) == 1


def test_npwhere():
    nparray = np.array([0, 1, 2, 3, 4, 5])
    above3 = np.where(nparray > 3)[0]
    assert 4 in above3
    assert 2 not in above3


def test_scale_unfiltered():
    angles_b_scaled, _, _, _, _ = scale_unfiltered(testdatapath)
    assert all(angles_b_scaled >= 0), "data should be between 0 and 1"
    assert all(angles_b_scaled <= 1), "data should be between 0 and 1"


def test_scale_filtered():
    angles_b_scaled, _, _, _, _ = scale_filtered(testdatapath, 0.8)
    assert all(angles_b_scaled >= 0), "data should be between 0 and 1"
    assert all(angles_b_scaled <= 1), "data should be between 0 and 1"


def test_scale_unfiltered_content():
    angles_b, _, _, _, rows = unfiltered_outputs(testdatapath)
    angles_b_scaled, _, _, _, _ = scale_unfiltered(testdatapath)
    assert ((angles_b_scaled[1, 0] > angles_b_scaled[100, 0]) ==
            (angles_b[1] > angles_b[100])
            ), "relations between data should be preserved after scaling"
    assert len(angles_b) == len(angles_b_scaled)
    assert angles_b.shape == (19681,)
    assert angles_b_scaled.shape == (19681, 1)
    assert angles_b[1] > 0
    assert angles_b_scaled[1, 0] > 0
    assert angles_b_scaled[100, 0] > 0
    assert isinstance(rows, np.ndarray), "rows should be numpy array"


def test_prepare_training_data():
    features1 = ["angles_b_scaled",
                 "head_dist_scaled",
                 "abd_dist_scaled",
                 "tilting_index_scaled"]
    features2 = ["angles_b_scaled",
                 "head_dist_scaled",
                 "abd_dist_scaled"]
    X1 = prepare_training_data(testdatapath, featurelist=features1)
    X2 = prepare_training_data(testdatapath, featurelist=features2)
    assert X1.shape == (19681, 4)
    assert X2.shape == (19681, 3)


def test_prepare_training_data_non_existent_feature():
    features1 = ["feature_does_not_exist"]
    with pytest.raises(NameError):
        _ = prepare_training_data(testdatapath, featurelist=features1)
