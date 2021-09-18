import pandas as pd
import numpy as np
import math
import scipy


def load_csv_file(path):
    """loads a csv file of deeplabcut data
    and returns a pandas dataframe"""
    datatable = pd.read_csv(path, header=1)
    FemaleHeadX = datatable['FemaleHead'][1:]
    FemaleHeadY = datatable['FemaleHead.1'][1:]
    FemaleHeadP = datatable['FemaleHead.2'][1:]
    MaleHeadX = datatable['MaleHead'][1:]
    MaleHeadY = datatable['MaleHead.1'][1:]
    MaleHeadP = datatable['MaleHead.2'][1:]
    MaleLeftShoulderX = datatable['MaleLeftShoulder'][1:]
    MaleLeftShoulderY = datatable['MaleLeftShoulder.1'][1:]
    MaleLeftShoulderP = datatable['MaleLeftShoulder.2'][1:]
    MaleRightShoulderX = datatable['MaleRightShoulder'][1:]
    MaleRightShoulderY = datatable['MaleRightShoulder.1'][1:]
    MaleRightShoulderP = datatable['MaleRightShoulder.2'][1:]
    MaleAbdomenX = datatable['MaleAbdomen'][1:]
    MaleAbdomenY = datatable['MaleAbdomen.1'][1:]
    MaleAbdomenP = datatable['MaleAbdomen.2'][1:]
    FemaleAbdomenX = datatable['FemaleAbdomen'][1:]
    FemaleAbdomenY = datatable['FemaleAbdomen.1'][1:]
    FemaleAbdomenP = datatable['FemaleAbdomen.2'][1:]
    localvars = locals()
    del localvars["path"]
    del localvars["datatable"]
    df = pd.DataFrame(data=localvars)
    dffloat = df.astype('float32')
    return dffloat


def angle_from_cos(FemaleHeadX,
                   FemaleHeadY,
                   FemaleAbdomenX, FemaleAbdomenY,
                   MaleHeadX, MaleHeadY,
                   MaleAbdomenX, MaleAbdomenY):
    """uses vector algebra to calculate angles"""
    vector_f = [(FemaleHeadX - FemaleAbdomenX), (FemaleHeadY - FemaleAbdomenY)]
    vector_m = [(MaleHeadX - MaleAbdomenX), (MaleHeadY - MaleAbdomenY)]
    unit_vector_f = vector_f / np.linalg.norm(vector_f)
    unit_vector_m = vector_m / np.linalg.norm(vector_m)
    dot_product = np.dot(unit_vector_f, unit_vector_m)
    if dot_product > 1:
        print("warning: vector dot product was {:.2f}".format(dot_product))
        dot_product = math.floor(dot_product)
    elif dot_product < -1:
        print("warning: vector dot product was {:.2f}".format(dot_product))
        dot_product = math.ceil(dot_product)
    angle = np.arccos(dot_product)

    return angle


def signDeltax(deltaxFly, deltayFly):
    """determines the sign of the fly's orientation -
    necessary for determining
if they face in the same direction"""
    if deltaxFly > 0:
        signDeltaxFly = 1
    elif deltaxFly < 0:
        signDeltaxFly = -1
    else:
        if deltayFly < 1:
            signDeltaxFly = 1
        else:
            signDeltaxFly = -1
    return signDeltaxFly


def signDeltax2(deltaxFly):
    """slight variation of signDeltax"""
    if deltaxFly > 0:
        signDeltaxFly = 1
    elif deltaxFly < 0:
        signDeltaxFly = -1
    else:
        signDeltaxFly = 0
    return signDeltaxFly


def mating_angle_from_angles(maleAngle, femaleAngle, relativeSign):
    """calculates the mating angles based on the male and female angle
and the relative sign of their facing direction"""
    if relativeSign > 0:
        matingAngle = abs(maleAngle - femaleAngle)
    else:
        matingAngle = math.pi - abs(maleAngle - femaleAngle)
    return matingAngle


def mating_angle_from_body_axis(FemaleHeadX, FemaleHeadY,
                                FemaleAbdomenX, FemaleAbdomenY,
                                MaleHeadX, MaleHeadY,
                                MaleAbdomenX, MaleAbdomenY):
    """determines mating angle from the head and abdomen position data"""
    deltaxF = FemaleHeadX - FemaleAbdomenX
    deltayF = FemaleHeadY - FemaleAbdomenY
    deltaxM = MaleHeadX - MaleAbdomenX
    deltayM = MaleHeadY - MaleAbdomenY

    femaleAngle = np.arctan(deltayF/deltaxF)
    maleAngle = np.arctan(deltayM/deltaxM)

    signDeltaxF = signDeltax2(deltaxF)
    if signDeltaxF == 0:
        signDeltaxF = signDeltax2(deltayF)
        signDeltaxM = signDeltax2(deltayM)
        femaleAngle = signDeltaxF * math.pi/2
    else:
        signDeltaxM = signDeltax2(deltaxM)
        if signDeltaxM == 0:
            signDeltaxF = signDeltax2(deltayF)
            signDeltaxM = signDeltax2(deltayM)
            maleAngle = signDeltaxM * math.pi/2
    relativeSign = signDeltaxF * signDeltaxM
    matingAngle = mating_angle_from_angles(maleAngle,
                                           femaleAngle,
                                           relativeSign)
    return matingAngle


def mating_angle_from_body_axis_pd_df(df):
    """applies the mating_angle_from_body_axis function
    to a row in a pandas dataframe"""
    matingAngleRow = mating_angle_from_body_axis(df.FemaleHeadX,
                                                 df.FemaleHeadY,
                                                 df.FemaleAbdomenX,
                                                 df.FemaleAbdomenY,
                                                 df.MaleHeadX,
                                                 df.MaleHeadY,
                                                 df.MaleAbdomenX,
                                                 df.MaleAbdomenY)
    return matingAngleRow


def mating_angles_all_rows_from_body_axis(path):
    """applies the mating_angle_from_body_axis function
    to all rows in a dataframe"""
    data = load_csv_file(path)
    angles = data.apply(mating_angle_from_body_axis_pd_df, axis=1)
    return angles


def mating_angle_from_cos_pd_df(df):
    """applies the mating_angle_from_cos function
    to a row in a pandas dataframe"""
    matingAngleRow = angle_from_cos(df.FemaleHeadX,
                                    df.FemaleHeadY,
                                    df.FemaleAbdomenX,
                                    df.FemaleAbdomenY,
                                    df.MaleHeadX,
                                    df.MaleHeadY,
                                    df.MaleAbdomenX,
                                    df.MaleAbdomenY)
    return matingAngleRow


def mating_angles_all_rows_from_cos(path):
    """applies the mating_angle_from_body_axis function
    to all rows in a dataframe"""
    data = load_csv_file(path)
    angles = data.apply(mating_angle_from_cos_pd_df, axis=1)
    return angles


def filter_by_likelihood_body(data, P):
    """filteres the data (in pandas dataframe format)
    by those where body axis parameters have a 
    likelihood >P """
    isLargeHeadP = data.MaleHeadP > P
    isLargeAbdP = data.MaleAbdomenP > P
    isLargeHeadFP = data.FemaleHeadP > P
    isLargeAbdFP = data.FemaleAbdomenP > P
    data_filtered = data[isLargeHeadP & isLargeAbdP & isLargeHeadFP
                         & isLargeAbdFP]
    rownumbers = np.where(isLargeHeadP & isLargeAbdP & isLargeHeadFP
                          & isLargeAbdFP)[0]
    return data_filtered, rownumbers


def filtered_mating_angles(path, P):
    """loads the csv file of deeplabcut data
    specified as the path argument and determines mating angle
    from both wing and body axis data;
    returns the angles based on wing data and the angles based on body axis
    (in this order)
    This is the function that should be used if you want filtering of data by 
    those with a likelihood > P"""
    data = load_csv_file(path)
    dataF, rownumbers = filter_by_likelihood_body(data, P)
    angles_b = dataF.apply(mating_angle_from_cos_pd_df, axis=1)
    return angles_b, rownumbers


def wing_distance_male(df):
    """calculates distance between wings"""
    wing_distance_male = wing_distance(df.MaleLeftShoulderX,
                                       df.MaleLeftShoulderY,
                                       df.MaleRightShoulderX,
                                       df.MaleRightShoulderY)
    return wing_distance_male


def wing_distance(wing1x, wing1y, wing2x, wing2y):
    distance = math.sqrt((wing1x-wing2x)**2+(wing1y-wing2y)**2)
    return distance


def wing_distance_all_rows(path):
    """loads the csv file of deeplabcut data
    specified as the path argument and determines wing distance
   This is the function that should be used if you want no filtering of data"""
    data = load_csv_file(path)
    wing_dist_male = data.apply(wing_distance_male, axis=1)
    return wing_dist_male


def filtered_wing_distance(path, P):
    """loads the csv file of deeplabcut data
    specified as the path argument and determines wing distance;
    This is the function that should be used if you want filtering of data by 
    those with a likelihood > P"""
    data = load_csv_file(path)
    dataF, rownumbers = filter_by_likelihood_body(data, P)
    wing_dist_male = dataF.apply(wing_distance_male, axis=1)
    return wing_dist_male, rownumbers


def filtered_outputs(path, P, removeWall=False, minWallDist=3):
    """loads the csv file of deeplabcut data
    specified as the path argument and determines mating angle
    from both wing and body axis data as well as wing distance;
    This is the function that should be used if you want filtering of data by 
    those with a likelihood > P"""
    data = load_csv_file(path)
    dataF, rownumbers = filter_by_likelihood_body(data, P)
    centroidx, centroidy, d = centroids(data)
    distanceToCentroid = dataF.apply(lambda df:
                                     centroid_distance(
                                         df, centroidx, centroidy), axis=1)
    if removeWall:
        dataF = dataF[distanceToCentroid < ((d/2)-minWallDist)]
        rownumbers = rownumbers[distanceToCentroid < (
            (d/2)-minWallDist)]  # check if this is correct
    angles_b = dataF.apply(mating_angle_from_cos_pd_df, axis=1)
    wing_dist_male = dataF.apply(wing_distance_male, axis=1)
    abd_dist = dataF.apply(abd_distance, axis=1)
    head_dist = dataF.apply(head_distance, axis=1)
    return angles_b, wing_dist_male, abd_dist, head_dist, rownumbers


def unfiltered_outputs(path, removeWall=False, minWallDist=3):
    """loads the csv file of deeplabcut data
    specified as the path argument and determines mating angle
    from both wing and body axis data as well as wing distance;
    This is the function that should be used if you don't want filtering of data by 
    those with a likelihood > P"""
    rownumbers = np.array([])
    data = load_csv_file(path)
    centroidx, centroidy, d = centroids(data)
    distanceToCentroid = data.apply(lambda df:
                                    centroid_distance(
                                        df,
                                        centroidx,
                                        centroidy),
                                    axis=1)
    if removeWall:
        data = data[distanceToCentroid < ((d/2)-minWallDist)]
        rownumbers = np.where(distanceToCentroid < ((d/2)-minWallDist))[0]
    angles_b = data.apply(mating_angle_from_cos_pd_df, axis=1)
    wing_dist_male = data.apply(wing_distance_male, axis=1)
    abd_dist = data.apply(abd_distance, axis=1)
    head_dist = data.apply(head_distance, axis=1)
    return angles_b, wing_dist_male, abd_dist, head_dist, rownumbers


def centroids(data):
    maxx = max(np.concatenate([data.FemaleHeadX, data.FemaleAbdomenX,
                               data.MaleHeadX, data.MaleAbdomenX]))
    minx = min(np.concatenate([data.FemaleHeadX, data.FemaleAbdomenX,
                               data.MaleHeadX, data.MaleAbdomenX]))
    dx = maxx-minx
    centroidx = minx+(dx/2)
    maxy = max(np.concatenate([data.FemaleHeadY, data.FemaleAbdomenY,
                               data.MaleHeadY, data.MaleAbdomenY]))
    miny = min(np.concatenate([data.FemaleHeadY, data.FemaleAbdomenY,
                               data.MaleHeadY, data.MaleAbdomenY]))
    dy = maxy-miny
    centroidy = miny+(dy/2)
    d = max([dx, dy])
    return centroidx, centroidy, d


def tilting_row(df):
    """calculates tilting index for one row"""
    tilting = df.male_wingdist
    return tilting


def tilting_index(malewingdist, copstartframe, rownumbers=np.array([])):
    """applies tilting_row function to the dataframe,
    taking all frames before copstartframe as baseline"""
    if rownumbers.size > 0:
        copstartframe = len([frame for frame in range(copstartframe)
                             if frame in rownumbers])
    male_resting = np.median(malewingdist[1:copstartframe-1])
    tilting = malewingdist[copstartframe:]
    tilting_ind = tilting/male_resting
    return tilting_ind


def tilting_index_all_frames(malewingdist,
                             copstartframe,
                             rownumbers=np.array([])):
    """applies tilting_row function to the dataframe,
    taking all frames before copstartframe as baseline"""
    if rownumbers.size > 0:
        copstartframe = len([frame for frame in range(copstartframe)
                             if frame in rownumbers])
    male_resting = np.median(malewingdist[1:copstartframe-1])
    tilting_ind = malewingdist/male_resting
    return tilting_ind


def abd_distance(df):
    """calculates abdominal distance for one row in dataframe"""
    distanceMF = distance(df.MaleAbdomenX, df.MaleAbdomenY,
                          df.FemaleAbdomenX, df.FemaleAbdomenY)
    return distanceMF


def head_distance(df):
    """calculates head distance for one row in dataframe"""
    distanceMF = distance(df.MaleHeadX, df.MaleHeadY,
                          df.FemaleHeadX, df.FemaleHeadY)
    return distanceMF


def centroid_distance(df, centroidx, centroidy):
    """calculates male abdomen to centroid distance for one row in dataframe"""
    distancec = distance(df.MaleAbdomenX, df.MaleAbdomenY,
                         centroidx, centroidy)
    return distancec


def distance(xmale, ymale, xfemale, yfemale):
    """calculates a distance between two points,
    each of which are entered as their x and y coordinates"""
    distance = math.sqrt((xmale-xfemale)**2+(ymale-yfemale)**2)
    return distance


def abd_distance_all_rows(path):
    """loads the csv file of deeplabcut data
    specified as the path argument and determines wing distance
   calculates distance between male and female abdomen"""
    data = load_csv_file(path)
    dist = data.apply(abd_distance, axis=1)
    return dist


def load_feat_file(path):
    """loads the angle_between data from the feat.mat file.
    returns feat.data and angle_between as 2 variables"""
    feat = scipy.io.loadmat(path, matlab_compatible=True)
    dataArr = feat['feat']['data'][0][0]
    angle_between = dataArr[:, :, 10]
    return feat, angle_between
