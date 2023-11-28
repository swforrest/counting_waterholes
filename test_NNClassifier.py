import pytest
import os
import NNclassifier
import yaml
import scipy
import numpy as np

"""
Unit Tests For NNClassifier.py
"""

class TestRemoveTempDirs():

    @pytest.fixture(autouse=True)
    def fixture(self):
        # remove the temp dirs if they exist
        if os.path.exists(NNclassifier.TEMP): os.rmdir(NNclassifier.TEMP)
        if os.path.exists(NNclassifier.TEMP_PNG): os.rmdir(NNclassifier.TEMP_PNG)
        yield
        # remove the temp dirs
        os.rmdir(NNclassifier.TEMP_PNG)
        os.rmdir(NNclassifier.TEMP)

    def test_prepare_temp_dirs_non_exist(self):
        NNclassifier.prepare_temp_dirs() 
        # ensure that the temp dirs now exist
        assert os.path.exists(NNclassifier.TEMP)
        assert os.path.exists(NNclassifier.TEMP_PNG)

    def test_prepare_temp_dirs_exist(self):
        # create the temp dirs
        os.mkdir(NNclassifier.TEMP)
        os.mkdir(NNclassifier.TEMP_PNG)
        NNclassifier.prepare_temp_dirs() 
        # ensure that the temp dirs now exist
        assert os.path.exists(NNclassifier.TEMP)
        assert os.path.exists(NNclassifier.TEMP_PNG)

class TestReadClassifications:
    @pytest.fixture(autouse=True)
    def fixture(self):
        self.path = os.path.join(os.getcwd(), "test_data", "test_classifications_2_12.txt")
        # create a sample classification file
        contents = """0 0.1 0.1 0.0020 0.0020 0.93\n0 0.6 0.4 0.0018 0.0021 0.23\n1 0.3 0.7 0.0030 0.0029 0.83"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write(contents)
        yield
        # remove the sample classification file
        if os.path.exists(self.path): os.remove(self.path)
        if os.path.exists(os.path.dirname(self.path)): os.rmdir(os.path.dirname(self.path))

    def test_file_info(self):
        # row 2, col 12 should have pixel values of 1248 across and 208 down 
        across, down, lines = NNclassifier.classification_file_info(self.path)
        assert across == 1248 
        assert down == 208
        assert len(lines) == 3

    def test_parse_classifications(self):
        classifications = NNclassifier.parse_classifications(self.path)
        assert type(classifications) == np.ndarray
        assert classifications.shape == (3, 6)
        # check the values are of the form [x, y, confidence, class, width, height]
        # x and y should be scaled for the full image (x * 416 + across), (y * 416 + down)
        # width and height should be scaled for the full image (width * 416), (height * 416)
        assert np.all(classifications[:, 0] == [1289.6, 1497.6, 1372.8])
        assert np.all(classifications[:, 1] == [249.6, 374.4, 499.2])
        assert np.all(classifications[:, 2] == [0.93, 0.23, 0.83])
        assert np.all(classifications[:, 3] == [0, 0, 1])
        assert np.all(classifications[:, 4] == [0.0020 * 416, 0.0018 * 416, 0.0030 * 416])
        assert np.all(classifications[:, 5] == [0.0020 * 416, 0.0021 * 416, 0.0029 * 416])

    def test_read_classifications(self):
        classifications, low_conf = NNclassifier.read_classifications(self.path, class_folder=os.path.dirname(self.path), confidence_threshold=0.5)
        assert type(classifications) == np.ndarray
        assert classifications.shape == (2, 6)
        assert type(low_conf) == np.ndarray
        assert low_conf.shape == (1, 6)
        # check the values are of the form [x, y, confidence, class, width, height, file_name]
        assert np.all(classifications[0, :] == [1289.6, 249.6, 0.93, 0, 0.0020 * 416, 0.0020 * 416])
        # check low confidence one is same
        assert np.all(low_conf[0, :] == [1497.6, 374.4, 0.23, 0, 0.0018 * 416, 0.0021 * 416])

class TestClustering:
    @pytest.fixture(autouse=True)
    def fixture(self):
        # make a sample classification array in pixel coordinates
        cutoff = 6
        self.classifications = np.array([   [1, 1, 0.93, 0, 2, 2],
                                            [4, 5, 0.91, 0, 2, 2],
                                            [9, 10, 0.65, 0, 2, 2]])
        self.no_clusters = np.array([   [1, 1, 0.93, 0, 2, 2],
                                            [22, 30, 0.91, 0, 2, 2],
                                            [9, 10, 0.65, 0, 2, 2]])
        self.one_point = np.array([   [1, 1, 0.93, 0, 2, 2]])

        self.clusters_w_files = np.array([   [1, 1, 0.93, 0, 2, 2, "test1.jpg"],
                                            [4, 5, 0.91, 0, 2, 2, "test2.jpg"]])
        yield

    def test_cluster(self):
        clusters = NNclassifier.cluster(self.classifications, cutoff=6)
        assert len(clusters) == 3                           # all 3 points returned
        assert set(np.unique(clusters[:, -1])) == {1, 2}    # 2 clusters 
        # get points in each cluster
        cluster1 = clusters[clusters[:, -1] == 1]
        cluster2 = clusters[clusters[:, -1] == 2]
        # cluster with 2 points should be close together
        assert np.all(scipy.spatial.distance.pdist(cluster1[:, :1], metric='euclidean') < 6)
        # cluster with 1 point should be far away
        assert np.all(scipy.spatial.distance.pdist(np.vstack((cluster1[0, :2],cluster2[0, :2])), metric='euclidean') > 6)

    def test_cluster_no_clusters(self):
        clusters = NNclassifier.cluster(self.no_clusters, cutoff=6)
        assert len(clusters) == 3                           # all 3 points returned
        assert set(np.unique(clusters[:,-1])) == {1, 2, 3}  # 3 clusters

    def test_cluster_one_point(self):
        clusters = NNclassifier.cluster(self.one_point, cutoff=6)
        assert len(clusters) == 1                    # 1 point returned
        assert set(np.unique(clusters[:,-1])) == {1} # in its own cluster

    def test_process_clusters(self):
        clusters = NNclassifier.cluster(self.classifications, cutoff=6)
        processed = NNclassifier.process_clusters(clusters)
        # should no longer have cluster numbers
        assert processed.shape == (2, 6)

    def test_process_clusters_no_clusters(self):
        clusters = NNclassifier.cluster(self.no_clusters, cutoff=6)
        processed = NNclassifier.process_clusters(clusters)
        # should no longer have cluster numbers, but should have all 3 points
        assert processed.shape == (3, 6)

    def test_process_clusters_one_point(self):
        clusters = NNclassifier.cluster(self.one_point, cutoff=6)
        processed = NNclassifier.process_clusters(clusters)
        # should get the point back
        assert processed.shape == (1, 6)

    def test_process_clusters_w_files(self):
        clusters = NNclassifier.cluster(self.clusters_w_files, cutoff=6)
        processed = NNclassifier.process_clusters(clusters)
        # should no longer have cluster numbers
        assert processed.shape == (1, 7)
        # should have both file names, space separated
        assert processed[0, -1] == "test1.jpg test2.jpg"








    
