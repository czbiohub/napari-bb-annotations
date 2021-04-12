from collections import OrderedDict
import os

from scipy.spatial import distance as dist
import numpy as np
import pandas as pd


class CentroidTracker():
    def __init__(self, max_disappeared=30):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextobject_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextobject_id] = centroid
        self.disappeared[self.nextobject_id] = 0
        self.nextobject_id += 1

    def deregister(self, object_id):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        # check to see if the list of input objects is empty
        number_of_objects = len(input_centroids)
        if number_of_objects == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for object_id in self.disappeared.keys():
                self.disappeared[object_id] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        if len(self.objects) == 0:
            for i in range(0, number_of_objects):
                self.register(input_centroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            paired_distance = dist.cdist(
                np.array(object_centroids), input_centroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = paired_distance.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = paired_distance.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            used_rows = set()
            used_cols = set()

            assert len(rows) == len(cols)

            # loop over the combination of the (row, column) index
            # tuples

            for (row, col) in zip(rows, cols):

                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in used_rows or col in used_cols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                used_rows.add(row)
                used_cols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unused_rows = set(
                range(0, paired_distance.shape[0])).difference(used_rows)
            unused_cols = set(
                range(0, paired_distance.shape[1])).difference(used_cols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            # loop over the unused row indexes
            for row in unused_rows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                # check to see if the number of consecutive
                # frames the object has been marked "disappeared"
                # for warrants deregistering the object
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            for col in unused_cols:
                self.register(input_centroids[col])
        self.object_ids = object_id
        # return the set of trackable objects
        return self.objects


def dist_between_centroids(centroid1, centroid2):
    """
    Returns euclidean distance between two centroids or cartesean coordinates
    """
    vect = np.array(
        centroid1, dtype=np.float64) - np.array(
        centroid2, dtype=np.float64)
    return np.linalg.norm(vect)


def update_trajectory_path_graph(
        current_cell_ids, trajectory_path_graph, frame):
    """
    Returns updated trajectory path graph.
    This function loops through the current_cell_ids which contain
    the cell id and centroid.
    If it doesn't exist, it will initialize them with zeros.
    For cells appearing in not exactly the first frame -
    updates them as with the 'frame' specified as beginning frame
    For other cells updates the trajectory path distances displacement -
    based on the previous trajectory path graph dictionary
    """
    previous_cell_ids = list(trajectory_path_graph.keys())

    # if new cells are formed add more empty rows for tracking
    for current_cell_id, centroid in current_cell_ids.items():
        # initialize trajectory path graph keys only for new cell ids
        if current_cell_id not in previous_cell_ids:
            trajectory_path_graph[current_cell_id] = dict(
                beginning_frame=0,
                ending_frame=0,
                parent_cell=0,
                distance=0,
                starting_centroid=centroid,
                ending_centroid=centroid,
                displacement=0)

    for current_cell_id, centroid in current_cell_ids.items():
        # cell already initialized from a previous frame
        if current_cell_id in previous_cell_ids:
            trajectory_path_graph[current_cell_id]["ending_frame"] = \
                trajectory_path_graph[current_cell_id]["ending_frame"] + 1

        # new cell or a division from parent cell
        else:
            trajectory_path_graph[current_cell_id]["beginning_frame"] = frame
            trajectory_path_graph[current_cell_id]["ending_frame"] = frame + 1

        trajectory_path_graph[current_cell_id]["parent_cell"] = 0
        previous_centroid = trajectory_path_graph[
            current_cell_id]["ending_centroid"]
        trajectory_path_graph[current_cell_id]["ending_centroid"] = centroid
        trajectory_path_graph[current_cell_id]["distance"] = \
            trajectory_path_graph[current_cell_id]["distance"] + \
            dist_between_centroids(centroid, previous_centroid)
        trajectory_path_graph[current_cell_id]["displacement"] = \
            dist_between_centroids(
                centroid,
                trajectory_path_graph[current_cell_id]["starting_centroid"])

    return trajectory_path_graph


def df_centroid_tracking_rectangles(df, max_disappeared, all_files):
    # Calculate speed  between frames euclidian distance between centroids
    # divided by time
    # initialize our centroid tracker and binary_image dimensions
    centroid_tracker = CentroidTracker(max_disappeared)

    # initialize trajectory path graph
    trajectory_path_graph = {}
    unique_cell_ids = [0] * len(df)
    # loop over the images in the binary annotations folder
    for frame_count, image_path in enumerate(all_files):
        centroids = []
        cell_indices = []
        for index, row in df.iterrows():
            if row.image_id == image_path:
                centroid = tuple((
                    (row.xmin + row.xmax) // 2, (row.ymin + row.ymax) // 2))
                centroids.append(centroid)
                cell_indices.append(index)
        # update our centroid tracker using the computed set of centroids
        current_cell_ids = centroid_tracker.update(centroids)
        for cell_idx in cell_indices:
            unique_cell_ids[cell_idx] + centroid_tracker.object_ids
        trajectory_path_graph = update_trajectory_path_graph(
            current_cell_ids,
            trajectory_path_graph,
            frame_count)
    # update unique cell ids column in dataframe "unique_cell_id"
    df['unique_cell_id'] = unique_cell_ids
    # Save the tracking path dataframe
    trajectory_df = pd.DataFrame(pd.DataFrame(trajectory_path_graph))
    trajectory_df.to_csv(
        os.path.join(
            os.path.dirname(all_files[0]) + "trajectory_path_graph.csv"))
    return df
