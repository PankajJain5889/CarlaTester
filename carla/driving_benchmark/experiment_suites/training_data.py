# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# CORL experiment set.

from __future__ import print_function

from carla.driving_benchmark.experiment import Experiment
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite
import pandas as pd


class trainingData(ExperimentSuite):

    @property
    def train_weathers(self):
        return [1]#[8,3,6,1] 

    @property
    def test_weathers(self):
        return [0]

    def _poses_town01(self):
        """
        Each matrix is a new task. We have all the four tasks
        """ 
        '''def _poses_straight():
                                    return [[36, 40], [68, 50],[39, 35], [110, 114], [7, 3], [0, 4],
                                             [61, 59], [47, 64], [147, 90], [33, 87],
                                            [26, 19], [80, 76], [45, 49], [55, 44], [29, 107],
                                            [95, 104], [84, 34], [53, 67], [22, 17], [91, 148],
                                            [20, 107], [78, 70], [95, 102], [68, 44], [45, 69]]
                        
                                def _poses_one_curve():
                                    return [[138, 17], [47, 16], [26, 9], [42, 49], [140, 124],
                                            [85, 98], [65, 133], [137, 51], [76, 66], [46, 39],
                                            [40, 60], [0, 29], [4, 129], [121, 140], [2, 129],
                                            [78, 44], [68, 85], [41, 102], [95, 70], [68, 129],
                                            [84, 69], [47, 79], [110, 15], [130, 17], [0, 17]]
                        
                                def _poses_navigation():
                                    return [[102, 87],[105, 29], [27, 130],  [132, 27], [24, 44],
                                            [96, 26], [34, 67], [28, 1], [140, 134], [105, 9],
                                            [148, 129], [65, 18], [21, 16], [147, 97], [42, 51],
                                            [30, 41], [18, 107], [69, 45], [102, 95], [18, 145],
                                            [111, 64], [79, 45], [84, 69], [73, 31], [37, 81]]'''
        _poses1 = [ [29, 105], [130, 27], [87, 102], [27, 132], [44, 24],
              [26, 96], [67, 34], [1, 28], [134, 140], [9, 105],
              [129, 148], [16, 65], [16, 21], [97, 147], [51, 42],
              [41, 30], [107, 16], [47, 69], [95, 102], [145, 16],
              [64, 111], [47, 79], [69, 84], [31, 73], [81, 37],
              [57, 35], [116, 42], [47, 75], [143, 132], [8, 145],
              [107, 43], [111, 61], [105, 137], [72, 24], [77, 0],
              [80, 17], [32, 12], [64, 3], [32, 146], [40, 33],
              [127, 71], [116, 21], [49, 51], [110, 35], [85, 91],
              [114, 93], [30, 7], [110, 133], [60, 43], [11, 98], [96, 49], [90, 85],
              [27, 40], [37, 74], [97, 41], [110, 62], [19, 2], [138, 114], [131, 76],
              [116, 95], [50, 71], [15, 97], [50, 133],
              [23, 116], [38, 116], [101, 52], [5, 108], [23, 79], [13, 68]]
        _poses2 = [[19, 66], [79, 14], [19, 57], [39, 53], [60, 26],
             [53, 76], [42, 13], [31, 71], [59, 35], [47, 16],
             [10, 61], [66, 3], [20, 79], [14, 56], [26, 69],
             [79, 19], [2, 29], [16, 14], [5, 57], [77, 68],
             [70, 73], [46, 67], [57, 50], [61, 49], [21, 12]]
        _poses3=[[71, 127], [21, 116],  [51, 49], [35, 110], [91, 85],
             [93, 114], [7, 30], [133, 110], [43, 60], [98, 11], [49, 96], [85, 90],
             [40, 27], [74, 37], [41, 97], [62, 110], [2, 19], [114, 138], [76, 131],
             [95, 116], [71, 50], [97, 15], [74, 71], [133, 50],
             [116, 23], [116, 38], [52, 101], [108, 5], [79, 23], [68, 13]]
        return [_poses1, 
                _poses2, 
                _poses3]
    '''def _poses_town02(self):
            
                    def _poses_straight():
                        return [[38, 34], [4, 2], [12, 10], [62, 55], [43, 47],
                                [64, 66], [78, 76], [59, 57], [61, 18], [35, 39],
                                [12, 8], [0, 18], [75, 68], [54, 60], [45, 49],
                                [46, 42], [53, 46], [80, 29], [65, 63], [0, 81],
                                [54, 63], [51, 42], [16, 19], [17, 26], [77, 68]]
            
                    def _poses_one_curve():
                        return [[37, 76], [8, 24], [60, 69], [38, 10], [21, 1],
                                [58, 71], [74, 32], [44, 0], [71, 16], [14, 24],
                                [34, 11], [43, 14], [75, 16], [80, 21], [3, 23],
                                [75, 59], [50, 47], [11, 19], [77, 34], [79, 25],
                                [40, 63], [58, 76], [79, 55], [16, 61], [27, 11]]
            
                    def _poses_navigation():
                        return [[19, 66], [79, 14], [19, 57], [23, 1],
                                [53, 76], [42, 13], [31, 71], [33, 5],
                                [54, 30], [10, 61], [66, 3], [27, 12],
                                [79, 19], [2, 29], [16, 14], [5, 57],
                                [70, 73], [46, 67], [57, 50], [61, 49], [21, 12],
                                [51, 81], [77, 68], [56, 65], [43, 54]]
            
                    return [
                            _poses_one_curve(),
                            _poses_navigation(),
                            _poses_straight(),
                            ]'''

    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.
        """

        # We set the camera
        # This single RGB camera is used on every experiment

        camera = Camera('CameraRGB')
        camera.set(FOV=100)
        camera.set_image_size(800, 600)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-15.0, 0, 0)

        if self._city_name == 'Town01':
            poses_tasks = self._poses_town01()
            vehicles_tasks = [0,25]
            pedestrians_tasks = [0,25]
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [0,25]
            pedestrians_tasks = [0, 25]

        experiments_vector = []
        #weathers = [1,3,6,8]
        for weather in self.weathers:
            #prinst(weather , vehicles_tasks)
            for scenario in range(len(vehicles_tasks)):
                for iteration in range(len(poses_tasks)):
                    #print(f"interation : {iteration} , scenario:{scenario}")
                    poses = poses_tasks[iteration]
                    #print("poses re",poses)
                    vehicles = vehicles_tasks[scenario]
                    #print("Vehicles: ",vehicles)
                    pedestrians = pedestrians_tasks[scenario]
                    #print("pedestrians: ",pedestrians)

                    conditions = CarlaSettings()
                    conditions.set(
                        SendNonPlayerAgentsInfo=True,
                        NumberOfVehicles=vehicles,
                        NumberOfPedestrians=pedestrians,
                        WeatherId=weather
                    )
                    # Add all the cameras that were set for this experiments

                    conditions.add_sensor(camera)

                    experiment = Experiment()
                    experiment.set(
                        Conditions=conditions,
                        Poses=poses,
                        Task=iteration,
                        Repetitions=1
                    )
                    experiments_vector.append(experiment)

        return experiments_vector
