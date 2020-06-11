# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import abc
import logging
import math
import time
import pandas as pd
from collections import OrderedDict
from carla.client import VehicleControl
from carla.client import make_carla_client
from carla.driving_benchmark.metrics import Metrics
from carla.planner.planner import Planner
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

from . import results_printer
from .recording import Recording


def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


class DrivingBenchmark(object):
    """
    The Benchmark class, controls the execution of the benchmark interfacing
    an Agent class with a set Suite.


    The benchmark class must be inherited with a class that defines the
    all the experiments to be run by the agent
    """

    def __init__(
            self,
            city_name='Town01',
            name_to_save='Test',
            continue_experiment=False,
            save_images=False,
            distance_for_success=2.0    
    ):

        self.__metaclass__ = abc.ABCMeta

        self._city_name = city_name
        self._base_name = name_to_save
        # The minimum distance for arriving into the goal point in
        # order to consider ir a success
        self._distance_for_success = distance_for_success
        # The object used to record the benchmark and to able to continue after
        self._recording = Recording(name_to_save=name_to_save,
                                    continue_experiment=continue_experiment,
                                    save_images=save_images
                                    )
        

        # We have a default planner instantiated that produces high level commands
        self._planner = Planner(city_name)

    def benchmark_agent(self, experiment_suite, agent, client):
        """
        Function to benchmark the agent.
        It first check the log file for this benchmark.
        if it exist it continues from the experiment where it stopped.


        Args:
            experiment_suite
            agent: an agent object with the run step class implemented.
            client:


        Return:
            A dictionary with all the metrics computed from the
            agent running the set of experiments.
        """

        # Instantiate a metric object that will be used to compute the metrics for
        # the benchmark afterwards.
        data = OrderedDict()
        metrics_object = Metrics(experiment_suite.metrics_parameters,
                                 experiment_suite.dynamic_tasks)

        # Function return the current pose and task for this benchmark.
        start_pose, start_experiment = self._recording.get_pose_and_experiment(
            experiment_suite.get_number_of_poses_task())
        


        logging.info('START')
        cols = ['Start_position' , 'End_position' , 'Distance', 'Time' ,"Directions", "Avg-Speed" ,"Collision-Pedestrian" , "Collision-Traffic", "Collision-Other", 
                 "Intersection-Lane" , "Intersection-Offroad" , "Traffic_Light_Infraction", "Success",'Success without Collision', "Success without Lane-Infraction"]
        df  = pd.DataFrame(columns=cols)
        ts = str(int(time.time()))
                
        for experiment in experiment_suite.get_experiments()[int(start_experiment):]:

            positions = client.load_settings(
                experiment.conditions).player_start_spots

            self._recording.log_start(experiment.task)

            for pose in experiment.poses[start_pose:]:
                data['Start_position'] = pose[0]
                data['End_position'] = pose[1]
                
                #print(df)
                
                for rep in range(experiment.repetitions):

                    start_index = pose[0]
                    end_index = pose[1]
                    
                    client.start_episode(start_index)
                    # Print information on
                    logging.info('======== !!!! ==========')
                    logging.info(' Start Position %d End Position %d ',
                                 start_index, end_index)
                    
                    

                    

                    self._recording.log_poses(start_index, end_index,
                                              experiment.Conditions.WeatherId)

                    # Calculate the initial distance for this episode
                    initial_distance = \
                        sldist(
                            [positions[start_index].location.x, positions[start_index].location.y],
                            [positions[end_index].location.x, positions[end_index].location.y])
                    data["Distance"] = initial_distance
                    time_out = experiment_suite.calculate_time_out(
                        self._get_shortest_path(positions[start_index], positions[end_index]))

                    # running the agent
                    (result, reward_vec, control_vec, final_time, remaining_distance, data) = \
                        self._run_navigation_episode(
                            agent, client, time_out, positions[end_index],
                            str(experiment.Conditions.WeatherId) + '_'
                            + str(experiment.task) + '_' + str(start_index)
                            + '.' + str(end_index) , data)
                    data["Time"] = final_time    
                    # Write the general status of the just ran episode
                    self._recording.write_summary_results(
                        experiment, pose, rep, initial_distance,
                        remaining_distance, final_time, time_out, result)

                    # Write the details of this episode.
                    self._recording.write_measurements_results(experiment, rep, pose, reward_vec,
                                                               control_vec)
                    data['Success'] = result
                    if result > 0:
                        logging.info('+++++ Target achieved in %f seconds! +++++',
                                     final_time)

                    else:
                        logging.info('----- Timeout! -----')
                #data['Start_position'] = start_index
                #data['End_position'] = end_index
                #df = df.append(data , ignore_index=True)
                #print(df)
                data["Avg-Speed"] = 3.6 *(data['Distance'] / data['Time'])
                df = df.append(data , ignore_index=True)
                df  = df[cols]
                try: 
                    df.to_csv("Test_result_"+ts + '.csv' ,columns = cols ,index= True)
                except:
                    print("File being used will save later")
            start_pose = 0
            
        self._recording.log_end()

        return metrics_object.compute(self._recording.path)

    def get_path(self):
        """
        Returns the path were the log was saved.
        """
        return self._recording.path

    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self._planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def _get_shortest_path(self, start_point, end_point):
        """
        Calculates the shortest path between two points considering the road netowrk
        """

        return self._planner.get_shortest_path_distance(
            [
                start_point.location.x, start_point.location.y, 0.22], [
                start_point.orientation.x, start_point.orientation.y, 0.22], [
                end_point.location.x, end_point.location.y, end_point.location.z], [
                end_point.orientation.x, end_point.orientation.y, end_point.orientation.z])

    def _run_navigation_episode(
            self,
            agent,
            client,
            time_out,
            target,
            episode_name,
            data):
        """
         Run one episode of the benchmark (Pose) for a certain agent.


        Args:
            agent: the agent object
            client: an object of the carla client to communicate
            with the CARLA simulator
            time_out: the time limit to complete this episode
            target: the target to reach
            episode_name: The name for saving images of this episode

        """

        # Send an initial command.
        direction2text = {0: 'Follow_lane',1: 'Follow_lane',2: 'Follow_lane',
        3:'Left' , 4:'Right' , 5:'Straight'}
        measurements, sensor_data = client.read_data()
        client.send_control(VehicleControl())

        initial_timestamp = measurements.game_timestamp
        current_timestamp = initial_timestamp

        # The vector containing all measurements produced on this episode
        measurement_vec = []
        # The vector containing all controls produced on this episode
        control_vec = []
        frame = 0
        distance = 10000
        success = False
        # Initialize lane and Offroad fields
        data["Intersection-Lane"] = 0
        data["Intersection-Offroad"] = 0
        data["Directions"] = []
        agent.command_follower.controller.params['target_speed'] = agent.command_follower.controller.params['default_speed_limit']
        agent.traffic_light_infraction = False

        while True: #

            # Read data from server with the client
            measurements, sensor_data = client.read_data()
            # The directions to reach the goal are calculated.
            directions = self._get_directions(measurements.player_measurements.transform, target)
            # Agent process the data.
            if direction2text[directions] not in data["Directions"]:
                data["Directions"].append(direction2text[directions])
            control , controller_state = agent.run_step(measurements, sensor_data, directions, target)
            # Send the control commands to the vehicle
            client.send_control(control)
            if measurements.player_measurements.intersection_otherlane > data["Intersection-Lane"]:
                data["Intersection-Lane"] = measurements.player_measurements.intersection_otherlane
            if measurements.player_measurements.intersection_offroad > data["Intersection-Offroad"]:
                data["Intersection-Offroad"] = measurements.player_measurements.intersection_offroad    
            # save images if the flag is activated
            self._recording.save_images(sensor_data, episode_name, frame)

            current_x = measurements.player_measurements.transform.location.x
            current_y = measurements.player_measurements.transform.location.y
            # If vehicle has stopped for pedestiran, vehicle or traffic light increase timeout by 1 unit
            if min(controller_state['stop_pedestrian'], controller_state['stop_vehicle'],\
                controller_state['stop_traffic_lights']) == 0 :
                time_out+= (measurements.game_timestamp - current_timestamp)/1000
            #logging.info("Controller sis Inputting:")
            #logging.info('Steer = %f Throttle = %f Brake = %f ',
            #             control.steer, control.throttle, control.brake)
            current_timestamp = measurements.game_timestamp
            # Get the distance travelled until now
            distance = sldist([current_x, current_y],
                              [target.location.x, target.location.y])
            # Write status of the run on verbose mode
            #logging.info('Status:')
            print(f"Distance to target: {float(distance):.2f} \t Time_taken: {(current_timestamp -initial_timestamp)/1000 :.2f}\t Timeout : {time_out :.2f}")
            # Check if reach the target
            if distance < self._distance_for_success:
                success = True
                break
            if (current_timestamp-initial_timestamp)/1000 > time_out:
                success = False
                print("Timeout")
                break

            # Increment the vectors and append the measurements and controls.
            frame += 1
            measurement_vec.append(measurements.player_measurements)
            control_vec.append(control)

            #Stop in case of collision or excessive lane infraction
            if measurements.player_measurements.collision_pedestrians >300 or \
                measurements.player_measurements.collision_vehicles > 300 or \
                measurements.player_measurements.collision_other>300:
                print("Collison detected")
                success = False
                break
        
            if data["Intersection-Lane"] > 0.5 or \
            data["Intersection-Offroad"] > 0.5 :
                print("Lane Infraction")
                success = False
                break

        data["Intersection-Lane"] *= 100
        data["Intersection-Offroad"] *= 100    
        data["Collision-Pedestrian"] = measurements.player_measurements.collision_pedestrians
        data["Collision-Traffic"] = measurements.player_measurements.collision_vehicles
        data["Collision-Other"] = measurements.player_measurements.collision_other
        if data["Collision-Pedestrian"] <= 300 and  data["Collision-Traffic"] <= 400  and  data["Collision-Other"] <= 400 and success ==True: 
            data['Success without Collision'] = 1
        else:
            data['Success without Collision'] = 0
        if  data["Intersection-Lane"] < 20 and data["Intersection-Offroad"] < 20 and success == True:
            data["Success without Lane-Infraction"] = 1
        else: 
            data["Success without Lane-Infraction"] = 0
        if success:
            return 1, measurement_vec, control_vec, float(
                current_timestamp - initial_timestamp) / 1000.0, distance, data
        return 0, measurement_vec, control_vec, time_out, distance, data


def run_driving_benchmark(agent,
                          experiment_suite,
                          city_name='Town01',
                          log_name='Test',
                          continue_experiment=False,
                          host='127.0.0.1',
                          port=2000
                          ):
    while True:
        try:

            with make_carla_client(host, port) as client:
                # Hack to fix for the issue 310, we force a reset, so it does not get
                #  the positions on first server reset.
                client.load_settings(CarlaSettings())
                client.start_episode(0)
               

                # We instantiate the driving benchmark, that is the engine used to
                # benchmark an agent. The instantiation starts the log process, sets

                benchmark = DrivingBenchmark(city_name=city_name,
                                             name_to_save=log_name + '_'
                                                          + type(experiment_suite).__name__
                                                          + '_' + city_name,
                                             continue_experiment=continue_experiment)
                # This function performs the benchmark. It returns a dictionary summarizing
                # the entire execution.

                benchmark_summary = benchmark.benchmark_agent(experiment_suite, agent, client)
               

                print("")
                print("")
                print("----- Printing results for training weathers (Seen in Training) -----")
                print("")
                print("")
                results_printer.print_summary(benchmark_summary, experiment_suite.train_weathers,
                                              benchmark.get_path())

                print("")
                print("")
                print("----- Printing results for test weathers (Unseen in Training) -----")
                print("")
                print("")

                results_printer.print_summary(benchmark_summary, experiment_suite.test_weathers,
                                              benchmark.get_path())

                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
