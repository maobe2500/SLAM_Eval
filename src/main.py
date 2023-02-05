import datetime
from typing import List, Tuple

import matplotlib
import yaml
import csv
import json
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from matplotlib.patches import Ellipse

"""
------------------------------------------
# CONE INFO FROM GITHUB:
# Description of a cone
#

# 2D-position of the cone (expressed in the frame stated in the Cones message)
float64 x
float64 y
float64 z

# Color of the cone
uint8 UNDEFINED = 0
uint8 YELLOW = 1
uint8 BLUE = 2
uint8 SMALL_ORANGE = 3
uint8 BIG_ORANGE = 4

uint8 color

# Covariance on the position [m^2] (2x2 matrix in row-major order)
float64[4] covariance

# Confidence in the detection
float64 probability

# Index of the cone (for fake data association) (0 if unknown)
int32 id

------------------------------------------
"""


def read_real_yaml_bag(path):
    """
    Reads the yaml file for a rosbag filled with real (not simulated) data
    and creates a dict with all data.

    WARNING: Very slow for big files, needs to be optimized. In the meantime
    write data to csv file directly after so it only needs to be done once!

    :param path:         Path to bag file in yaml format

    :return map_data:    A dict with the same structure as the bag file
                         where each time_stamp has its own corresponding map,
                         but with all the clutter removed.
    """
    with open(path) as f:
        map_data = {}
        maps = list(yaml.load_all(f, yaml.Loader))
        # Needed try/except since we end with Ctrl-C which cuts data off uncleanly at the end
        try:
            for slam_map in maps:
                time_stamp = slam_map["header"]["seq"]
                mapped_cones = slam_map["cones"]
                map_data[time_stamp] = []
                for i, cone_with_stats in enumerate(mapped_cones):
                    x = cone_with_stats["x"]
                    y = cone_with_stats["y"]
                    covariance = cone["covariance"]
                    map_data[time_stamp].append({"time_stamp": time_stamp, "x": x, "y": y, "covariance": covariance, "id": i})
        except Exception as e:
            print("Whoops", e)
        return map_data

def read_yaml_bag(path):
    """
    Reads the yaml file and creates a dict with all data

    WARNING: Very slow for big files, needs to be optimized. In the meantime
             write data to csv file directly after so it only needs to be done once!

    See structure of data above
    :param path:         Path to bag file in yaml format

    :return map_data:    A dict with the same structure as the bag file
                         where each time_stamp has its own corresponding map,
                         but with all the clutter removed.
    """
    with open(path) as f:
        map_data = {}
        maps = list(yaml.load_all(f, yaml.Loader))
        # Needed try/except since we end with Ctrl-C which cuts data off uncleanly at the end
        try:
            for slam_map in maps:
                time_stamp = slam_map["header"]["seq"]
                mapped_cones = slam_map["cones"]
                map_data[time_stamp] = []
                for i, cone_with_stats in enumerate(mapped_cones):
                    cone = cone_with_stats["cone"]
                    stats = cone_with_stats["stats"]
                    inn_x = stats["avg_innovation_x"]
                    inn_y = stats["avg_innovation_y"]
                    inn_std = stats["std_innovation"]
                    # color = cone["color"]
                    # prob = cone["probability"]      # Is always zero for now
                    # id = cone["id"]                 # Is also always zero, otherwise it'd be very useful
                    x = cone["x"]
                    y = cone["y"]
                    covariance = cone["covariance"]
                    map_data[time_stamp].append({"time_stamp": time_stamp, "x": x, "y": y, "covariance": covariance, "id": i, "inn_x":inn_x, "inn_y": inn_y, "inn_std":inn_std})
        except Exception as e:
            print("Whoops", e)
        return map_data


def write_to(filename, data, fieldnames=None):
    """
    Writes data from a dictionary to a file of type csv or json
    depending on the filename.

    :param filename:    The filename of the new file
    :param data:        The data to write
    :param fieldnames:  The name of each column as a list of strings

    :return: None
    """
    print("First key : ", list(data.keys())[0])

    filetype = filename.split(".")[1]
    if filetype != "csv" and filetype != "json":
        print("Invalid filetype")
        return
    with open(filename, "w") as f:
        print("Writing to {}...".format(filename))
        if filetype == "json":
            json.dump(data, f)
        else:
            if not fieldnames:
                first_key = list(data.keys())[0]
                fieldnames = data[first_key][0].keys()
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            for time_stamp, cones in data.items():
                for cone in cones:
                    wr.writerow(cone)
        print("{} Done!".format(filename))

def get_cones(path):
    """
    Gets the cones from the specified track 

    :param path: path to the file of the track
    
    :return: returns a list of lists with the blue cones, big_orange cones
             small_orange cones and the yellow cones in that order

    """
    with open(path) as f:
        track = list(yaml.load_all(f, yaml.Loader))

        return [track[0]['cones']['blue'],
                track[0]['cones']['big_orange'],
                track[0]['cones']['small_orange'],
                track[0]['cones']['yellow']]

def plot_real_cones(axes_limits=25):
    """
    Plots the cones from the list of coordinates in "blue", "big_orange" and "yellow" (Maybe wrong?)


    :param axes_limits: sets the limits both ways for x and y centered on the origin

    :return: fig, ax:   Standard matplotlib Figure and Axes objects

    :TODO:              Should be changed to read the .yaml files from ARCS directly instead!
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-axes_limits, axes_limits)
    ax.set_ylim(-axes_limits, axes_limits)
    [blue, big_orange, _, yellow] = get_cones("tracks/trackdrive.yaml")

    #zips the x and y values in to separate lists
    bx, by = zip(*blue)
    Ox, Oy = zip(*big_orange)
    yx, yy = zip(*yellow)

    ax.scatter(bx, by, c="blue")
    ax.scatter(Ox, Oy, c="orange")
    ax.scatter(yx, yy, c="yellow")
    return fig, ax


def read_map_csv(path, animate_maps = False):
    """
    Reads the csv file created by the read_yaml_bag function.

    :param path:            Path to the csv file
    :param animate_maps:    If set to true: gives back a list of maps, where each map is a 
                            map is a list of (x, y) tuples for a given timestamp.

    :return (x_vals, y_vals, times):    tuple containing a all x and y values with times stamps
                                        The time stamp is what signifies what

    :TODO:  Make readable by humans. Current state is due to bugfixing at 3:am.

    """
    with open(path) as f:
        rows = csv.reader(f)
        field_names = next(rows)
        map_frames = []
        times, x_vals, y_vals = [], [], []
        x, y, cov, inn_x, inn_y, inn_std = [], [], [], [], [], []
        for i, row in enumerate(rows):  # Every row contains data for a single cone
            times.append(int(row[0]))
            x_vals.append(float(row[1]))
            x.append(float(row[1]))
            y.append(float(row[2]))
            y_vals.append(float(row[2]))
            # --- Not using these right now ---
            cov.append(([float(var) for var in row[3].split(",")[1:-2]]))
            inn_x.append(float(row[5]))
            inn_y.append(float(row[6]))
            inn_std.append(float(row[7]))
            #----------------------------------
            if animate_maps and int(row[4]) == 0 and i != 0:
                map_frames.append([x_vals, y_vals, cov, inn_x, inn_y, inn_std])
                x_vals, y_vals = [], []
        if animate_maps:
            return map_frames
        return x, y, times


class MapAnimation:
    def __init__(self, map_frames):
        """
        Creates an animation of all the maps in map_frames list

        :param map_frames: A list of lists of (x, y) tuples of a map at each timestamp

        :TODO:  Needs to be able to read different track setups
        """
        self.map_frames = map_frames
        self.map_fig = plt.figure()
        self.lim = 25
        self._ax = plt.axes(xlim=(-self.lim, self.lim), ylim=(-self.lim, self.lim))
        selected_track = "tracks/trackdrive.yaml"
        [self.blue, self.big_orange, self.small_orange, self.yellow] = get_cones(selected_track)
        self.init_anim()
        self.col = [(1-i/len(self.map_frames), i/len(self.map_frames), 0, 0.5) for i in range(int(len(self.map_frames)))]
        self.map_plot = self._ax.scatter([], [], s=100)
        self.init_anim()

    def init_anim(self):
        """ Plots the real cones"""
         #zips the x and y values in to separate lists
        bx, by = zip(*self.blue)
        Ox, Oy = zip(*self.big_orange)
        yx, yy = zip(*self.yellow)

        self._ax.scatter(bx, by, c="blue")
        self._ax.scatter(Ox, Oy, c="orange")
        self._ax.scatter(yx, yy, c="yellow")


    def animate_map(self, i):
        """
        The "artist" function that the funcanimation needs. Can be thought of as
        drawing one frame for every i

        :param i:   standard parameter according to matplotlib documentation for Funcanimation
        :return:    The axes object for the animation, the trailing comma is because
                    of matplotlibs Funcanimation. It expects a tuple i think.
        """
        data = self.map_frames[i]
        #self.col = [n for n in range(len(data))]
        self.map_plot.set_offsets(data[1:2])
        self.map_plot.set_color(self.col[i])
        self.map_plot.set_offsets(np.stack((self.map_frames[i][0], self.map_frames[i][1]), axis=1))
        return self.map_plot,

    def animate(self, filename=None):
        """
        Animates the maps and saves the file to filename if it is not None

        :param filename:    A .gif file to save the animation to
        :return:    None
        """
        ani = FuncAnimation(self.map_fig, self.animate_map, frames=len(self.map_frames), blit=True, interval=20)
        plt.show()
        if filename:
            try:
                ani.save(filename, writer=PillowWriter(fps=30))
            except Exception as e:
                print(e)
                print("Something went wrong, saving to .gif file")
                filename = datetime.datetime.now().strftime("map_at_time_%m-%d-%H:%M.gif")
                ani.save(filename, writer=PillowWriter(fps=30))


def main():
    matplotlib.style.use("ggplot")
    # Change this to the path of the bag you want to read
    map_data = read_real_yaml_bag(path="bags/real_cone_data.yaml")
    # print(map_data)
    # Change this to the path of the csv file you want to write to
    write_to("csv_files/real_cone_data.csv", map_data)
    fig, ax = plot_real_cones()
    # Link to all colormaps: https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    # To use a colormap, add "_r" to the end of the name
    #x_vals, y_vals, times = read_map_csv("./csv_files/mapped_cones_data.csv")
    #ax.scatter(x_vals, y_vals, s=20, alpha=0.1, c=inn_std, cmap="cool_r")
    #plt.show()
    map_frames = read_map_csv("csv_files/real_cone_data.csv", animate_maps=True)
    ani = MapAnimation(map_frames)
    ani.animate()



if __name__ == '__main__':
    main()
