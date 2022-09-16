import yaml
import csv
import json
import matplotlib.pyplot as plt

# Type definitions: (mostly for readability)

Mapped_Coordinate = tuple[float, float]
Covariance = list[float, float, float, float]
Time_Stamp = int
Index = int
Figure = plt.Figure
Axes = plt.axes
Time_Stamped_data = list[tuple[Mapped_Coordinate, Covariance, Time_Stamp]]

blue = [
    [-17.0129, 9.72719],
    [-12.7541, 12.8712],
    [-10.31569, 13.0924],
    [9.62956, 15.8379],
    [12.492, 17.0404],
    [15.1389, 16.5025],
    [19.5034, 13.6361],
    [20.8922, 11.7012],
    [17.5221, 15.3812],
    [21.4761, 8.79574],
    [20.9992, 5.27662],
    [19.9925, 2.24053],
    [19.0983, -0.671871],
    [17.1824, -3.23994],
    [11.114, -6.74408],
    [-4.06552, 13.3637],
    [14.2831, -5.26437],
    [8.25363, -8.53889],
    [5.06185, -10.1551],
    [1.42086, -11.9634],
    [-2.4975, -14.0305],
    [-5.74864, -16.1217],
    [-9.34841, -17.1551],
    [-12.2114, -16.6459],
    [-14.4625, -14.9249],
    [-16.2427, -13.276],
    [-0.131239, 13.3125],
    [-17.8615, -11.5349],
    [-19.1945, -8.52656],
    [-18.9382, -5.15509],
    [-18.8625, -4.02997],
    [-18.6872, -0.45783],
    [-18.1384, 2.83132],
    [-17.8432, 6.27091],
    [-15.5443, 12.0063],
    [3.50416, 13.7245],
    [7.13676, 14.727]
]

big_orange = [
    [-6.90273, 7.20128],
    [-7.90273, 7.20128],
    [-6.90273, 13.0924],
    [-7.90273, 13.0924]
]

small_orange = []

yellow = [
    [-12.2686, 6.61784],
    [-9.65737, 7.2378],
    [7.33516, 9.03961],
    [10.548, 10.5892],
    [13.1479, 11.1316],
    [15.2725, 10.0791],
    [16.8037, 8.23821],
    [16.7063, 6.10772],
    [15.876, 3.47906],
    [15.106, 1.5027],
    [13.6765, -0.472612],
    [9.41953, -2.84739],
    [12.0732, -1.43122],
    [7.26282, -3.9408],
    [4.65159, -5.15509],
    [1.78774, -6.66723],
    [-1.97969, -8.56329],
    [-3.80615, 7.33616],
    [-5.18123, -10.5555],
    [-8.34841, -12.1551],
    [-9.78487, -11.8592],
    [-11.3484, -11.1551],
    [-13.2946, -8.64709],
    [-14.2998, -6.05581],
    [-14.0528, -3.05427],
    [-13.6807, -0.101053],
    [-13.3484, 2.84491],
    [-0.595997, 7.27843],
    [-13.3484, 4.84491],
    [2.90884, 7.57594]
]

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
Map_data Structure:

    Example of data:
        {"4146": 
            [
                {"id": 0, 
                "x": -4.17621032625, 
                "y": 13.3964256111, 
                "covariance": [0.00011083743842364538, 3.257323331016572e-23, 
                               -3.257323331016572e-23, 0.00011083743842364538]}, 

                ...

                {"id": i, 
                "x": x, 
                "y": y, 
                "covariance": [variance_x, variance_yx, 
                               variance_xy, variance_y]}
            ]
        },

         {"time+1": 
            [
                {"id": i, 
                "x": x, 
                "y": y, 
                "covariance": [variance_x, variance_yx, 
                               variance_xy, variance_y]
                }, 
            ] 
            ...

"""


def read_yaml_bag(path: str, data_length: int = 10, from_back: bool = False) -> dict:
    """
    Reads the yaml file and creates two different data dicts

    See structure of data above
    :param path:        Path to bag
    :param data_length: Specifies the amount of maps to read
    :param from_back:   If true will read data from the back instead of front

    :return map_data:    A dict with the same structure as the bag file
                         where each time_stamp has its own corresponding map,
                         but with all the clutter removed.


    :todo:  Add the id, probability and color of each cone in the data (currently not useful as they are always zero)

    """
    with open(path) as f:
        map_data = {}
        if from_back:
            maps = list(yaml.load_all(f, yaml.FullLoader))[-data_length:]
        else:
            maps = list(yaml.load_all(f, yaml.FullLoader))[:data_length]
        # Needed try/except since we end with Ctrl-C which cuts data off uncleanly at the end
        try:
            i = 0
            for slam_map in maps:
                print("Working still")
                time_stamp = slam_map["header"]["seq"]
                mapped_cones = slam_map["cones"]
                map_data[time_stamp] = []
                for i, cone in enumerate(mapped_cones):
                    # color = cone["color"]
                    # prob = cone["probability"]      # Is always zero for now
                    # id = cone["id"]                 # Is also always zero, otherwise it'd be very useful
                    x = cone["x"]
                    y = cone["y"]
                    covariance = cone["covariance"]
                    map_data[time_stamp].append(
                        {"time_stamp": time_stamp, "x": x, "y": y, "covariance": covariance, "id": i})
                if i >= data_length:
                    break
                print(i)
                i += 1
                #print(f"{(i / len(maps)) * 100}% complete!")
                print(map_data)
        except Exception as e:
            print("Whoops", e)
            return map_data
        return map_data


def write_to(filename, data: dict, fieldnames: list[str] = None) -> None:
    """
    Writes data from a dictionary to a file of type csv or json.

    :param filename:    The filename of the new file
    :param data:        The data to write
    :param fieldnames:  The name of each column as a list of strings

    :return: None
    """
    filetype = filename.split(".")[1]
    if filetype != "csv" and filetype != "json":
        print("Invalid filetype")
        return
    with open(filename, "w") as f:
        print(f"Writing to {filename}...")
        if filetype == "json":
            json.dump(data, f)
        else:
            if not fieldnames:
                first_key = list(data.keys())[0]
                fieldnames = data[first_key][0].keys()
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            for k, v in data.items():
                for cone in v:
                    wr.writerow(cone)
        print(f"{filename} Done!")


def plot_real_cones(axes_limits: int = 25) -> tuple[Figure, Axes]:
    """
    Plots the cones from the list of coordinates in "blue", "big_orange" and "yellow" (Maybe wrong?)

    :param axes_limits: sets the limits both ways for x and y centered on the origin

    :return: fig, ax:   Standard matplotlib Figure and Axes objects
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-axes_limits, axes_limits)
    ax.set_ylim(-axes_limits, axes_limits)
    # Ugly code below
    bx, by = [], []
    for blue_cone in blue:
        bx.append(blue_cone[0])
        by.append(blue_cone[1])
    Ox, Oy = [], []
    for big_orange_cone in big_orange:
        Ox.append(big_orange_cone[0])
        Oy.append(big_orange_cone[1])
    yx, yy = [], []
    for yellow_cone in yellow:
        yx.append(yellow_cone[0])
        yy.append(yellow_cone[1])

    ax.scatter(bx, by, c="blue")
    ax.scatter(Ox, Oy, c="orange")
    ax.scatter(yx, yy, c="yellow")
    return fig, ax


def read_map_csv(path: str) -> tuple[list[float], list[float], list[int]]:
    """
    Reads the csv file created by the read_yaml_bag function

    :param path:    Path to the csv file

    :return (x_vals, y_vals, times):    tuple containing a all x and y values with times stamps
                                        The time stamp is what signifies what
    """
    with open(path) as f:
        rows = csv.reader(f)
        field_names = next(rows)
        times, x_vals, y_vals = [], [], []
        for row in rows:  # Every row contains data for a single cone
            times.append(int(row[0]))
            x_vals.append(float(row[1]))
            y_vals.append(float(row[2]))
        return x_vals, y_vals, times

def main():
    map_data = read_yaml_bag(path="bags/bag.yaml", data_length=10)
    print(map_data)
    write_to("csv_files/map_with_stats.csv", map_data)
    write_to("json_files/map_with_stats.json", map_data)
    fig, ax = plot_real_cones()
    # Link to all colormaps: https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    # To use a colormap, add "_r" to the end of the name
    x_vals, y_vals, times = read_map_csv("./csv_files/slam_map_from_back.csv")
    ax.scatter(x_vals, y_vals, s=2, alpha=0.4, c=times, cmap="cool_r")
    plt.show()


if __name__ == '__main__':
    main()
