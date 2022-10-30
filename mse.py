import main 
import csv
import numpy as np
import operator


#Constants used in calculate_closest_cone
TRUTH_CONES = main.get_cones("tracks/trackdrive.yaml")
FLAT_CONES = [item for sublist in TRUTH_CONES for item in sublist]
TRUTH_CONES_ARR = np.asarray(FLAT_CONES)

def mse():
    cones = create_cone_centers("csv_files/mapped_cones_data.csv")
    print_cones(cone_centers_mse(cones))


def calculate_closest_cone(position):
    """
    Gets the index of the cone closest to the measurement

    :param position: the measured position of the cone represented as a (x,y)-tuple
    :return: the index of the cone closes to the measurement in cones_arr
    """
    dist = np.sum((TRUTH_CONES_ARR - position)**2, axis=1)
    return np.argmin(dist)


def calc_mse(path):
    """
    Calculates the Mean Squared Error (MSE) of a file with cone information.

    :param path: path to the .csv file where the cone information is found
    :return: returns the Mean Squared Error of the cones in the file from path
    """
    with open(path) as f:
        rows = csv.reader(f)
        num_rows = 0
        #First line is only metadata
        next(rows)
        sum = 0
        for row in rows:
            num_rows += 1
            (x,y) = (float(row[1]), float(row[2]))
            index = calculate_closest_cone((x,y))
            closest = TRUTH_CONES_ARR[index]
            sum += (closest[0] - x + closest[1] - y)**2
        return sum / num_rows

def create_cone_centers(path):
    """
    Calculates a center for each seen cone by adding the measurements and 
    dividing by the number of times the cone has been seen

    :param path: path to the .csv file where the cone information is found
    :return: returns a dictionary with a center for all the seen cones and
    the number of times it has been seen
    """

    with open(path) as f:
        rows = csv.reader(f)
        #First line consists only of metadata
        next(rows)
        #For each cone, store the combined values of the measurements for that cone
        #and the number of times it has been seen
        seen_cones = {}
        for row in rows:
            (x,y) = (float(row[1]), float(row[2]))
            index = calculate_closest_cone((x,y))
            if index not in seen_cones:
                seen_cones[index] = [x,y,1]
            else:
                temp = seen_cones[index]
                #Add the x value, y value and 1 for having seen it
                temp = [temp[0] + x, temp[1] + y, temp[2] + 1]
                seen_cones[index] = temp
        return seen_cones

def cone_centers_mse(seen_cones):
    """
    Calculates the MSE for each cone in seen_cones

    :param seen_cones: measurements for seen cones and the number of times it has been seen 
    :return seen_cones_mse: a list with the index of a cone, the mse and the number of times it has
    been seen
    """
    seen_cones_mse = []
    for key in seen_cones:
        cone = seen_cones[key]
        cone[0] = cone[0] / cone[2]
        cone[1] = cone[1] / cone[2]
        index = calculate_closest_cone((cone[0],cone[1]))
        closest = TRUTH_CONES_ARR[index]
        cone_mse = (closest[0] - cone[0] + closest[1] - cone[1])**2
        seen_cones_mse.append((index, cone_mse, cone[2]))

    return seen_cones_mse

def cone_mse(path):
    """
    Calculates a center for each seen cone and then performs MSE based on each center.

    :param path: path to the .csv file where the cone information is found
    :return: returns the Mean Squared Error of the cones in the file from path
    """

    with open(path) as f:
        rows = csv.reader(f)
        #First line consists only of metadata
        next(rows)
        #For each cone, store the combined values of the measurements for that cone
        #and the number of times it has been seen
        seen_cones = {}
        for row in rows:
            (x,y) = (float(row[1]), float(row[2]))
            index = calculate_closest_cone((x,y))
            if index not in seen_cones:
                seen_cones[index] = [x,y,1]
            else:
                temp = seen_cones[index]
                #Add the x value, y value and 1 for having seen it
                temp = [temp[0] + x, temp[1] + y, temp[2] + 1]
                seen_cones[index] = temp
    cone_measurements = []
    for key in seen_cones:
        cone = seen_cones[key]
        cone[0] = cone[0] / cone[2]
        cone[1] = cone[1] / cone[2]
        index = calculate_closest_cone((cone[0],cone[1]))
        closest = TRUTH_CONES_ARR[index]
        cone_mse = (closest[0] - cone[0] + closest[1] - cone[1])**2
        cone_measurements.append((index, cone_mse, cone[2]))

    return cone_measurements
    

def print_cones(cones):
    cones.sort()
    print("Cone index | (Center - Closest)^2 | Time seen")
    for cone in cones:
        print(cone[0], " ", cone[1], " ", cone[2])


if __name__ == "__main__":
    mse()