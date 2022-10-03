import main 
import csv
import numpy as np

#Constants used in calculate_closest_cone
CONES = main.get_cones("tracks/trackdrive.yaml")
FLAT_CONES = [item for sublist in CONES for item in sublist]
CONES_ARR = np.asarray(FLAT_CONES)

def mse():
    print(calc_mse("csv_files/mapped_cones_data.csv"))

"""
Gets the index of the cone closest to the measurement

:param position: the measured position of the cone represented as a (x,y)-tuple
:return: the index of the cone closes to the measurement in cones_arr
"""
def calculate_closest_cone(position):
   dist = np.sum((CONES_ARR - position)**2, axis=1)
   return np.argmin(dist)

"""
Calculates the Mean Squared Error (MSE) of a file with cone information.

:param path: path to the .csv file where the cone information is found
:return: returns the Mean Squared Error of the cones in the file from path
"""
def calc_mse(path):
    with open(path) as f:
        rows = csv.reader(f)
        num_rows = 0
        #First line is only metadata
        _ = next(rows)
        sum = 0
        for row in rows:
            num_rows += 1
            (x,y) = (float(row[1]), float(row[2]))
            index = calculate_closest_cone((x,y))
            closest = CONES_ARR[index]
            sum += (closest[0] - x + closest[1] - y)**2
            #For debugging purposes
            #print("Measurement : ", position)
            #print("Closest : ", closest)
            #print("Add : ", (closest[0] - position[0])**2 + (closest[1] - position[1])**2)
            #print("Num_rows : ", num_rows)
        return sum / num_rows


if __name__ == "__main__":
    mse()