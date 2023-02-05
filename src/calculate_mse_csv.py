import csv

def sum_mse(path):
    """
    Adds up the MSE for each run of the simulation 
    """
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        csv_data = [float(row[8]) for row in reader]
        print(sum(csv_data))

def calculate_mse(path):
    """
    Takes the total_x and total_y which is the sum of the arithmetic_x and arithmetic_y
    for each run of the simulation and divides by the number of times the 
    simulation has been run. It then calculates the MSE based on that.
    """
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        csv_data = [row for row in reader]
        sum = 0
        for i, elem in enumerate(csv_data):
            total_x = float(elem[3])
            total_y = float(elem[4])
            true_x = float(elem[1])
            true_y = float(elem[2])
            times = float(elem[7])
            print(times)
            sum += (total_x / times - true_x + total_y / times - true_y)**2

        print(sum)

if __name__ == "__main__":
    #path = "/root/.ros/slam_data.txt"
    path = "../csv_files/slam_eucl.csv"
    sum_mse(path)
    calculate_mse(path)

#slam_maha
#768.5845840339
#251.22265083839991

#slam_eucl
#793.6744750649135
#149.53904887456042