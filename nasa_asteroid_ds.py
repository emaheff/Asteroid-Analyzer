"""
@Project: NASA Asteroid Data Analysis

@Description: This project analyzes NASA's asteroid data. It includes functions for loading and processing
the data, calculating various statistics, and creating visualizations. The main functionalities include
finding asteroids with extreme properties, analyzing orbit patterns, and visualizing diameter distributions
and hazard classifications.

@ID: 312566540
@Author: Emmanuel Heffes
@semester: 24b
"""
import csv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

CLOSE_APPROACH_YEAR = 2000
NAMES_TO_FILTER = ['Neo Reference ID', 'Orbiting Body', 'Equinox']
CSV_FILE_NAME = "nasa.csv"

# Add these new constants
MIN_DIAMETER_COLUMN = 'Est Dia in KM(min)'
MAX_DIAMETER_COLUMN = 'Est Dia in KM(max)'
ABSOLUTE_MAGNITUDE_COLUMN = 'Absolute Magnitude'
MILES_PER_HOUR_COLUMN = 'Miles per hour'
NAME_COLUMN = 'Name'
ORBIT_ID_COLUMN = 'Orbit ID'
HAZARDOUS_COLUMN = 'Hazardous'
CLOSE_APPROACH_DATE_COLUMN = 'Close Approach Date'
MISS_DISTANCE_COLUMN = 'Miss Dist.(kilometers)'


def load_data(file: str) -> np.ndarray:
    try:
        # Read headers directly from the CSV file
        with open(file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)

        headers_array = np.array(headers, dtype=str).reshape(1, -1)

        # Load data from CSV without headers, using the same delimiter as the csv.reader
        data = np.genfromtxt(file, delimiter=",", encoding='utf-8', dtype=str, skip_header=1)

        # Ensure data has the same number of columns as headers
        if data.shape[1] != len(headers):
            print(f"Warning: Data has {data.shape[1]} columns, but there are {len(headers)} headers.")
            print("Adjusting data to match headers...")
            data = data[:, :len(headers)]

        # Combine headers and data arrays
        result = np.vstack((headers_array, data))

        return result

    except FileNotFoundError:
        print(f"The file {file} was not found. Please check the file path and name.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def scoping_data(data: np.ndarray, names: list) -> np.ndarray:
    """
    Filter the data to exclude specified columns.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
        names (list): The list of column names to filter out.

    Returns:
        np.ndarray: Filtered data array.
    """
    headers = data[0]
    # Get indices of columns to keep
    indices_to_keep = [i for i, header in enumerate(headers) if header not in names]
    # Filter data based on indices
    return data[:, indices_to_keep]


def mask_data(data: np.ndarray) -> np.ndarray:
    """
    Filter the data to include only rows where the close approach year is >= CLOSE_APPROACH_YEAR.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        np.ndarray: Filtered data array.
    """
    header = data[0]
    try:
        col = extract_column_by_header(data, CLOSE_APPROACH_DATE_COLUMN)
        # Extract year from date and create boolean mask
        years = np.array([int(date.split('-')[0]) for date in col])
        condition = years >= CLOSE_APPROACH_YEAR
        filtered_data = data[1:][condition]
        # Combine header with filtered data
        return np.vstack((header, filtered_data))
    except Exception as e:
        print(f"An error occurred while masking data: {e}")
        return data


def data_details(data: np.ndarray):
    """
    Print the details of the filtered data.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
    """
    try:
        new_data = scoping_data(data, NAMES_TO_FILTER)
        # Print number of rows and columns
        print(f"num of rows: {len(new_data)}.\tnum of columns: {len(new_data[0])}")
        # Print header row
        print(new_data[0])
    except Exception as e:
        print(f"An error occurred while displaying data details: {e}")


def max_absolute_magnitude(data: np.ndarray) -> tuple:
    """
    Find the asteroid with the maximum absolute magnitude.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        tuple: Name and maximum absolute magnitude.
    """
    return find_extreme_value(data, ABSOLUTE_MAGNITUDE_COLUMN)


def closest_to_earth(data: np.ndarray) -> str:
    """
    Find the asteroid closest to Earth.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        str: Name of the closest asteroid.
    """
    return find_extreme_value(data, MISS_DISTANCE_COLUMN, 'min')[0]


def common_orbit(data: np.ndarray) -> dict:
    """
    Returns a sorted dictionary when the keys have the orbit id and the value of each key is the number of asteroids
    that share the orbit id

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        dict: Dictionary of orbit IDs and their counts, sorted by orbit ID.
    """
    try:
        orbit_col = extract_column_by_header(data, ORBIT_ID_COLUMN)
        orbit_dic = {}
        # Count occurrences of each orbit ID
        for orbit_id in orbit_col:
            if orbit_id in orbit_dic:
                orbit_dic[orbit_id] += 1
            else:
                orbit_dic[orbit_id] = 1
        # Return sorted dictionary
        return dict(sorted(orbit_dic.items()))
    except Exception as e:
        print(f"An error occurred while finding common orbits: {e}")
        return {}


def min_max_diameter(data: np.ndarray) -> tuple:
    """
    Calculates the average minimum and maximum estimated diameters of asteroids.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        tuple: A tuple containing the average minimum diameter and the average maximum diameter.
    """
    try:
        # Extract columns for minimum and maximum estimated diameters
        min_diameters = extract_column_by_header(data, MIN_DIAMETER_COLUMN)
        max_diameters = extract_column_by_header(data, MAX_DIAMETER_COLUMN)

        # Sum up all the values in the columns
        total_min = sum(float(d) for d in min_diameters)
        total_max = sum(float(d) for d in max_diameters)

        avg_min = total_min / len(min_diameters)
        avg_max = total_max / len(max_diameters)

        return avg_min, avg_max
    except Exception as e:
        print(f"An error occurred while calculating min and max diameters: {e}")


def plt_hist_diameter(data: np.ndarray):
    """
    Plots a histogram of the average diameters of asteroids.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
    """
    try:
        average_est = get_average_diameter(data)
        min_diameter, max_diameter = min_max_diameter(data)

        # Plot histogram
        plt.hist(average_est, bins=10, color='steelblue', edgecolor='black', range=(min_diameter, max_diameter))

        # Add labels and title
        plt.xlabel('Average Diameter (km)')
        plt.ylabel('Number of Asteroids')
        plt.title('Histogram of Average Diameters of Asteroids')
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the histogram: {e}")


def plt_hist_common_orbit(data: np.ndarray):
    """
    Plots a histogram of asteroids by their minimum orbit intersection.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
    """
    try:
        # Extract the 'Orbit ID' column
        orbit_id_col = extract_column_by_header(data, ORBIT_ID_COLUMN)

        # Convert orbit IDs to float (assuming they are numeric)
        orbit_id_col = np.array(orbit_id_col, dtype=float)

        # Plot histogram
        plt.hist(orbit_id_col, bins=6, color='steelblue', edgecolor='black')

        # Add labels and title
        plt.xlabel('Minimum Orbit Intersection')
        plt.ylabel('Number of Asteroids')
        plt.title('Histogram of Asteroids by Minimum Orbit Intersection')

        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the histogram: {e}")


def plt_pie_hazard(data: np.ndarray):
    """
    Plot a pie chart showing the distribution of hazardous and non-hazardous asteroids.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
    """
    try:
        hazard_col = extract_column_by_header(data, HAZARDOUS_COLUMN)

        # Convert the hazard column to boolean
        hazard_bool = np.array([str(val).lower() == 'true' for val in hazard_col])

        # Count hazardous and non-hazardous asteroids
        sum_dangerous = np.sum(hazard_bool)
        sum_not_dangerous = len(hazard_bool) - sum_dangerous

        labels = [HAZARDOUS_COLUMN, 'Non-Hazardous']
        values = [sum_dangerous, sum_not_dangerous]
        colors = ['red', 'green']

        # Plot pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title(f"Percentage of {HAZARDOUS_COLUMN} and Non-Hazardous Asteroids")
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the pie chart: {e}")


def plt_liner_motion_magnitude(data: np.ndarray):
    """
    Plot a linear regression of absolute magnitude vs miles per hour.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
    """
    try:
        magnitude_col = np.array(extract_column_by_header(data, ABSOLUTE_MAGNITUDE_COLUMN), dtype=float)
        miles_per_hour_col = np.array(extract_column_by_header(data, MILES_PER_HOUR_COLUMN), dtype=float)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(magnitude_col, miles_per_hour_col)

        # Check if the p-value indicates a significant relationship (linear relationship)
        if p_value < 0.05:
            plt.scatter(magnitude_col, miles_per_hour_col)
            plt.plot(magnitude_col, slope * magnitude_col + intercept, color='red')
            plt.xlabel(ABSOLUTE_MAGNITUDE_COLUMN)
            plt.ylabel(MILES_PER_HOUR_COLUMN)
            plt.title(f'Linear Relationship between {ABSOLUTE_MAGNITUDE_COLUMN} and {MILES_PER_HOUR_COLUMN}')
            plt.show()
        else:
            print("No significant linear relationship between the absolute magnitude and speed of asteroids.")
    except Exception as e:
        print(f"An error occurred while plotting linear regression: {e}")


def find_extreme_value(data: np.ndarray, column_header: str, operation: str = 'max') -> tuple:
    """
    Find the extreme value (max or min) in a specified column.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
        column_header (str): The header of the column to search.
        operation (str): 'max' for maximum value, 'min' for minimum value.

    Returns:
        tuple: Name and extreme value.
    """
    try:
        col = extract_column_by_header(data, column_header)
        # Find index of extreme value
        if operation == 'max':
            extreme_index = np.argmax(col)
        else:
            extreme_index = np.argmin(col)

        name_col = extract_column_by_header(data, NAME_COLUMN)
        return name_col[extreme_index], col[extreme_index]
    except Exception as e:
        print(f"An error occurred while finding the extreme value in column {column_header}: {e}")


def extract_column_by_header(data: np.ndarray, target_header: str) -> np.ndarray:
    """
    Extract a column from the data given the header name.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
        target_header (str): The header name of the column to extract.

    Returns:
        np.ndarray: The column data.
    """
    header_index = np.where(data[0] == target_header)[0][0]
    return data[1:, header_index]


def get_average_diameter(data: np.ndarray) -> np.ndarray:
    """
    Calculate the average diameter from the given data.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        np.ndarray: The average estimated diameter.
    """
    try:
        min_diameters = np.array(extract_column_by_header(data, MIN_DIAMETER_COLUMN), dtype=float)
        max_diameters = np.array(extract_column_by_header(data, MAX_DIAMETER_COLUMN), dtype=float)
        return (min_diameters + max_diameters) / 2
    except ValueError as e:
        raise ValueError(f"Error converting diameter data to float: {e}")
    except KeyError as e:
        raise KeyError(f"Required column not found in data: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while calculating average diameter: {e}")


if __name__ == '__main__':
    # Load data from CSV file
    csv_data = load_data(CSV_FILE_NAME)

    plt_hist_diameter(csv_data)
    plt_hist_common_orbit(csv_data)
    plt_pie_hazard(csv_data)
    plt_liner_motion_magnitude(csv_data)
