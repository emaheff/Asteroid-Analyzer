import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

CLOSE_APPROACH_YEAR = 2000
HEADER_CLOSE_APPROACH_YEAR = 'Close Approach_Date'
HEADER_ABSOLUTE_MAGNITUDE = 'Absolute Magnitude'
HEADER_MISS_DISTKILOMETERS = 'Miss Distkilometers'
NAMES_TO_FILTER = ['Neo_Reference ID', 'Orbiting Body', 'Equinox']


def load_data(file):
    """
    Load the CSV file and return a combined array of headers and data.

    Parameters:
        file (str): The path to the CSV file.

    Returns:
        np.ndarray: Combined array of headers and data, or None if an error occurs.
    """
    try:
        # Load data from CSV with headers
        data = np.genfromtxt(file, delimiter=",", encoding='utf-8', dtype=None, names=True)

        # Extract and process headers
        headers = [header.replace('_', ' ') for header in data.dtype.names]
        headers_array = np.array(headers, dtype=str).reshape(1, -1)

        # Convert structured array to regular ndarray
        data_array = np.array(data.tolist(), dtype=object)

        # Combine headers and data arrays
        result = np.vstack((headers_array, data_array))
    except FileNotFoundError:
        print(f"The file {file} was not found. Please check the file path and name.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return result


def scoping_data(data, names):
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


def mask_data(data):
    """
    Filter the data to include only rows where the close approach year is >= CLOSE_APPROACH_YEAR.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        np.ndarray: Filtered data array.
    """
    header = data[0]
    try:
        col = extract_column_by_header(data, HEADER_CLOSE_APPROACH_YEAR)
        # Extract year from date and create boolean mask
        years = np.array([int(date.split('-')[0]) for date in col])
        condition = years >= CLOSE_APPROACH_YEAR
        filtered_data = data[1:][condition]
        # Combine header with filtered data
        return np.vstack((header, filtered_data))
    except Exception as e:
        print(f"An error occurred while masking data: {e}")
        return data


def data_details(data):
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


def max_absolute_magnitude(data):
    """
    Find the asteroid with the maximum absolute magnitude.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        tuple: Name and maximum absolute magnitude.
    """
    return find_extreme_value(data, HEADER_ABSOLUTE_MAGNITUDE)


def closest_to_earth(data):
    """
    Find the asteroid closest to Earth.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        str: Name of the closest asteroid.
    """
    return find_extreme_value(data, HEADER_MISS_DISTKILOMETERS, 'min')[0]


def common_orbit(data):
    """
    Returns a sorted dictionary when the keys have the orbit id and the value of each key is the number of asteroids
    that share the orbit id

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        dict: Dictionary of orbit IDs and their counts, sorted by orbit ID.
    """
    try:
        orbit_col = extract_column_by_header(data, 'Orbit ID')
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


def min_max_diameter(data):
    """
    Calculate the mean of the minimum and maximum estimated diameters.

    Parameters:
        data (np.ndarray): The combined array of headers and data.

    Returns:
        tuple: Mean of the minimum and maximum estimated diameters.
    """
    try:
        min_est_col = extract_column_by_header(data, 'Est Dia in KMmin')
        max_est_col = extract_column_by_header(data, 'Est Dia in KMmax')
        # Calculate and return means of both columns
        return np.mean(min_est_col), np.mean(max_est_col)
    except Exception as e:
        print(f"An error occurred while calculating min and max diameters: {e}")
        return None, None


def plt_pie_hazard(data):
    """
    Plot a pie chart showing the distribution of hazardous and non-hazardous asteroids.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
    """
    try:
        hazard_col = extract_column_by_header(data, 'Hazardous')
        # Count hazardous and non-hazardous asteroids
        sum_dangerous = np.sum(hazard_col)
        sum_not_dangerous = len(hazard_col) - sum_dangerous

        labels = ['Dangerous', 'Not Dangerous']
        values = [sum_dangerous, sum_not_dangerous]
        colors = ['red', 'green']

        # Plot pie chart
        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title("The distribution of dangerous asteroids")
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the pie chart: {e}")


def plt_liner_motion_magnitude(data):
    """
    Plot a linear regression of absolute magnitude vs miles per hour.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
    """
    try:
        magnitude_col = np.array(extract_column_by_header(data, 'Absolute Magnitude'), dtype=float)
        miles_per_hour_col = np.array(extract_column_by_header(data, 'Miles per hour'), dtype=float)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(magnitude_col, miles_per_hour_col)

        # Check if the p-value indicates a significant relationship (linear relationship)
        if p_value < 0.05:
            plt.scatter(magnitude_col, miles_per_hour_col)
            plt.plot(magnitude_col, slope * magnitude_col + intercept, color='red')
            plt.xlabel('Absolute Magnitude')
            plt.ylabel('Miles per hour')
            plt.title('Linear Relationship between Absolute Magnitude and Miles per hour')
            plt.show()
        else:
            print("No significant linear relationship between the absolute magnitude and speed of asteroids.")
    except Exception as e:
        print(f"An error occurred while plotting linear regression: {e}")


def find_extreme_value(data, column_header, operation='max'):
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

        name_col = extract_column_by_header(data, 'Name')
        return name_col[extreme_index], col[extreme_index]
    except Exception as e:
        print(f"An error occurred while finding the extreme value in column {column_header}: {e}")
        return None, None


def extract_column_by_header(data, target_header):
    """
    Extract a column from the data array based on the header.

    Parameters:
        data (np.ndarray): The combined array of headers and data.
        target_header (str): The header of the column to extract.

    Returns:
        np.ndarray: The extracted column.
    """
    try:
        headers = data[0]
        # Find index of the target header
        col_index = np.where(headers == target_header)[0][0]
        # Return the column values
        return data[1:, col_index]
    except IndexError:
        print(f"The header '{target_header}' was not found in the data.")
        return np.array([])
    except Exception as e:
        print(f"An error occurred while extracting column '{target_header}': {e}")
        return np.array([])


if __name__ == '__main__':
    # Load data from CSV file
    data1 = load_data("nasa.csv")
    if data1 is not None:
        # Perform plotting of linear regression and pie chart
        plt_liner_motion_magnitude(data1)
        plt_pie_hazard(data1)
