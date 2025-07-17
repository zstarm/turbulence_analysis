# Vortex Core Turbulence Analysis

### Written by Z. Starman 
 Contact Info: starman@uiowa.edu
 
### July 17, 2025

## Description:
A Python code for extracting turbulence characteristics from the velocity time series at a vortex core. 

## Notes:
This program may require the use of basic Python packages, such as NumPy, Pandas, SciPy, etc. These should be easily installed using the Pip package manager. 

Additionally, this program was originally written to be run from the command line using the argument parser package for Python. The arguments can be viewed using '--help' on the command line. If it is desired for the code to be run from an IDE, please modify the "if __name__ == "__main__" block as necessary or set up the IDE to provide the CLI arguments. 

The data files passed into the program should be delimited data files with columns of Time, X velocity, Y velocity, and Z velocity. Please see the example data file (example_data/example.dat). The default assumptions are for the files to be delimited by spaces and columns to be in the following order: "Time", "Vx", "Vy", "Vz". However, the function arguments inside the code can be modified if your data files are not formatted in this manner. Please look through the function descriptions if this is necessary.
