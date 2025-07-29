# Vortex Core Turbulence Analysis

### Written by Z. Starman 
  Author's contact Info: starman@uiowa.edu
  Additional PoC: frederick-stern@uiowa.edu
	
  Last Updated: July 29, 2025

## User Agreement and Approval for Usage

Users of this software are not allowed re-distribute the files of this repository (including all subdirectories) anywhere. Furthermore, usage of this software is only granted with the expressed approval of IIHR - Hydroscience & Engineering. Please contact Dr. Frederick Stern (frederick-stern@uiowa.edu) for approval.
 

## Description:
A Python code for extracting turbulence characteristics from the velocity time history at a vortex core. This program is designed to output Reynolds shear stresses, longitudinal autocorrelation function, macro and micro scales, and 1D turbulent energy spectra data.

## Notes:
This program may require the use of basic Python packages, such as NumPy, Pandas, SciPy, etc. These should be easily installed using the Pip package manager. A requirements.txt file has been provided. The simplest way to install the dependencies is to use pip. The command 'pip install -r requirements.txt' will install all the packages required to run this program. You can add these to a virtual environment for python if desired.

<br>

Additionally, this program was originally written to be run from the command line using the argument parser package for Python. The arguments can be viewed using '--help' on the command line. If it is desired for the code to be run from an IDE, please either modify the "if __name__ == "__main__" block to "hard" code the input parameters as necessary or set up the IDE to provide the CLI arguments. 

<br>

The data files passed into the program should be delimited data files similar to a CSV file, but there is no requirement for the files to be comma separated. The default behavior is to assume the files are delimited by a space. By default, the program assumes data files to be formatted with four columns corresponding to variables of Time, X velocity, Y velocity, and Z velocity, respectively. Please see the example data file (*example_data/simple_example/example.dat*). There are several optional command line arguments that can be configured for the program to run with files formatted differently than described above. Refer to the '--help' output and see the examples below to learn now to alter the program configuration. 

<br>

The output of the programs are written into the same directory as the data file. 


## Examples

The simplest example is to have a file formatted like *example_data/simple_example/example.dat*. The large eddy length scale was determined to be ~25 mm, and the velocity data is nondimensionalized. The time variable is already in dimensional units of seconds. The program can be configured and ran with the following command:
```
python run_analysis.py --files example_data/simple_example/example.dat --l0 0.025 --Us 1.531
```

Here the **--Us** argument is used to scale the velocities back to dimensional units of **m/s**. The $l_0$ value and Time variable was already in dimensional units, so the **--Ls** and **--Ts** arguments are not used. 

<br>

If the data file has no header (i.e. no column names at the top of the file) or the header line does not appear on the first line of the file, the **--header** argument can be used to specify which line to use or to turn off the header using -1. This is shown for the file *example_data/missing_headeer/no_header.dat*, which is the same data *example.dat*.
```
python run_analysis.py --files example_data/missing_header/no_header.dat --l0 0.025 --Us 1.531 --header -1
```

The **--vars** argument was not used as the order of the data columns matches the default/expected order of *Time, Vx, Vy, Vz*. 

<br>

If the data file is missing one of variables like Vy, Vz, or even Time, then these variables can be turned off by setting **--vars** argument appropriately. An example is provided using the *example_data/only_Vx_vel_component/vx_only.dat* data file.
```
python run_analysis.py --files example_data/only_Vx_vel_component/vx_only.dat --l0 0.030 --Us 1.531 --vars 0 1 -1 -1
```

Note that if the time variable is missing, a sampling frequency should be provided using the corresponding argument flag. Furthermore, if data is missing for the secondary velocity components, isotropic turublence conditions are used unless Reynold shear stress values are provided using the corresponding argument flag. Refer to '--help' for more details. 
