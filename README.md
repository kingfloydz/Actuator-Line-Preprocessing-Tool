# Actuator Line Preprocessing Tool

This is a comprehensive tool for propeller aerodynamic analysis. It can process 3D models (STL/OBJ), extract airfoil sections from blades, calculate aerodynamic parameters, perform aerodynamic analysis using XFOIL, and extrapolate aerodynamic data to the full range of angles of attack (360 degrees) using the Viterna method.

## Features

* **3D Model Processing**: Supports import of propeller blade models in STL and OBJ formats. Input the axial direction, and the tool automatically identifies the spanwise direction and separates blades.
* **Automatic Section Extraction**: Automatically identifies blade geometric features, slices along the spanwise direction, and extracts airfoil sections.
* **Airfoil Generation**: Converts extracted section geometry into XFOIL-compatible coordinate format, including smoothing and reconstruction functions.
* **Aerodynamic Parameter Calculation**: Calculates aerodynamic parameters such as Reynolds number for each section based on operating conditions (rotational speed, inflow velocity, etc.).
* **XFOIL Integrated Analysis**: Built-in XFOIL core (via `xfoil_core`), automatically performs batch aerodynamic analysis for multiple sections and multiple angles of attack.
* **Data Extrapolation**: Uses the Viterna method to extrapolate XFOIL-calculated data from a limited angle of attack range to the full range of -180° to +180°.
* **Parallel Processing**: Supports multi-process parallel computing to improve analysis efficiency.

## Project Structure

* `main.py`: Main entry point of the program, coordinating the entire analysis workflow.
* `parameter_manager.py`: Parameter management module, responsible for reading configurations (e.g, `parameter.dat`) and managing file paths.
* `split_blades.py`: 3D model analysis module, responsible for reading propeller model files and identifying/separating blades.
* `geometry.py`: Geometric analysis module, responsible for calculating blade span length, slicing sections, etc.
* `airfoil_generator.py`: Airfoil generation module, responsible for converting geometric sections into airfoil coordinate point files.
* `airfoil_aerodynamics.py`: Aerodynamic calculation module, responsible for calculating Reynolds number, tangential velocity, etc.
* `xfoil_analyzer.py`: XFOIL analysis encapsulation module, responsible for calling the XFOIL core for calculations.
* `viterna_extrapolation.py`: Data extrapolation module, implementing the Viterna extrapolation algorithm.
* `airfoil_analysis.py`: Contains advanced airfoil solver logic to assist in aerodynamic analysis.
* `xfoil_core/`: Contains the core Fortran code of XFOIL and its Python wrapper.

## Environment Dependencies

Please ensure the following Python libraries are installed:

* numpy
* scipy
* pandas
* numpy-stl
* logging

## Usage

1. **Prepare Model and Configuration**:

   * Prepare the 3D model file of the propeller (STL or OBJ).
   * Configure the `parameter.dat` file (the program will automatically read it; ensure the path is correct), set the axial direction, rotational speed, fluid density, viscosity, inflow velocity, and other parameters.  
     - `num_sections`: Number of sections along the spanwise direction for analysis (since some sections may diverge at large angles of attack, the final number of successful sections will be less than this value).  
     - `section_start_ratio`: Relative position along the spanwise direction where the first section is located, starting from the root.  
     - `section_end_ratio`: Relative position along the spanwise direction where the last section is located, starting from the root.  
     - `max_processes`: Maximum number of parallel threads for XFOIL analysis.

2. **Run the Program**:
   Run `main.py` in the terminal:

   ```bash
   python main.py
   ```

3. **View Results**:
   The program will generate a log file `process_log.txt` during operation.  
   Analysis results (airfoil files, aerodynamic data) will be saved in the `Temp` directory. Please refer to the "Output File Description" section below for detailed instructions.

## Output File Description

After the program runs, the following files will be generated in the `Temp` directory:

1. **Airfoil Coordinate Files**:

   * `airfoil_blade{i}_section{j}.dat`: Smoothed airfoil coordinate file (XFOIL format) for the `j`-th section of the `i`-th blade.
   * `airfoil_blade{i}_section{j}_raw.dat`: Corresponding raw section coordinate file.

2. **Geometric Information File**:

   * `section_geometry.txt`: Contains a geometric information table of all valid sections, including section number, spatial position (X, Y, Z), chord length, and stagger angle.

3. **Aerodynamic Data Files (OpenFOAM Format)**:

   * `airfoilProperties_section{j}`: Full-range angle of attack aerodynamic data (Cl, Cd) for the `j`-th section, in a format compatible with OpenFOAM.

4. **Aerodynamic Data Summary (Excel)**:

   * `airfoil_results.xlsx`: Summary Excel spreadsheet containing full-range angle of attack aerodynamic data for all sections.

5. **Log File**:

   * `process_log.txt`: Records detailed log information during the program's operation.

## Workflow Description

1. **Initialization**: Load parameters and clean the working directory.
2. **Model Analysis**: Read the 3D model and identify the spanwise direction and blade structure.
3. **Geometric Processing**: Slice the blade along the spanwise direction and extract geometric data of multiple sections.
4. **Airfoil Generation**: Smooth the extracted section points and generate standard airfoil coordinate files.
5. **Aerodynamic Calculation**: Calculate the Reynolds number for each section under specific operating conditions.
6. **XFOIL Analysis**: Call XFOIL to calculate lift coefficient (Cl) and drag coefficient (Cd) of each section at different angles of attack.
7. **Data Extrapolation**: Perform Viterna extrapolation for well-converged sections in the high angle of attack range to generate aerodynamic data tables for the full range of angles of attack.

## Notes

* Testing shows that wind turbine blades yield good convergence results, while multi-rotor blades perform less well.
* Reference for large-range angle of attack extrapolation method: [A generalized method to extend airfoil polars over the full range of angles of attack - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0960148120304833)
* Reference for maximum drag coefficient calculation method: [A simple method to estimate the airfoil maximum drag coefficient - IOPscience](https://iopscience.iop.org/article/10.1088/1742-6596/1618/5/052068)
