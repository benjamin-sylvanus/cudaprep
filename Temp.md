## Reader needs map implementation

Change dt of Data Members of Simulation class.
New dt should be Variant<T> \*
Reader can then store all data in map and dynamically assign values in constructor

Or we can get rid of the simulation class def and point to the map keys

## Verify Cuda Unified memory Model works on server.

Clone branch and run.

If it works rebase main onto this branch

Could perform benchmarking or profiling to compare performance.

## Figure out gpuErrchk on this branch

## Try installing qt app on the server and running with mapped model.

- Develop a new class named `NewSimReader` to manage the reading of the JSON configuration and binary files.
- In the `NewSimReader` class, implement functions to:
  - Read and parse the JSON file for variable definitions.
  - Read the binary file and extract data based on the JSON definitions.
- Independently test the `NewSimReader` class to confirm accurate reading and data extraction from the JSON and binary files.
- In the simulation class, perform the following:
  - Introduce a constructor that takes a `NewSimReader` instance.
  - Adjust member variables to align with the `NewSimReader` data structure.
  - Revise getter and setter methods to accommodate the new member variables.
- Validate the revised simulation class to ensure proper data storage and access using the `NewSimReader`.
- In the controller class, update the following:
  - Alter the Setup function to utilize the `NewSimReader`.
  - Amend the handlecommand function for any new or changed commands due to the data structure update.
- Conduct tests on the modified controller class to verify correct simulation initialization and command handling with the `NewSimReader`.
- For data extraction and usage refactoring:
  - Locate all code segments where simulation data is being extracted and utilized.
  - Modify these segments to be compatible with the new data structures from `NewSimReader` and simulation classes.
- Test the altered code to confirm it operates correctly with the new data structures.
- After thorough testing and validation of the new implementation:
  - Eliminate the old `simreader` class and any related obsolete code.
  - Update any remaining code references to the old `simreader` class.
