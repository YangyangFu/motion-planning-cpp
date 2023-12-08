# Route Planning Project

This repo contains the code for the Route Planning project.

<img src="map.png" width="600" height="450" />

## Local Environment Setup
This repo provides a self-conatined Dockefile to setup the local environment. To setup the environment, run the following command from the `.devcontainer` directory:
```
make build
```

This will be a docker image in your local machine named `cpp-route-panning:latest`.  
To build and run the project, we need run the image as a devcontainer in VSCode. 

## Compiling and Running

### Compiling
To compile the project, first, create a `build` directory and change to that directory:
```
mkdir build && cd build
```
From within the `build` directory, then run `cmake` and `make` as follows:
```
cmake ..
make
```
### Running
The executable will be placed in the `build` directory. From within `build`, you can run the project as follows:
```
./OSM_A_star_search
```
Or to specify a map file:
```
./OSM_A_star_search -f ../<your_osm_file.osm>
```

## Testing

The testing executable is also placed in the `build` directory. From within `build`, you can run the unit tests as follows:
```
./test
```
