# Dial-a-ride


Python educational implementation of [*An adaptive large neighborhood search for the multi-depot dynamic vehicle routing problem with time windows*](https://www.sciencedirect.com/science/article/abs/pii/S0360835224002432) by Wang et al., 2024. 

## Introduction
The Vehicle Routing Problem (VRP) is a classic combinatorial optimization problem where the objective is to determine optimal routes for a fleet of vehicles to serve a set of customers, subject to various constraints. VRP has numerous real-world applications, including logistics, transportation, and delivery services.

This project extends the standard VRP to include additional complexities, addressing the **Multi-Depot Dynamic Vehicle Routing Problem with Time Windows (MDVRPTW)**. It incorporates the following features:

### Features and Constraints
- **Multi-Depot Configuration**: Vehicles are dispatched from multiple depots, and the solution must assign each customer to a specific depot and vehicle.

- **Dynamic Requests**: Customer requests are not fully known in advance and can arrive dynamically over time, requiring real-time adjustments to the routing plan.

- **Time Windows**: Each customer must be served within a specific time window, adding a temporal constraint to the routing problem.

- **Dial-a-Ride**: Requests involve both pickup and delivery locations, making it necessary to schedule both tasks within the constraints of time windows.

- **Capacity Constraints**: Each vehicle has a limited carrying capacity, which must not be exceeded.

- **Objective**: The goal is to minimize the total operational cost, which includes travel distances and the number of vehicles used, while ensuring timely and efficient service.


This project uses **Adaptive Large Neighborhood Search (ALNS)**, a metaheuristic approach that combines global and local search strategies, to efficiently solve the MDVRPTW problem. The well-known [ALNS](https://alns.readthedocs.io/en/latest/) Python library is used for the algorithms.

## How to set up environment
This code has been implemented using Python 3.12.2 and a **modified version of ALNS library**, which can be obtained with 'requirements.txt' file. For set up, ```conda``` is recommended while ```git``` is necessary. If ```conda``` is not used, skip the next two commands.
First, create a new conda environment with Python 3.12.2
```
conda create -n myenv python=3.12.2
conda activate myenv
``` 

Clone this repository and enter it
```
git clone git@github.com:pettepiero/Dial-a-ride.git
cd Dial-a-ride
```

To install requirements, run 
```
pip install -r requirements.txt
```

## VRP variants
The following different variants of VRP are implemented and solved:

- CVRPTW
- CVRPPDTW

The code to run them is in the `cvrptw` and `cvrppdtw` repositories.

## How to execute the code
An instance of the static problem can be obtained by executing cvrptw.py for CVRPTW problem, or analogously for the other instances. The command can be followed by the desired command line options. A description of these options is available running: 

```
python cvrptw.py --help
```
Currently, the newest implementation of the article [(Wang et al., 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0360835224002432) is in `cvrptw.py`. This should be ready to run with the mentioned requirements. A more experimental notebook is `wang-2024.ipynb`, which produces useful plots on the statistics of the operators.

The notebook can also be run on colab, without installing anything through [this link](https://colab.research.google.com/github/pettepiero/Dial-a-ride/blob/main/wang-2024.ipynb). To execute it, uncomment the first cell and set the recommended data_file path in cell before *Solution state*. **Note:** This notebook might produce some errors on `plot_solution function`.

## Plots and videos
Running `cvrptw/wang-2024.ipynb` will save plots inside `outputs/plots/[foldername]` where `foldername` is given by the year, month, day, hour, minute and seconds of execution. Therefore, **it is recommended to remove these folders to save space after some executions**.

To generate a video of the solution, copy the video output folder that is printed after the `iterate` method. Then execute 

```
python cvrptw/generate_video.py --image_folder=<copied-path> 
```
and optionally specify a video output name with argument `--video_name`. The video will be saved in `outputs/videos` folder.

## Data notation
The dataset convention that is used in this project is that of [**Cordeau**](https://www.bernabe.dorronsoro.es/vrp/index.html?/Problem_Instances/CVRPTWInstances.html) for CVRPTW variant and [**Reinelt**](https://pubsonline.informs.org/doi/10.1287/ijoc.3.4.376) for the CVRPPDTW variant.
