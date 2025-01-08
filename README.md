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
This code has been implemented using Python 3.12.2 and a **modified version of ALNS library**, which can be obtained with 'requirements.txt' file. First, create a new conda environment with Python 3.12.2
```
conda create -n myenv python=3.12.2
``` 
To install requirements, run 
```
pip install -r requirements.txt
```

## How to execute the code
Currently, the newest implementation of the article [(Wang et al., 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0360835224002432) is in `cvrptw/wang-2024.ipynb`. This should be ready to run with the mentioned requirements.#

## Plots and videos
Running `cvrptw/wang-2024.ipynb` will save plots inside `cvrptw/plots/[foldername]` where `foldername` is given by the year, month, day, hour, minute and seconds of execution. Therefore, **it is recommended to remove these folders to save space after some executions**.

To generate a video of the solution, copy the video output folder that is printed after the `iterate` method. Then execute 

```
python cvrptw/generate_video.py --image_folder=<copied-path> 
```
and optionally specify a video output name with argument `--video_name`. The video will be saved in `cvrptw/videos` folder.