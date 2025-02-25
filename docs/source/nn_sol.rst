Initial Solution
================

-----------
Description
-----------
The provided initial solution is an extension of the nearest neighbour solution. 
This heuristic has been modified to work with the pick up and delivery problem, therefore 
dealing with two time windows and two locations for each customer. 
The only hyperparameter that has to be set is ``n``, representing the *number of new possible customers
that are considered at each step*. A simplified diagram of this algorithm is shown below.
Note that, even if it is not shown in the diagram, adding a node to the route means:
 
- Removing the node from the list of possible customers
- Calculating the planned arrival time and adding it to the ``route_schedule`` vector
- Calculating the new load of the vehicle and updating the ``vehicle_load`` variable

The algorithm is implemented in ``nearest_neighbor`` method of ``cvrptw/initial_solutions/initial_solution.py``.

.. image:: ../figures/heuristic_nn.drawio.svg
   :width: 400px
   :align: center
   :alt: Nearest Neighbour Heuristic

