Initial Solution
================

-----------------------
CVRPTW initial solution
-----------------------

This is an implementation of the nearest neighbour solution for standard Capacitated Vehicle Routing Problem.


.. autofunction:: lib.initial_solutions.initial_solutions.nearest_neighbor_tw

-------------------------
CVRPPDTW initial solution
-------------------------

The provided initial solution is an extension of the nearest neighbour solution
for the standard Capacitated Vehicle Routing Problem. 
This heuristic has been modified to work with Pick up and Delivery problem instances, 
therefore dealing with two time windows and two locations for each customer. 
The only hyperparameter that has to be set is ``n``, representing the *number of new possible customers
that are considered at each step*. A simplified diagram of this algorithm is shown below.
Note that, even if it is not shown in the diagram, adding a node to the route means:
 
- Removing the node from the list of possible customers
- Calculating the planned arrival time (#TODO: or times?) and adding it to the ``route_schedule`` vector
- Calculating the new load of the vehicle and updating the ``vehicle_load`` variable

The algorithm is implemented in the following function:

.. autofunction:: lib.initial_solutions.initial_solutions.nearest_neighbor_pdtw


The following diagram explains the workflow of ``nearest_neighbor_pdtw``

.. image:: ../figures/heuristic_nn.drawio.svg
   :width: 400px
   :align: center
   :alt: Nearest Neighbour Heuristic for CVRPPDTW

Notes:

- "*Serve all remaining nodes*" before the last decision block should also
  check that delivery nodes are served after pickup nodes
- This algorithm is needed to create a first solution which we expect
  to drastically change, so not too much attention and energy should 
  be focused on it.


.. toctree::
   :maxdepth: 2
