Initial Solution
================

For the NLNS project [HoTi20]_ adapted to Multi depot case, see section :ref:`NLNS MD initial solution`

-----------
Description
-----------
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

The algorithm is implemented in ``nearest_neighbor`` method of ``lib.initial_solutions/initial_solution.py``.

.. image:: figures/heuristic_nn.drawio.svg
   :width: 400px
   :align: center
   :alt: Nearest Neighbour Heuristic

Notes:

- "*Serve all remaining nodes*" before the last decision block should also
  check that delivery nodes are served after pickup nodes
- This algorithm is needed to create a first solution which we expect
  to drastically change, so not too much attention and energy should 
  be focused on it.


.. _NLNS MD initial solution:
------------------------------------
NLNS to Multi Depot initial solution
------------------------------------
The simple greedy procedure proposed by [HoTi20]_ can be adapted to MD as follows:

- A clustering algorithm creates a cluster of customer for each depot, based
  on the locations.
- Inside each cluster, the greedy initial solution proposed by [HoTi20]_ is
  applied.
- For a cluster, the first route starts from the depot and is created by always
  adding the closest customer to a tour.
- A new route is created if the demand of a candidate customers exceeds the 
  remaining load of the vehicle. In this case the old route returns to the depot.
- The process is repeated until all customers have been visited.

.. toctree::
   :maxdepth: 2
