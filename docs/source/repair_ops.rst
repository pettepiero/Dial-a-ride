.. _repair_ops:
=================
Repair operators
=================

-----------
Description
-----------

Repair operators are used to repair a solution by adding a set 
of customers to the routes of the solution. Different heuristics 
are implemented to do this. The heuristics that are implemented 
are based on :ref:`WaSH24` and are the following:

- :ref:`greedy_repair`


The repair algorithms are implemented in ``cvrptw/operators/repair.py`` and 
``cvrptw/operators/wang_operators.py``.

.. _greedy_repair:
-------------
Greedy repair
-------------
Greedy repair iterates over a shuffled version of unassigned customers and tries to insert them
into the routes of the solution. The insertion is done by selecting the best position in the 
current solution routes. This means that the method is defined together with a *best insertion*
heuristic. In our case, the *best insertion* is the one that:

- Does not exceed the vehicle capacity
- Does not violate the time window constraints
- Minimizes the total cost of the solution

The following flow chart explicits the *best insertion* heuristic that lies at the heart 
of the *greedy repair* algorithm:

.. image:: ../figures/best_insert.svg
   :width: 400px
   :align: center
   :alt: Best insertion flow chart

At the moment, *greedy repair* finds best positions for pickup nodes of unassigned customers, 
and then proceeds to find a feasible position for the delivery node. A possible improvement 
would be to find the best position for both pickup and delivery nodes.

.. autofunction:: cvrptw.operators.repair.greedy_repair_tw

The following flow chart shows the *greedy repair* heuristic:

.. image:: ../figures/greedy_repair.svg
   :width: 400px
   :align: center
   :alt: Greedy repair flow chart
