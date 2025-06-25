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
are based on [WaSH24]_ and are the following:

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

.. figure:: ../figures/best_insert.svg
   :width: 400px
   :align: center
   :alt: Best insertion flow chart

   Best insertion flow chart

At the moment, *greedy repair* finds best positions for pickup nodes of unassigned customers, 
and then proceeds to find a feasible position for the delivery node. A possible improvement 
would be to find the best position for both pickup and delivery nodes.

.. autofunction:: cvrptw.operators.repair.greedy_repair_tw

The following flow chart shows the *greedy repair* heuristic:

.. _greedy-repair-flow:
.. figure:: ../figures/greedy_repair.svg
   :width: 400px
   :align: center
   :alt: Greedy repair flow chart
   
   Greedy repair flow chart

.. _wang_repair:

------------------
Wang greedy repair
------------------
*Wang greedy repair* denotes a repair operator based on [WaSH24]_. The algorithm is similar to
the *greedy repair* algorithm, but it uses a different heuristic to find the best position to
insert the nodes of a customer. In particular, the insertion checks if the nodes can be 
inserted in the route using the following time window compatibility check:

.. math::

      \text{max} \left\{ e_\mu (L) + t_{i_\mu c}, a_c  \right\} \leq 
      \text{min} \left\{ l_{\mu +1}(L) - t_{ci_{\mu +1}}, b_c \right\}

where :math:`c` is the node that we are checking to insert in arc :math:`(i_{\mu}, i_{\mu +1})` 
of route *r*, :math:`0 \leq \mu \leq u`, and :math:`u` is the number of nodes in route *r*.
Furthermore, :math:`e_\mu (L)` and :math:`l_{\mu +1}(L)` are the earliest and latest start
times at node :math:`i_{\mu}` and :math:`i_{\mu +1}` respectively, and :math:`t_{i_\mu c}` and
:math:`t_{ci_{\mu +1}}` are the travel times between nodes :math:`i_{\mu}` and :math:`c`, and
between :math:`c` and :math:`i_{\mu +1}` respectively. Finally, :math:`a_c` and :math:`b_c` 
are the earliest arrival and latest departure times requested by :math:`c` respectively.
If the above condition is verified, then node :math:`c` can be inserted in the arc.

It is recommended to read [WaSH24]_ for a better intuition of the heuristic, based on formula
(15) of the paper. The flow chart of *Wang greedy repair* is identical to 
:ref:`greedy-repair-flow`, since only the check of best solution is different.



.. toctree::
   :maxdepth: 2