Removal operators
=================

.. _introduction:
.. _vrp:
.. _installation:
.. _nn_sol:
.. _removal_ops:

-----------
Description
-----------

Removal operators destroy a solution by removing a set of customers from the solution. 
This page documents the different heuristics to do this.

The algorithms are implemented in ``cvrptw/operators/removal.py`` 
and ``cvrptw/operators/wang_operators.py``. All the algorithms go through the same procedure
when removing customers from the solution. The following figure shows the general flow chart
of the removal procedure:

.. image:: ../figures/removal_procedure.svg
   :width: 400px
   :align: center
   :alt: Removal procedure flow chart

--------------
Random removal
--------------
.. autofunction:: cvrptw.operators.destroy.random_removal

The following flow chart shows the random removal heuristic:

.. image:: ../figures/random_removal.svg
   :width: 400px
   :align: center
   :alt: Random removal flow chart


--------------------
Random route removal
--------------------
.. autofunction:: cvrptw.operators.destroy.random_route_removal

The following flow chart shows the random route removal heuristic:

.. image:: ../figures/random_route_removal.svg
   :width: 400px
   :align: center
   :alt: Random route removal flow chart

------------
Shaw removal
------------
.. autofunction:: cvrptw.operators.destroy.shaw_removal
   
**Notes on this algorithm**

The first seed customer is sampled from the list of planned customers only, 
instead of the union of planned and unassigned customers. This is because only the planned
customers have the earliest start times available. In the future, it would be interesting to
consider the unassigned customers as well.

Following the notation of reference [WaSH24], the relatedness function is given by formula (4) of reference [WaSH24]:

.. math::

    \text{relatedness}(i, j) = \alpha_1 \frac{d_{ij}}{d^{max}} + \alpha_2 \frac{|e_i - e_j|}{H} + \alpha_3 \frac{|q_i-q_j|}{q^{max}}

where :math:`d^{max} = \text{max}\left\{ d_{i,j}:i,j \in V_0^p \right\}` is the maximum 
travel distance between any two customers in the set of planned customers
:math:`V_0^p`, :math:`H` is the time horizon of the problem, and 
:math:`q^{max} = \text{max} \left\{ q_i :i \in V_0^p \right\}` is the maximum demand among all
customers :math:`i` in the set of planned customers :math:`V_0^p`.
For the pickup and delivery problem, the equation above is evaluated first for the pickup
nodes, then for the delivery nodes, and the two values are summed up. The customer *j* that
is removed is the one with the smallest sum.

The following flow chart shows the random route removal heuristic:


.. image:: ../figures/shaw-removal.svg
   :width: 400px
   :align: center
   :alt: Shaw removal flow chart


---------------------
Cost reducing removal
---------------------

.. autofunction:: cvrptw.operators.destroy.cost_reducing_removal

The following **confusing** flow chart shows the cost reducing removal process, adapted to 
pickup and delivery problem. In this figure, *partner node* refers to the corresponding
pickup or delivery node of a customer. The decision block containing the condition
'Can *w* be inserted in some arch at lower cost in the same route?' hides a loop over either 
following nodes or preceding nodes of *v*, depending on whether *v* is a pickup or delivery node.
Furthermore, this check is not necessary if the partner node is in a valid position, because this
would already be a better solution than the starting one.

.. image:: ../figures/cost_reducing_removal.svg
   :width: 400px
   :align: center
   :alt: Cost reducing removal flow chart

