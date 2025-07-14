Removal operators
=================

-----------
Description
-----------

Removal operators destroy a solution by removing a set of customers from the solution. 
This page documents the different heuristics to do this.
The heuristics that are implemented are based on [WaSH24]_ and are the following:

- :ref:`random_removal`
- :ref:`random_route_removal`
- :ref:`shaw_removal`
- :ref:`worst_removal`
- :ref:`cost_reducing_removal`

The *Exchange reducing removal* heuristic, described in [WaSH24]_, is currently not implemented as it is not
trivial to adapt to the pickup and delivery problem.

The algorithms are implemented in ``lib/operators/removal.py`` 
and ``lib/operators/wang_operators.py``. All the algorithms go through the same procedure
when removing customers from the solution. The following figure shows the general flow chart
of the removal procedure:

.. image:: ../figures/removal_procedure.svg
   :width: 400px
   :align: center
   :alt: Removal procedure flow chart

.. _random_removal:

--------------
Random removal
--------------
.. autofunction:: lib.operators.destroy.random_removal

The following flow chart shows the *random removal* heuristic:

.. image:: ../figures/random_removal.svg
   :width: 400px
   :align: center
   :alt: Random removal flow chart

.. _random_route_removal:

--------------------
Random route removal
--------------------
.. autofunction:: lib.operators.destroy.random_route_removal

The following flow chart shows the *random route removal* heuristic:

.. image:: ../figures/random_route_removal.svg
   :width: 400px
   :align: center
   :alt: Random route removal flow chart

.. _shaw_removal:

------------
Shaw removal
------------
.. autofunction:: lib.operators.destroy.shaw_removal
   
**Notes on this algorithm**

The first seed customer is sampled from the list of planned customers only, 
instead of the union of planned and unassigned customers. This is because only the planned
customers have the earliest start times available. In the future, it would be interesting to
consider the unassigned customers as well.

Following the notation of reference [WaSH24]_, the relatedness function is given by formula (4) of reference [WaSH24]_:

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

The following flow chart shows the *Shaw removal* heuristic:


.. image:: ../figures/shaw-removal.svg
   :width: 400px
   :align: center
   :alt: Shaw removal flow chart

.. _worst_removal:

-------------
Worst removal
-------------

The worst removal heuristic is based on calculating a cost associated with each customer,
and removing the customer with the highest cost. In our case, **the cost is calculated as the 
increase in the objective function value when the customer is inserted in the arc it is already
currently in**. For example, if the considered customer is customer :math:`i` and which is 
currently between nodes :math:`(j, k)`, then the cost measure is given by:

.. math::

    \text{cost}(i) = d_{ji} + d_{ik} - d_{jk}

Where :math:`d_{ij}` is the distance between nodes :math:`i` and :math:`j`, and so on.

.. image:: ../figures/distance.svg
   :width: 200px
   :align: center
   :alt: Distances between nodes

.. autofunction:: lib.operators.destroy.worst_removal

The following flow chart shows the *worst removal* heuristic:

.. image:: ../figures/worst_removal.svg
   :width: 400px
   :align: center
   :alt: Worst removal flow chart

.. _cost_reducing_removal:

---------------------
Cost reducing removal
---------------------

.. autofunction:: lib.operators.destroy.cost_reducing_removal

The following **confusing** flow chart shows the cost reducing removal process, adapted to 
pickup and delivery problem. In this figure, *partner node* refers to the corresponding
pickup or delivery node of a customer. The decision block containing the condition
'Can *w* be inserted in some arch at lower cost in the same route?' hides a loop over either 
following nodes or preceding nodes of *v*, depending on whether *v* is a pickup or delivery node.
Furthermore, this check is not necessary if the partner node is in a valid position, because this
would already be a better solution than the starting one.

The following flow chart shows the *cost reducing removal* heuristic:

.. image:: ../figures/cost_reducing_removal.svg
   :width: 400px
   :align: center
   :alt: Cost reducing removal flow chart


.. toctree::
   :maxdepth: 2
