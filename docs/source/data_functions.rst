Data module functions
=====================

-----------
Description
-----------
Description for the following functions:

- :ref:`calculate_depots`
- :ref:`read_cordeau_data`



.. _calculate_depots:
----------------
calculate_depots
----------------
.. autofunction:: cvrptw.myvrplib.data_module.calculate_depots


.. _read_cordeau_data:
-----------------
read_cordeau_data
-----------------
.. autofunction:: cvrptw.myvrplib.data_module.read_cordeau_data

    Notes
    -----
    The input file must conform to the format specified in Cordeau et al. (2001). Specifically:

    - The first line contains:
        ``type m n t`` where:

            - ``m``: number of vehicles
            - ``n``: number of customers
            - ``t``: number of depots

    - Coordinate and demand vectors
    - Followed by `t` lines of:
        ``D Q`` (maximum route duration, vehicle capacity)
    - Followed by `n` customer lines in format ``i x y d q f a list e l``
        - ``i``: customer number
        - ``x``: x coordinate
        - ``y``: y coordinate
        - ``d``: service duration
        - ``q``: demand
        - ``f``: frequency of visit
        - ``a``: number of possible visit combinations
        - ``list``: list of all possible visit combinations
        - ``e``: beginning of time window
        - ``l``: end of time window
    - Followed by `t` depot lines in the same format as customers.

    Visit combinations (`list`) and frequency of visit are ignored in this implementation.




.. toctree::
   :maxdepth: 2
