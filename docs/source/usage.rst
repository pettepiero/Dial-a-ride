Installation
============

----------------------
Set up the environment
----------------------
The project is available at the following repository: `dial-a-ride <https://github.com/pettepiero/Dial-a-ride>`_.

This code has been implemented using Python 3.12.2 and a modified version of 
`ALNS library <https://alns.readthedocs.io/en/latest/>`_
which can be obtained with the ``requirements.txt`` file. For environment set up, 
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_ is 
recommended while `git <https://git-scm.com/>`_ is necessary. A complete list of the
requirements is therefore available reading the ``requirements.txt`` file.

If conda is not used, skip the next two commands. First, create a new conda environment 
with Python 3.12.2. 

.. code-block:: bash

    conda create -n myenv python=3.12.2
    conda activate myenv

Clone the repository and enter it

.. code-block:: bash

    git clone git@github.com:pettepiero/Dial-a-ride.git
    cd Dial-a-ride

To install requirements, run

.. code-block:: bash

    pip install -r requirements.txt


-----------------------
Build the documentation
-----------------------
This documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ at 
version v8.1.3. To build the documentation, `make <https://www.gnu.org/software/make/>`_ 
has to be installed and then it is sufficient to run the following commands:

.. code-block:: bash

    cd docs
    make html

The documentation will be available in the ``docs/build/html`` folder. Open the
``index.html`` file in a browser to view the documentation.