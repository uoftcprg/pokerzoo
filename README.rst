========
PokerZoo
========

PokerZoo is an open-source Python library for multi-agent reinforcement learning
environment for poker, developed by the University of Toronto Computer Poker
Research Group. PokerZoo supports an extensive array of poker variants and it
provides a flexible architecture for users to define their custom games. These
facilities are exposed via the PettingZoo multi-agent reinforcemnet learning
API. The library can be used for the development of AI poker agents through
various machine learning techniques, namely reinforcement learning. PokerZoo's
reliability and robustness is achieved by using the PokerKit library as the
backbone, whose reliability has been established through static type checking,
extensive doctests, and unit tests, achieving 99% code coverage.

Features
--------

* Implements PettingZoo multi-agent reinforcement learning API standard.
* Extensive poker game logic for major and minor poker variants.
* Customizable game states and parameters.
* Robust implementation with extensive unit tests and doctests.

Installation
------------

The PokerZoo library can be installed using pip:

.. code-block:: bash

   pip install pokerzoo

Usage
-----

TODO

.. code-block:: python

   ...

Testing and Validation
----------------------

PokerZoo has extensive test coverage, and has been validated through extensive
use in real-life scenarios.

Contributing
------------

Contributions are welcome! Please read our Contributing Guide for more
information.

License
-------

PokerZoo is distributed under the MIT license.

Citing
------

If you use PokerZoo in your research, please cite our library:

.. code-block:: bibtex

   @misc{pokerzoo,
     title={PokerZoo: An open-source Python library for multi-agent reinforcement learning environment for poker},
     author={Juho Kim},
     year={2023},
     url={https://github.com/uoftcprg/pokerzoo}
   }
