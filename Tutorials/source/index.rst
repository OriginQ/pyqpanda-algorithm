.. pyqpanda algorithm documentation master file, created by
   sphinx-quickstart on Tue Jan 22 14:31:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyqpanda algorithm
====================================

**A Quantum Algorithm Development and Runtime Environment Kit, based on pyqpanda3**

The **pyqpanda_alg** is a collection of fundamental quantum algorithms and functions that are commonly used in developer's quantum algorithm.

Overall, it provides a standardized set of tools for developers, allowing them to write quantum programs that run locally on pyqpanda3's CPU simulator and, through the optional qpanda3-runtime backend of the ``pyqpanda_alg.execution`` layer, to submit them to a remote qpanda3-runtime service. It is an important resource for the development of quantum software and the advancement of quantum computing research.

.. toctree::
    :maxdepth: 2
    :caption: Introduction

    GettingStarted

.. toctree::
    :maxdepth: 2
    :caption: Execution Layer

    execution
    runtime_algorithms

.. toctree::
    :maxdepth: 2
    :caption: Algorithms

    VQE
    HHL
    Shor

.. toctree::
    :maxdepth: 2
    :caption: Changelog

    Changelog

.. toctree::
    :caption: API Reference
    :maxdepth: 2

    autoapi/index
