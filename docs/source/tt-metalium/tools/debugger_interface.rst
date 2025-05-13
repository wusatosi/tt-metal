Debugger Interface
==================

Overview
--------

The debugger interface provides a standardized way for debugging tools to communicate with and control tt-metalium
during program execution:

- It saves the mapping between kernel IDs and source files. The mapping is stored in a file called `runtime_data.yaml`
in the `generated/silicon_debugger` directory.

Enabling
--------

Enable the Watcher by setting the following environment variables:

.. code-block::

   export TT_METAL_DEBUGGER_INTERFACE=1

After starting the program, the debugger interface will be enabled:

- Kernel mapping is saved in the `runtime_data.yaml` file.

Details
-------

TODO: How to use the tt-triage script
