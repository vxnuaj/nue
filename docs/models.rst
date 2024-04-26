Models
==============

.. automodule:: models
   :members:
   :undoc-members:
   :show-inheritance:

.. important::
   Each model expects input data and target labels in the following shape:

   **Input Features**: (n, m), where n is the number of features and m is the number of samples.
   
   **Target Labels**: (1, m), where m is the number of samples.

.. _models.linreg:

models.linreg
--------------------

A base class for implementing linear regression!

.. automodule:: models.linreg
   :members:
   :undoc-members:
   :show-inheritance:

.. _models.logreg:

models.logreg
--------------------

A base class for implementing logistic regression!

.. automodule:: models.logreg
   :members:
   :undoc-members:
   :show-inheritance:


.. _models.nn:

models.nn
----------------

A base class for implementing a neural network for classification tasks!

.. important::

   This class is geared towards classification tasks and hasn't been tested for other applications.

   It's been validated on the MNIST Digits, Fashion MNIST and CIFAR-10!

   Implementation onto other datasets may not be guaranteed but feel free to experiment and contribute!

.. automodule:: models.nn
   :members:
   :undoc-members:
   :show-inheritance: