Experiments
===========

All experiments are inherited from the base class. Therefore, after 
initializing the corresponding experiment, you can start it by simply calling
the ``run`` function.

.. automethod:: dpeeg.exps.base.Experiment.run

.. currentmodule:: dpeeg.exps

Classification Experiments
------------------------------------

.. autosummary::
   :toctree: ../generated
   :nosignatures:
   :template: exps.rst

   HoldOut
   KFold
   LOSO_HoldOut
   LOSO_KFold