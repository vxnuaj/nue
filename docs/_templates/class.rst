.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: model

   .. rubric:: Other Methods

   {% for member in members if member.name != 'model' -%}
   .. automethod:: {{ member.name }}
   {% endfor %}
