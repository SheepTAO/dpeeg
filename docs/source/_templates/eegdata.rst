{{ name }}
{{ underline }}

.. autoclass:: {{ fullname }}
    :members:
    :exclude-members: {% for item in attributes %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}

    {% block methods %}
    {% if methods %}
    .. rubric:: Methods

    .. autosummary::
        :nosignatures:
        {% for item in methods %}
        ~{{ name }}.{{ item }}
        {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
        {% for item in attributes %}
        ~{{ name }}.{{ item }}
        {%- endfor %}
    {% endif %}
    {% endblock %}