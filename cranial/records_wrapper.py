"""
functions to deal with the need for models to operate on dictionaries only
"""


def wrap_apply_to_fields(fn, in_field, out_field):
    """
    Modifies a given function that performs transformation "in_data -> out_data" to the one that
    performs "{in_field: in_data} -> {in_field: in_data, out_field: out_data}"

    Parameters
    ----------
    fn
        original function

    in_field
        field of an input dictionary where in_data will be

    out_field
        field of an output dictionary where to put transformed data

    Returns
    -------
        modified function
    """

    def fn_wrapped(record):
        if in_field in record.keys():
            record[out_field] = fn(record[in_field])
        return record

    return fn_wrapped


def get_copy_fields_fn(fields, defaults=None):
    """
    creates a function to use in map that copies incoming objects but leaves only certain fields
    Parameters
    ----------
    fields
        list of fields to copy
    defaults
        if field is emty what default value to use, can be None, a single value, or a dictionary of
        values for each field in fields
    Returns
    -------
        function to copy objects
    """
    if isinstance(fields, list):
        # convert to dictionary
        fields = {f: f for f in fields}

    if defaults is None:
        defaults = {f: None for f in fields.keys()}
    elif not isinstance(defaults, dict):
        defaults = {f: defaults for f in fields.keys()}
    else:
        defaults = {f: defaults.get(f) for f in fields.keys()}

    def copy_fields_fn(obj):
        """
        copy a subset of fields, optionally rename and input defaults for missing values
        """
        return {out_f: obj.get(in_f, defaults[in_f]) for in_f, out_f in fields.items()}

    return copy_fields_fn