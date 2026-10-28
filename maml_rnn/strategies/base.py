class Strategy:
    """A training strategy.

    ``params`` is the ``strategies.<name>`` section of the config and ``build_fn`` a
    zero-argument function returning a fresh, uncompiled model of the architecture being
    trained (used e.g. by MAML for its shadow model).

    ``fit`` trains ``model`` in place and returns a history ``{curve_name: [values per epoch]}``.
    """

    name = None

    def __init__(self, params, build_fn, verbose=1):
        self.params = dict(params)
        self.build_fn = build_fn
        self.verbose = verbose

    def fit(self, model, data):
        raise NotImplementedError
