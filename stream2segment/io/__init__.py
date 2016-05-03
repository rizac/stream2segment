class Session(object):

    def __init__(self, sql_alchemy_engine):
        self._engine = sql_alchemy_engine

    def __enter__(self):
        self._temporary_session = Session(self._engine)
        return self.cr

    def __exit__(self, type_, value, traceback):
        method_ = self._temporary_session.commit if \
            value is None else self._temporary_session.rollback
        try:
            method_()
        except:
            try:
                self._temporary_session.close()
            except:
                pass
