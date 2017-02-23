from stream2segment.io.db.pd_sql_utils import colnames
from stream2segment.io.db import models
from sqlalchemy.inspection import inspect
from collections import OrderedDict as odict

def get_doc(model_t, tablename="__tablename__"):

    def cols2list(columns, fkeys=set([])):
        lst = []
        for colname, col in columns.items():
            if colname in fkeys:
                continue
            typ = col.type
            try:
                pytyp = str(typ.python_type)
            except:
                pytyp = "no python type found"
            lst.append("%s: %s (SQL: %s)" % (colname, pytyp, str(typ)))
        return lst

    cname = model_t.__tablename__ if tablename == "__tablename__" else tablename
    res = inspect(model_t)
    fkeys = set((fk.parent for fk in model_t.__table__.foreign_keys))
    ret = odict()
    ret[cname] = cols2list(res.columns, fkeys)

    rels = res.relationships
    for rel in rels:
        relname = str(rel)[str(rel).find(".")+1:]  # note that if no dot, returns the string
        rside = rel.remote_side
        if len(rside) == 1:
            tbl = next(iter(rside)).table
            cols = tbl.columns
            ret[relname] = cols2list(cols)


    finallst = []
    for i, k in enumerate(ret.iterkeys()):
        # first element is the table and its "scalar" attributes:
        if i > 0:
            finallst.append("")
        finallst.append("\n".join((k + "." + l for l in ret[k])))
    return "\n".join(finallst)
        
        
        
        
if __name__ == "__main__":
    print get_doc(models.Segment, "segment")