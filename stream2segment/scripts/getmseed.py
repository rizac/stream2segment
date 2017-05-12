'''
Created on Mar 29, 2017

@author: riccardo
'''
from stream2segment.utils import get_session
from obspy.core.stream import read
from cStringIO import StringIO
from sqlalchemy.orm import load_only
from os.path import join, isfile
import argparse
from click.termui import progressbar
from stream2segment.io.db.models import Segment
# from stream2segment.process.utils import segstr


def load(dburl, out, ids):
    if not ids:
        return 0, 0, 0
    segs = get_session(dburl).query(Segment).filter(Segment.id.in_(ids)).\
        options(load_only(Segment.id, Segment.data))
    saved = 0
    ov = 0
    total = segs.count()
    with progressbar(length=total) as pbar:
        for s in segs:
            pbar.update(1)
            try:
                if not s.data:
                    raise ValueError('')  # for safety ...
                stream = read(StringIO(s.data))
                fout = join(out, "%s__%s__%s__%s.mseed" % (stream[0].get_id(),
                                                           stream[0].stats.starttime.isoformat(),
                                                           stream[0].stats.endtime.isoformat(),
                                                           s.id))
                if isfile(fout):
                    ov += 1
                stream.write(fout, format="MSEED")
                saved += 1
            except Exception as exc:
                pass

    return saved, ov, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dburl", help="the sourcedb url")
    parser.add_argument("outdir", help="the output directory")
    parser.add_argument('ids', nargs='*', help='segment db ids')
    args = parser.parse_args()

    save, ov, total = load(args.dburl, args.outdir, args.ids)
    print ("%d mseed found, %d not saved (errors), "
           "%d file overridden, %d total file saved") % (total, total-save, ov, save)
