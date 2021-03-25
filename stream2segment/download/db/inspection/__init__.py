# -*- encoding: utf-8 -*-
'''
Module implementing the download info (print statistics and generate html page)

:date: Mar 15, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

# there is no much added value in providing a html page for the moment. However, the
# function was implemented (NOT TESTED) and we leave it below for the moment

# def get_dreport_html_template_arguments(session, download_ids=None, config=True, log=True):
#     '''Returns an html page (string) yielding the download statistics and information matching the
#     given parameters.
#
#     :param session: an sql-alchemy session denoting a db session to a database
#     :param download_ids: (list of ints or None) if None, collect stats from all downloads run.
#         Otherwise limit the output to the downloads whose ids are in the list. In any case, in
#         case of more download runs to be considered, this function will
#         yield also the statistics aggregating all downloads in a table at the end
#     :param config: boolean (default: True). Whether to show the download config
#     :param log: boolean (default: True). Whether to show the download log messages
#     '''
#     log_types = set()
#     data = infoquery(session, download_ids, config, log)
#     html_data = []
#     for dwnl_id, dwnl_time, configtext, logtext in data:
#         loglist = []
#         prevmatch = None
#         for match in re.finditer(r'^\[(\w+)\](.*?)(?=(?:$|^\[\w+\]))', logtext,
#                                  re.MULTILINE | re.DOTALL):  # @UndefinedVariable
#             if prevmatch is not None:
#                 log_types.add(match.group(1))
#                 loglist.append([match.group(1), match.group(2)])
#         html_data.append({'id': dwnl_id, 'time': dwnl_time,
#                           'config': configtext, 'logs': loglist})
#     return dict(data=data, log_types=sorted(log_types))


# def get_template(mode='outcome', ):
#     '''Returns the jinja2 template for the html page of the download statistics'''
#     thisdir = os.path.dirname(__file__)
#     templatespath = os.path.join(os.path.dirname(thisdir), 'webapp', 'templates')
#     csspath = os.path.join(os.path.dirname(thisdir), 'webapp', 'static', 'css')
#     env = Environment(loader=FileSystemLoader([thisdir, templatespath, csspath]))
#     return env.get_template('d.%s.info.html' % mode)


