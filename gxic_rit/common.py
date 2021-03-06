# -*- coding: utf-8 -*-
import json
import copy
import re
import logging
logger = logging.getLogger(__file__)


def smart_log(data):
    data = copy.deepcopy(data)
    data = dict([[x, smart_truncate(y)] for x, y in data.iteritems()])
    return json.dumps(data, sort_keys=True)


def smart_truncate(text, length=100, suffix='...'):
    """Truncates `text`, on a word boundary, as close to
    the target length it can come.
    """

    slen = len(suffix)
    pattern = r'^(.{0,%d}\S)\s+\S+' % (length - slen - 1)
    if len(text) > length:
        match = re.match(pattern, text)
        if match:
            length0 = match.end(0)
            length1 = match.end(1)
            if abs(length0 + slen - length) < abs(length1 + slen - length):
                return match.group(0) + suffix
            else:
                return match.group(1) + suffix
    return text
