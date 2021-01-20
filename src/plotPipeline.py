import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crux'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils'));

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from algorithms import plot;

def main():
    args = sys.argv[1:len(sys.argv)-1];
    legends = []; files = []; islegend = True;

    for i in args:
        if islegend:
            legends.append(i);
        else:
            files.append(i);
            logger.info('file={}'.format(i));
            assert os.path.isfile(i);
        islegend = not islegend;

    plot.plot(zip(files, legends),sys.argv[-1]);


if __name__ == "__main__":
    main();

