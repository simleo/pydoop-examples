import itertools

LOG_LEVELS = 'CRITICAL', 'DEBUG', 'ERROR', 'INFO', 'WARNING'
GRAPH_ARCH_KEY = 'tensorflow.graph.architecture'

# The following should be defined in pydoop
PYDOOP_EXTERNALSPLITS_URI_KEY = 'pydoop.mapreduce.pipes.externalsplits.uri'
NUM_MAPS_KEY = 'mapreduce.job.maps'


def balanced_split(seq, N):
    """\
    Partition seq into exactly N balanced groups.

    list(range(10)), 3 ==> [0, 1, 2, 3], [4, 5, 6], [7, 8, 9]

    Returns an iterator through the groups.
    """
    q, r = divmod(len(seq), N)
    lengths = r * [q + 1] + (N - r) * [q]
    for end, l in zip(itertools.accumulate(lengths), lengths):
        yield seq[end - l: end]
