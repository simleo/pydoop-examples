import itertools

LOG_LEVELS = 'CRITICAL', 'DEBUG', 'ERROR', 'INFO', 'WARNING'
LOG_LEVEL_KEY = 'pydeep.log.level'

BNECKS_DIR_KEY = 'pydeep.bottlenecks.dir'
EVAL_STEP_INTERVAL_KEY = 'tensorflow.eval.step.interval'
GRAPH_ARCH_KEY = 'tensorflow.graph.architecture'
LEARNING_RATE_KEY = 'tensorflow.learning.rate'
NUM_STEPS_KEY = 'tensorflow.train.num.steps'
SEED_KEY = 'pydeep.random.seed'
TRAIN_BATCH_SIZE_KEY = 'tensorflow.train.batch.size'
VALIDATION_BATCH_SIZE_KEY = 'tensorflow.validation.batch.size'
VALIDATION_PERCENT_KEY = 'tensorflow.train.validation.percent'

# The following should be defined in pydoop
PYDOOP_EXTERNALSPLITS_URI_KEY = 'pydoop.mapreduce.pipes.externalsplits.uri'
NUM_MAPS_KEY = 'mapreduce.job.maps'


def balanced_parts(L, N):
    """\
    Find N numbers that sum up to L and are as close as possible to each other.

    >>> balanced_parts(10, 3)
    [4, 3, 3]
    """
    if not (1 <= N <= L):
        raise ValueError("number of partitions must be between 1 and %d" % L)
    q, r = divmod(L, N)
    return r * [q + 1] + (N - r) * [q]


def balanced_chunks(L, N):
    """\
    Same as balanced_part, but as an iterator through (offset, length) pairs.

    >>> list(balanced_chunks(10, 3))
    [(0, 4), (4, 3), (7, 3)]
    """
    lengths = balanced_parts(L, N)
    return zip(itertools.accumulate([0] + lengths), lengths)


def balanced_split(seq, N):
    """\
    Partition seq into exactly N balanced groups.

    Returns an iterator through the groups.

    >>> seq = list(range(10))
    >>> list(balanced_split(seq, 3))
    [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    for offset, length in balanced_chunks(len(seq), N):
        yield seq[offset: offset + length]
