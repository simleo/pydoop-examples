import itertools

LOG_LEVELS = 'CRITICAL', 'DEBUG', 'ERROR', 'INFO', 'WARNING'
LOG_LEVEL_KEY = 'pydeep.log.level'

EVAL_STEP_INTERVAL_KEY = 'tensorflow.eval.step.interval'
MODEL_EXPORT_DIR_KEY = 'tensorflow.model.export.dir'
GRAPH_ARCH_KEY = 'tensorflow.graph.architecture'
NUM_STEPS_KEY = 'tensorflow.train.num.steps'
TRAIN_BATCH_SIZE_KEY = 'tensorflow.train.batch.size'
VALIDATION_PERCENT_KEY = 'tensorflow.train.validation.percent'

# The following should be defined in pydoop
PYDOOP_EXTERNALSPLITS_URI_KEY = 'pydoop.mapreduce.pipes.externalsplits.uri'
NUM_MAPS_KEY = 'mapreduce.job.maps'


def balanced_split(seq, N):
    """\
    Partition seq into exactly N balanced groups.

    list(range(10)), 3 ==> [0, 1, 2, 3], [4, 5, 6], [7, 8, 9]

    Returns an iterator through the groups.
    """
    if N < 1:
        raise ValueError("number of groups must be strictly positive")
    if N > len(seq):
        raise ValueError("not enough elements in sequence")
    q, r = divmod(len(seq), N)
    lengths = r * [q + 1] + (N - r) * [q]
    for end, l in zip(itertools.accumulate(lengths), lengths):
        yield seq[end - l: end]
