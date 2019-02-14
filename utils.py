import sys
def print_progress(i, time_elapsed = None, before = "progress:", after = ""):
    """show progress of process, often used in training a neural network model
    params
        i: progress value in [0, 1.], double
        time_elapsed: time elapsed from progress 0.0 to current progress, double
        before: descriptive content displayed before progress i, str
        after: descriptive content displayed after progress i, str
    """
    if time_elapsed is not None:
        if i >= 1:
            time_remaining = 0
        elif 0 < i < 1:
            time_remaining = time_elapsed * (1. - i) / i
        else:
            time_remaining = float('inf')
            
    progress_info = '{:>7.2%}'.format(i) # align right, 7 characters atmost
    if time_elapsed is not None:
        progress_info += ' {:.0f}m{:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
        if 0 < i < 1.0:
            progress_info += ' {:.0f}m{:.0f}s'.format(
                time_remaining // 60, time_remaining % 60)

    # display progress repeately in the same line.
    progress_info = '\r' + before + progress_info + after
    sys.stdout.flush()
    sys.stdout.write(progress_info)
