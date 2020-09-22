import matplotlib.pyplot as plt

import numpy as np


def show_horizontal_bar_chart(exec_time_dict, title):
    """
    Plot execution times as a horizontal bar chart.

    :param exec_time_dict: a dict where keys are descriptions and values are execution times
    :param title: The title of the plot
    :return:
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Execution time (s)\nNote: log-scaled axis!")
    ax.invert_yaxis()  # labels read top-to-bottom

    ax.set_title(title, loc="left")

    labels = exec_time_dict.keys()
    exec_times = [exec_time_dict[label] for label in labels]
    labels = [label if label is not None else "unknown" for label in labels]

    zipped_data = zip(exec_times, labels)
    zipped_data = sorted(zipped_data, reverse=True)
    exec_times, labels = zip(*zipped_data)

    ypos = np.arange(len(exec_times))
    plt.barh(ypos, exec_times, align="center")
    plt.yticks(ypos, labels, fontsize=12)
    plt.xscale("log")
    max_exec_time = max(exec_times)

    for i, value in enumerate(exec_times):
        if value < 0.5 * max_exec_time:
            plt.text(value, i, " {:.2f} s".format(value), va="center")
        else:
            plt.text(
                value,
                i,
                "{:.2f} s ".format(value),
                va="center",
                ha="right",
                color="white",
            )

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(left=0.4)
    plt.show()
    plt.close(fig)
