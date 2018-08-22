import matplotlib.pyplot as plt
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def plot_durations(x,y):
    plt.figure(1)
    plt.clf()
    # plt.subplot(111)
    plt.plot(x, y, 'ro')
    plt.xlim((0, 100))
    plt.ylim((0, 1))
    plt.pause(0.001)  # 暂停一点，以便更新绘图
    # if is_ipython:
    #     display.clear_output(wait=True)           # 清除页面上的内容
        # display.display(plt.gcf())



plot_durations(np.array(loss_list['x']), np.array(loss_list['y']))   #