"""
    1D plotting routines
"""

from matplotlib import pyplot as pp
import h5py
import argparse
import numpy as np
from datetime import datetime

def plot_1d_loss_err(surf_file, xmin=-1.0, xmax=1.0, loss_max=5, log=False, show=False, timestamp=True, fileformat="png"):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    if timestamp:
        current_timpestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    else:
        current_timpestamp = ""

    f = h5py.File(surf_file, 'r')
    print(f.keys())
    x = f['xcoordinates'][:]
    assert 'train_loss' in f.keys(), "'train_loss' does not exist"
    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    print("train_loss")
    print(train_loss)
    print("train_acc")
    print(train_acc)

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)

    # loss and accuracy map
    fig, ax1 = pp.subplots()
    ax2 = ax1.twinx()
    if log:
        tr_loss, = ax1.semilogy(x, train_loss, 'b-', label='Training loss', linewidth=1)
    else:
        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
    tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)

    if 'test_loss' in f.keys():
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]
        if log:
            te_loss, = ax1.semilogy(x, test_loss, 'b--', label='Test loss', linewidth=1)
        else:
            te_loss, = ax1.plot(x, test_loss, 'b--', label='Test loss', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Test accuracy', linewidth=1)

    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    ax1.set_ylim(0, loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    ax2.set_ylim(0, 100)

    image_file_name = surf_file + '_1d_loss_acc' + ('_log' if log else '') + "_" + current_timpestamp + "_JHA." + fileformat

    pp.savefig(image_file_name, dpi=300, bbox_inches='tight', format=fileformat)

    print(f"Saved to file {image_file_name}")

    # train_loss curve
    pp.figure()
    if log:
        pp.semilogy(x, train_loss)
    else:
        pp.plot(x, train_loss)
    pp.ylabel('Training Loss', fontsize='xx-large')
    pp.xlim(xmin, xmax)
    pp.ylim(0, loss_max)

    image_file_name = surf_file + '_1d_train_loss' + ('_log' if log else '') +  "_" + current_timpestamp + "_JHA." + fileformat

    pp.savefig(image_file_name, dpi=300, bbox_inches='tight', format=fileformat)

    print(f"Saved to file {image_file_name}")

    # train_err curve
    pp.figure()
    pp.plot(x, 100 - train_acc)
    pp.xlim(xmin, xmax)
    pp.ylim(0, 100)
    pp.ylabel('Training Error', fontsize='xx-large')

    image_file_name = surf_file + '_1d_train_err' + "_" + current_timpestamp + "_JHA." + fileformat

    pp.savefig(image_file_name, dpi=300, bbox_inches='tight', format=fileformat)

    print(f"Saved to file {image_file_name}")

    if show:
        pp.show()

    f.close()


def plot_1d_loss_err_repeat(prefix, idx_min=1, idx_max=10, xmin=-1.0, xmax=1.0,
                            loss_max=5, show=False, timestamp=True, fileformat="png"):
    """
        Plotting multiple 1D loss surface with different directions in one figure.
    """

    fig, ax1 = pp.subplots()
    ax2 = ax1.twinx()

    if timestamp:
        current_timpestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    else:
        current_timpestamp = ""

    for idx in range(idx_min, idx_max + 1):
        # The file format should be prefix_{idx}.h5
        f = h5py.File(prefix + '_' + str(idx) + '.h5','r')

        x = f['xcoordinates'][:]
        train_loss = f['train_loss'][:]
        train_acc = f['train_acc'][:]
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]

        xmin = xmin if xmin != -1.0 else min(x)
        xmax = xmax if xmax != 1.0 else max(x)

        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
        te_loss, = ax1.plot(x, test_loss, 'b--', label='Testing loss', linewidth=1)
        tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Testing accuracy', linewidth=1)

    pp.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    ax1.set_ylim(0, loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    ax2.set_ylim(0, 100)

    image_file_name = prefix + '_1d_loss_err_repeat' + "_" + current_timpestamp + "_JHA." + fileformat

    pp.savefig(image_file_name, dpi=300, bbox_inches='tight', format = fileformat)

    print(f"Saved to file {image_file_name}")

    if show:
        pp.show()


def plot_1d_eig_ratio(surf_file, xmin=-1.0, xmax=1.0, val_1='min_eig', val_2='max_eig', ymax=1, show=False,
                      timestamp=True, fileformat="png"):
    print('------------------------------------------------------------------')
    print('plot_1d_eig_ratio')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    x = f['xcoordinates'][:]

    if timestamp:
        current_timpestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    else:
        current_timpestamp = ""

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])
    abs_ratio = np.absolute(np.divide(Z1, Z2))

    pp.plot(x, abs_ratio)
    pp.xlim(xmin, xmax)
    pp.ylim(0, ymax)

    image_file_name = surf_file + '_1d_eig_abs_ratio ' + "_" + current_timpestamp + "_JHA." + fileformat

    pp.savefig(image_file_name, dpi=300, bbox_inches='tight', format = fileformat)

    print(f"Saved to file: {image_file_name}")

    ratio = np.divide(Z1, Z2)
    pp.plot(x, ratio)
    pp.xlim(xmin, xmax)
    pp.ylim(0, ymax)

    image_file_name = surf_file + '_1d_eig_ratio' + "_" + current_timpestamp + "_JHA." + fileformat

    pp.savefig(image_file_name, dpi=300, bbox_inches='tight', format = fileformat)

    print(f"Saved to file {image_file_name}")

    f.close()

    if show:
        pp.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plott 1D loss and error curves')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file contains loss values')
    parser.add_argument('--log', action='store_true', default=False, help='logarithm plot')
    parser.add_argument('--xmin', default=-1, type=float, help='xmin value')
    parser.add_argument('--xmax', default=1, type=float, help='xmax value')
    parser.add_argument('--loss_max', default=5, type=float, help='ymax value')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    parser.add_argument('--prefix', default='', help='The common prefix for surface files')
    parser.add_argument('--idx_min', default=1, type=int, help='min index for the surface file')
    parser.add_argument('--idx_max', default=10, type=int, help='max index for the surface file')
    parser.add_argument('--timestamp', action="store-true", default=False, help='include timestamps in file names')
    parser.add_argument('--fileformat', default="png", help='file format the saved plots are encoded with')

    args = parser.parse_args()

    if args.prefix:
        plot_1d_loss_err_repeat(args.prefix, args.idx_min, args.idx_max,
                                args.xmin, args.xmax, args.loss_max, args.show)
    else:
        plot_1d_loss_err(args.surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
