#!/usr/bin/env python

"""
Parse training log

Evolved from parse_log.sh

------------------------------

Modified by soonmin, Dec 15, 2016.
"""

import os
import re
import sys
import datetime
import argparse
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

def extract_datetime_from_line(line, year):
    # Expected format: I0210 13:39:22.381027 25210 solver.cpp:204] Iteration 100, lr = 0.00992565
    line = line.strip().split()
    month = int(line[0][1:3])
    day = int(line[0][3:])
    timestamp = line[1]
    pos = timestamp.rfind('.')
    ts = [int(x) for x in timestamp[:pos].split(':')]
    hour = ts[0]
    minute = ts[1]
    second = ts[2]
    microsecond = int(timestamp[pos + 1:])
    dt = datetime.datetime(year, month, day, hour, minute, second, microsecond)
    return dt


def get_log_created_year(input_file):
    """Get year from log file system timestamp
    """

    log_created_time = os.path.getctime(input_file)
    log_created_year = datetime.datetime.fromtimestamp(log_created_time).year
    return log_created_year


def get_start_time(line_iterable, year):
    """Find start time from group of lines
    """

    start_datetime = None
    for line in line_iterable:
        line = line.strip()
        if line.find('Iteration ') != -1:
            start_datetime = extract_datetime_from_line(line, year)
            break
    return start_datetime


def extract_seconds(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    log_created_year = get_log_created_year(input_file)
    start_datetime = get_start_time(lines, log_created_year)
    assert start_datetime, 'Start time not found'

    out = open(output_file, 'w')
    for line in lines:
        line = line.strip()        

        if line.find('Iteration') != -1:
            dt = extract_datetime_from_line(line, log_created_year)
            elapsed_seconds = (dt - start_datetime).total_seconds()
            out.write('%f\n' % elapsed_seconds)
    out.close()


def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, test_dict_list)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows
    """

    # Added by soonmin, ignore training speed check in py-faster-rcnn
    regex_ignore = re.compile('speed:')
    regex_terminate = re.compile('done')

    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')    
    regex_learning_rate = re.compile('lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')
    regex_train_total_loss = re.compile(', loss = ([\.\deE+-]+)')
    regex_test_total_loss = re.compile('Test loss: ([\.\deE+-]+)')

    # Pick out lines of interest
    iteration = -1
    learning_rate = float('NaN')

    train_dict_list = []
    test_dict_list = []

    log_basename = os.path.basename(path_to_log)
    
    train_filename = os.path.join(os.path.dirname(path_to_log), log_basename + '.train')
    if os.path.exists(train_filename):
        # Check whether the parsed log file exists
        with open(train_filename, 'r') as f:
            lines = [line.rstrip('\r\n') for line in f.readlines()]
        keys = lines[0].split(',')
        
        for line in lines[1:]:
            log = []
            val = line.split(',')
            for ii, key in enumerate(keys):
                log.append( (key, float(val[ii])) )
            train_dict_list.append( OrderedDict( log ) )

        init_iteration = train_dict_list[-1]['NumIters']
    else:
        init_iteration = 0

    

    test_filename = os.path.join(os.path.dirname(path_to_log), log_basename + '.test')
    if os.path.exists(test_filename):
        # Check whether the parsed log file exists
        with open(test_filename, 'r') as f:
            lines = [line.rstrip('\r\n') for line in f.readlines()]
        keys = lines[0].split(',')
        
        for line in lines[1:]:
            log = []
            val = line.split(',')
            for ii, key in enumerate(keys):
                log.append( (key, val[ii]) )
            test_dict_list.append( OrderedDict( log ) )

    train_row = None
    test_row = None

    train_total_loss = None
    test_total_loss = None

    logfile_year = get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        
        # import ipdb
        # ipdb.set_trace()

        start_time = get_start_time(f, logfile_year)
        

        for line in f:
            ignore_match = regex_ignore.search(line)
            terminate_match = regex_terminate.search(line)
            if ignore_match:
                continue
            if terminate_match:
                break

            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration < init_iteration or iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue

            try:
                time = extract_datetime_from_line(line, logfile_year)
            except ValueError:
                # Skip lines with bad formatting, for example when resuming solver
                continue

            seconds = (time - start_time).total_seconds()

            learning_rate_match = regex_learning_rate.search(line)
            if learning_rate_match:
                learning_rate = float(learning_rate_match.group(1))

            # Training total loss
            train_total_loss_match = regex_train_total_loss.search(line)
            if train_total_loss_match:                
                train_total_loss = float(train_total_loss_match.group(1))
                
            # Testing total loss
            test_total_loss_match = regex_test_total_loss.search(line)
            if test_total_loss_match:
                test_total_loss = float(test_total_loss_match.group(1))

            # import ipdb
            # ipdb.set_trace()

            train_dict_list, train_row = parse_line_for_net_output(
                regex_train_output, train_row, train_dict_list,
                line, iteration, seconds, learning_rate, train_total_loss )

            test_dict_list, test_row = parse_line_for_net_output(
                regex_test_output, test_row, test_dict_list,
                line, iteration, seconds, learning_rate, test_total_loss )

    fix_initial_nan_learning_rate(train_dict_list)
    fix_initial_nan_learning_rate(test_dict_list)

    return train_dict_list, test_dict_list


def parse_line_for_net_output(regex_obj, row, row_dict_list,
                              line, iteration, seconds, learning_rate, total_loss):
    """Parse a single line for training or test output

    Returns a a tuple with (row_dict_list, row)
    row: may be either a new row or an augmented version of the current row
    row_dict_list: may be either the current row_dict_list or an augmented
    version of the current row_dict_list
    """

    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one
            if row:
                # If we're on a new iteration, push the last row
                # This will probably only happen for the first row; otherwise
                # the full row checking logic below will push and clear full
                # rows
                row_dict_list.append(row)

            row = OrderedDict([
                ('NumIters', iteration),
                ('Seconds', seconds),
                ('total_loss', total_loss),
                ('LearningRate', learning_rate)
            ])

        # output_num is not used; may be used in the future
        # output_num = output_match.group(1)
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        row[output_name] = float(output_val)

    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        # The row is full, based on the fact that it has the same number of
        # columns as the first row; append it to the list
        row_dict_list.append(row)
        row = None

    return row_dict_list, row


def fix_initial_nan_learning_rate(dict_list):
    """Correct initial value of learning rate

    Learning rate is normally not printed until after the initial test and
    training step, which means the initial testing and training rows have
    LearningRate = NaN. Fix this by copying over the LearningRate from the
    second row, if it exists.
    """

    if len(dict_list) > 1:
        dict_list[0]['LearningRate'] = dict_list[1]['LearningRate']


def save_csv_files(logfile_path, output_dir, train_dict_list, test_dict_list,
                   delimiter=',', verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)
    train_filename = os.path.join(output_dir, log_basename + '.train')
    write_csv(train_filename, train_dict_list, delimiter, verbose)

    test_filename = os.path.join(output_dir, log_basename + '.test')
    write_csv(test_filename, test_dict_list, delimiter, verbose)


def write_csv(output_filename, dict_list, delimiter, verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))


    args = parser.parse_args()
    return args

def drawPlot(ax, dict_list, xLabel, yLabel, avg_step, clr, prop):
    xx = [ dl[xLabel] for dl in dict_list]
    yy = [ dl[yLabel] for dl in dict_list]

    yy_avg = []
    for ii in range(0, len(yy), avg_step):
        s = max(0, ii-avg_step/2)
        e = min(len(yy), ii+avg_step/2)
        yy_avg.append(np.mean(yy[s:e]))

    ax.plot(xx, yy, clr, alpha=prop['alpha'])
    ax.plot(xx[::avg_step], yy_avg, clr, label=prop['label'])

    if 'mAP' not in yLabel:
        ax.set_yscale('log')

def main():
    args = parse_args()
    train_dict_list, test_dict_list = parse_log(args.logfile_path)

    # Save to csv files
    output_dir = os.path.dirname(args.logfile_path)
    save_csv_files(args.logfile_path, output_dir, train_dict_list,
                   test_dict_list, delimiter=args.delimiter) 

    # import ipdb
    # ipdb.set_trace()

    try:
        loss_term = [ loss for loss in test_dict_list[0].keys() if 'loss' in loss or 'eval' in loss ]
    except:
        loss_term = [ loss for loss in train_dict_list[0].keys() if 'loss' in loss ]

    cols = int(np.ceil(len(loss_term) / 2.0))
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.subplots_adjust(hspace=0.5)
    axes = axes.flatten()

    for ii, term in enumerate(loss_term):
        
        if term in train_dict_list[0]:
            drawPlot(axes[ii], train_dict_list, 'NumIters', term, 20, 'b-', {'alpha':0.3, 'label':'Train'})

        try:	
            if 'total' in term:
                drawPlot(axes[ii], test_dict_list,  'NumIters', 'detection_eval', 6,  'r-', {'alpha':0.3, 'label':'Test'})
            else: 
                drawPlot(axes[ii], test_dict_list,  'NumIters', term, 6,  'r-', {'alpha':0.3, 'label':'Test'})
        except:
            print('Cannot find test loss')

        axes[ii].set_title(term)
        axes[ii].set_xlabel('Iteration')
        axes[ii].set_ylabel('Loss')
        xticklabel = [ '{}k'.format(int(int(xl)/1000)) for xl in axes[ii].get_xticks().tolist()[1:] ]
        axes[ii].set_xticklabels( [''] + xticklabel )
        axes[ii].legend()	


    sns.set_style("white")
    sns.set_context("poster")
    
    plt.savefig(args.logfile_path + '.png')

if __name__ == '__main__':

    main()

