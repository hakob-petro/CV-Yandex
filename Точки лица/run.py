#!/usr/bin/env python3

from json import dumps, load
from numpy import array
from os import environ
from os.path import join
from sys import argv


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


def save_csv(facepoints, filename):
    with open(filename, 'w') as fhandle:
        print('filename,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14',
              file=fhandle)
        for filename in sorted(facepoints.keys()):
            points_str = ','.join(map(str, facepoints[filename]))
            print(f'{filename},{points_str}', file=fhandle)


def check_test(data_dir):
    gt_dir = join(data_dir, 'gt')
    output_dir = join(data_dir, 'output')

    def read_img_shapes(gt_dir):
        img_shapes = {}
        with open(join(gt_dir, 'img_shapes.csv')) as fhandle:
            next(fhandle)
            for line in fhandle:
                parts = line.rstrip('\n').split(',')
                filename = parts[0]
                n_rows, n_cols = map(int, parts[1:])
                img_shapes[filename] = (n_rows, n_cols)
        return img_shapes

    detected = read_csv(join(output_dir, 'output.csv'))
    gt = read_csv(join(gt_dir, 'gt.csv'))
    img_shapes = read_img_shapes(gt_dir)

    error = 0.0
    all_found = True
    for filename, gt_coords in gt.items():
        if filename not in detected:
            all_found = False
            res = f'Error, keypoints for "{filename}" not found'
            break

        coords = detected[filename]
        n_rows, n_cols = img_shapes[filename]

        diff = (coords - gt_coords)
        diff[::2] /= n_cols
        diff[1::2] /= n_rows
        diff *= 100
        error += (diff ** 2).mean()
    error /= len(gt)

    if all_found:
        res = f'Ok, error {error:.4f}'

    if environ.get('CHECKER'):
        print(res)
    return res


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    result = results[-1]['status']

    if not result.startswith('Ok'):
        res = {'description': '', 'mark': 0}
    else:
        error_str = result[10:]
        error = float(error_str)

        if error <= 5:
            mark = 10
        elif error <= 6:
            mark = 9
        elif error <= 8:
            mark = 8
        elif error <= 10:
            mark = 7
        elif error <= 12:
            mark = 6
        elif error <= 14:
            mark = 5
        elif error <= 16:
            mark = 4
        elif error <= 18:
            mark = 3
        elif error <= 20:
            mark = 2
        else:
            mark = 0

        res = {'description': error_str, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


def run_single_test(data_dir, output_dir):
    from detection import train_detector, detect
    from os.path import abspath, dirname, join

    train_dir = join(data_dir, 'train')
    test_dir = join(data_dir, 'test')

    train_gt = read_csv(join(train_dir, 'gt.csv'))
    train_img_dir = join(train_dir, 'images')

    model = train_detector(train_gt, train_img_dir, fast_train=True)

    code_dir = dirname(abspath(__file__))
    model_filename = join(code_dir, 'facepoints_model.ckpt')
    test_img_dir = join(test_dir, 'images')
    detected_points = detect(model_filename, test_img_dir)
    save_csv(detected_points, join(output_dir, 'output.csv'))


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally, run on dir with tests
        if len(argv) != 2:
            print(f'Usage: {argv[0]} tests_dir')
            exit(0)

        from glob import glob
        from json import dump
        from re import sub
        from time import time
        from traceback import format_exc
        from os import makedirs
        from os.path import basename, exists
        from shutil import copytree

        tests_dir = argv[1]

        results = []
        for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_*_input'))):
            output_dir = sub('input$', 'check', input_dir)
            run_output_dir = join(output_dir, 'output')
            makedirs(run_output_dir, exist_ok=True)
            gt_src = sub('input$', 'gt', input_dir)
            gt_dst = join(output_dir, 'gt')
            if not exists(gt_dst):
                copytree(gt_src, gt_dst)

            try:
                start = time()
                run_single_test(input_dir, run_output_dir)
                end = time()
                running_time = end - start
            except:
                status = 'Runtime error'
                traceback = format_exc()
            else:
                try:
                    status = check_test(output_dir)
                except:
                    status = 'Checker error'
                    traceback = format_exc()

            test_num = basename(input_dir)[:2]
            if status == 'Runtime error' or status == 'Checker error':
                print(test_num, status, '\n', traceback)
                results.append({'status': status})
            else:
                print(test_num, f'{running_time:.2f}s', status)
                results.append({
                    'time': running_time,
                    'status': status})

        dump(results, open(join(tests_dir, 'results.json'), 'w'))
        res = grade(tests_dir)
        print('Mark:', res['mark'], res['description'])
