from reading_data import *

if __name__ == '__main__':
    data_dir = Path('./input')
    out_dir = data_dir / 'npz'

    file_paths = list(out_dir.glob('*Training*.npz'))
    print(file_paths)

    for fp in file_paths:
        print(fp.name)
        X, y = load_npz_file(fp)
        print('Number of samples: ')
        print(y.shape[0])
        print('frac of positives: ')
        print(y.sum() / y.shape[0])
        print('frac of negatives: ')
        print((y.shape[0] - y.sum())/y.shape[0])
