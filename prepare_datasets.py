import subprocess
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any
from threading import Lock

import pandas as pd

datawarehouse_lock = Lock()

def ensure_datawarehouse_downloaded() -> None:
    global datawarehouse_lock
    with datawarehouse_lock:
        os.makedirs('data/3rd-party', exist_ok=True)
        if os.path.exists('data/3rd-party/datawarehouse/datawarehouse/higgsml.py'):
            return

        print('Downloading datawarehouse...')
        if os.path.exists('data/3rd-party/datawarehouse'):
            shutil.rmtree('data/3rd-party/datawarehouse')
        subprocess.run(
            ['git', 'clone', 'https://github.com/kodo-pp/datawarehouse', '--branch', 'fixes'],
            check=True,
            cwd='data/3rd-party',
        )
        print('Downloaded datawarehouse')

base_dataset_lock = Lock()

def ensure_base_dataset_downloaded() -> None:
    global base_dataset_lock
    with base_dataset_lock:
        os.makedirs('data', exist_ok=True)
        if os.path.exists('data/atlas-higgs-challenge-2014-v2.csv.gz'):
            return

        print('Downloading base dataset...')
        subprocess.run(
            ['curl', '-O', 'https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz'],
            check=True,
            cwd='data',
        )
        print('Downloaded base dataset')

def catch(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return None
        except BaseException as e:
            print(f'Exception: {e}')
            return e
    return inner

def ensure_skewed_dataset_prepared(z: str) -> None:
    ensure_base_dataset_downloaded()
    ensure_datawarehouse_downloaded()

    os.makedirs('data/skewed', exist_ok=True)

    output_csv_path = os.path.realpath(f'data/skewed/HiggsML_TES_{z}.csv.gz')
    output_h5_path = os.path.realpath(f'data/skewed/HiggsML_TES_{z}.h5')
    input_path = os.path.realpath(f'data/atlas-higgs-challenge-2014-v2.csv.gz')
    if os.path.exists(output_h5_path):
        return

    print(f'Preparing skewed dataset for z = {z}...')
    if not os.path.exists(output_csv_path):
        subprocess.run(
            [
                'python', '-m', 'datawarehouse.higgsml',
                '-i', input_path,
                '-o', output_csv_path,
                '--tes', z,
            ],
            check=True,
            cwd='data/3rd-party/datawarehouse',
        )

    print('Reading CSV...')
    df = pd.read_csv(output_csv_path)
    print('Applying filter...')
    df = df[df['PRI_tau_pt'] > 22]
    df['Z'] = float(z)
    print('Writing HDF5...')
    df.to_hdf(output_h5_path, key='data_syst', complib='bzip2', complevel=3)
    os.remove(output_csv_path)
    print(f'Prepared skewed dataset for z = {z}, length = {len(df)}')

class Counter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def inc(self) -> None:
        with self._lock:
            self._value += 1

    def get(self) -> int:
        with self._lock:
            return self.value

def prepare_datasets_for(z_values: List[str]) -> None:
    n = len(z_values)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(catch(ensure_skewed_dataset_prepared), z): z for z in z_values}
        for i, future in enumerate(as_completed(futures)):
            exc = future.result()
            if exc is not None:
                raise exc
            print(f'Done ({i+1}/{n}) z = {futures[future]}')

def prepare_all_datasets() -> None:
    z_values=[
        0.7,  0.74, 0.78, 0.8,  0.84, 0.88,
        0.9,  0.92, 0.94, 0.96, 0.98, 0.99,
        1.0,  1.01, 1.02, 1.04, 1.06, 1.08,
        1.09, 1.1,  1.11, 1.12, 1.13, 1.14,
    ]
    prepare_datasets_for([str(z) for z in z_values])

if __name__ == '__main__':
    prepare_all_datasets()
