# Signal_Processing_MA1_FPMS_UMONS
Signal Processing Project (2018, FPMs at the University of Mons)

The aim of this project is to build a rule based classifier to classified (english) speakers by gender. The pitch of each sample is use to find a threshold value which allows to discriminate the samples.

## Requirements

In order to run this project you need Python 3 with the following modules:

- time (standard python module);
- numpy;
- scipy;
- matplotlib.

In a terminal run pip to install the modules if necessary:

In Linux:

```bash
pip install numpy scipy matplotlib
```

In Windows:

```bash
py -m pip install numpy scipy matplotlib
```

Or:

```bash
py -3 -m pip install numpy scipy matplotlib
```

## Usage

Run the `main.py` script with the known samples to find a threshold. For the determination of the threshold pitch the (train) samples are stored in the `samples` folder which have to be in the same directory as the `main.py` script. For the test the samples are stored in the `samples_to_test` folder which is in the same directory as the `main.py` script. Each sample is a `.wav` file, the female speakers are label with a `(2).wav`.

Finally to run the script:

In Linux:

```bash
python3 main.py
```

In Windows:

```bash
py main.py
```

Or:

```bash
py -3 main.py
```

