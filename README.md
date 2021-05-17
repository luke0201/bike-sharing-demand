# Bike Sharing Demand

This is my Kaggle submission code.

## Requirements

You need Python 3.6 or higher version.

The following libraries are required.

- numpy
- pandas
- scikit-learn
- lightgbm

## Usage

Download the dataset using the following command.

```
kaggle competitions download -c bike-sharing-demand
unzip bike-sharing-demand.zip && rm bike-sharing-demand.zip
```

Then run `bike-sharing-demand.py` as follows.

```
python bike-sharing-demand.py
```

Finally, submit the result using the following command. Replace the `<submission_message>` by yours.

```
kaggle competitions submit -c bike-sharing-demand -f submission.csv -m <submission_message>
```
