# Assignment 4

- name: Alexander Cooper
- student ID: 20629774

## Dependencies

- json
- numpy

## Running `main.py`

To run `main.py`, use

```
python main.py -p param/params.json -r results -d data/even_mist.csv -n 100

```

```
-p param/params.json

```
The location of the paramater file containing the program's hyper paramaters

```
-r results

```
The directory for the results to be stored

```
-d data/even_mist.csv

```
The directory and file name of the datafile to read from

```
-n 100

```
The number of sample images to produce

## Paramater File

The paramater file should be a `.json` of the form:

```
{"learning rate":1,

```
The learning rate our neural net uses

```
"display epochs":true,

```
Whether or not to show the current epoch in the training loop

```
"num epochs":10}

```
The number of epochs for our training loop


