# neural-networks-and-deep-learning-in-golang

Code freely adaptated from Michael Nielsen's ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) book in Go.

_NB: Instead of using the 'mnist.pkl.gz' file, I used the MNIST set [in CSV](https://pjreddie.com/projects/mnist-in-csv/) as provided by Joseph Redmon and zipped the test and the train sets in the 'data' folder._


### Installation

```console
$ git clone https://github.com/cyrildever/neural-networks-and-deep-learning-in-golang.git
$ cd neural-networks-and-deep-learning-in-golang
$ go build
```


### Usage

```console
$ ./neuraldeep -n=1 -op=train -layers=784,30,10 -data=training -useMNIST=true -epochs=30 -size=10 -eta=3.0 -load=false
```

```
Usage of ./neuraldeep:
  -data string
        a single data set to feed the first layer (a comma-separated list of float64), or the name of the MNIST set (test | training | validation)
  -epochs int
        number of epochs (default 1)
  -eta float
        learning rate (default 0.1)
  -eval true
        set to true to add evaluation at each training epoch
  -label string
        the label/target of the passed value as a float64 number
  -layers string
        comma-separated list of number of neurons per layer (the first one being the size of the input layer)
  -load true
        set to true if you want to load an existing network
  -n string
        the network implementation to use: 1 | 2 | 3 (default "1")
  -op string
        operation to proceed: predict | test | train
  -path string
        path to the existing file (default "./data/saved/network/")
  -size int
        mini-batch size (default 10)
  -src string
        the source file to use as input data
  -useMNIST
        set to true to use MNIST dataset (the layers flag should start with 784 and end with 10)
```


## License

The code in Go is distributed under an [MIT license](LICENSE).
Please see [Michael Nielsen's website](http://neuralnetworksanddeeplearning.com/) for his authorization.


<hr />
&copy; 2020 Cyril Dever. All rights reserved.