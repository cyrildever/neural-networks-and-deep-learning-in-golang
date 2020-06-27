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
$ ./neuraldeep --layers 6,50,20,3 --data "1,2.4,3,4,5,-6" --label 3
```

```
Usage of ./neuraldeep:
  -data string
        a single data set to feed the first layer (a comma-separated list of float64), or the name of the MNIST set (test | training | validation)
  -label string
        the label/target of the passed value as a float64 number
  -layers string
        comma-separated list of number of neurons per layer (the first one being the size of the input layer)
  -op string
        operation to proceed: predict | test | train
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