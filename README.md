# neural-networks-and-deep-learning-in-golang

Code freely adaptated from Michael Nielsen's ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) book in Go.


### Installation

```console
$ git clone https://github.com/cyrildever/neural-networks-and-deep-learning-in-golang.git
$ cd neural-networks-and-deep-learning-in-golang
$ go build
```


### Usage

```console
$ ./neuraldeep --layers 6,50,20,3 --data "1,2.4,3,4,5,-6"
```

_NB: Instead of using the provided 'mnist.pkl.gz' file, I used the original MNIST set [in CSV](https://pjreddie.com/projects/mnist-in-csv/) as provided by Joseph Redmon and zipped the test and the train sets in the 'data' folder.


## License

The code in Go is distributed under an [MIT license](LICENSE).
Please see [Michael Nielsen's website](http://neuralnetworksanddeeplearning.com/) for his authorization.


<hr />
&copy; 2020 Cyril Dever. All rights reserved.