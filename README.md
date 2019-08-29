Here I attempt to solve the Synthesis Visual Reasoning Test by Relational Networks.

## Synthesis Visual Reasoning Test (SVRT)
The SVRT was firstly published in [Comparing machines and humans on a visual categorization test](https://www.pnas.org/content/108/43/17621.short). It has been shown to be a task, relatively easy for human subjects, but challenging for machine agents.

The original code for the SVRT generator can be found [here](https://www.idiap.ch/~fleuret/svrt/). My colleague [Scott](https://github.com/scottclowe) has been developing a more complicated version of the generator (which is in his private repository; please contact him if you are interested). Based on one of his new versions, I have made some modifications (which I would probably import to my own repository at some time point in the future).

## Relational Networks (RN)
An RN was initially published in [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf). It has been shown to be a neural network good at detecting relationships between objects.

The original code of the Pytorch implementation of the RN can be found [here](https://github.com/kimhc6028/relational-networks).

### Requirements
- Python 2.7
- [numpy](http://www.numpy.org/)
- [pytorch](http://pytorch.org/)
- [opencv](http://opencv.org/)

### Branches
- [master](https://github.com/anish-lu-yihe/SVRT-by-RN):
Currently this branch is a clone from the original code, without any modifications.

- [null-qst](https://github.com/anish-lu-yihe/SVRT-by-RN/tree/null-qst):
The original code contains not only an RN but other components; particularly, qst is an input other than the input images, which is obtained by an LSTM from questions. However, the SVRT does not have the component of questions. So before moving this qst entry systematically from the neural network, I have nullified it first by setting all its instantiations to be zero vectors. In this way, the compatibility of the code and the SVRT inputs can be checked, and an initial test of the RN performance on the SVRT can be obtained, which is expected to be not so good.

- **RN-on-image**:
Under development.

### Usage
1. Generate SVRT problems (not included in this project).
2. Run main.py, or run.sh.

### Modifications
Not available.

## Results by **RN-on-image**
Not available.
