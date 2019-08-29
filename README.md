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

- **null-qst**:
The original code contains not only an RN but other components; particularly, qst is an input other than the input images, which is obtained by an LSTM from questions. However, the SVRT does not have the component of questions. So before moving this qst entry systematically from the neural network, I have nullified it first by setting all its instantiations to be zero vectors. In this way, the compatibility of the code and the SVRT inputs can be checked, and an initial test of the RN performance on the SVRT can be obtained, which is expected to be not so good.

- [RN-on-image](https://github.com/anish-lu-yihe/SVRT-by-RN/tree/RN-on-image): Under development.

### Usage
1. Generate SVRT problems (not included in this project).
2. Run main.py, or run.sh.

### Modifications
- load_svrt.py has been added for loading SVRT images.
- qst has been nullified.

## Results by **null-qst**
In the original paper (and code), the RN is compared to a multi-layer perceptron (MLP). I have kept the MLP for comparison; it runs much faster than the RN. Here either network was trained on 9k vignettes (half in-class and half out-class) and tested on 1k unseen vignettes. The training and testing processes were repeated 20 times, which led to the following results:

|    | RN      |        | MLP     |        |
|----|---------|--------|---------|--------|
| \# | mean    | s.e.   | mean    | s.e.   |
| 1  | 50\.20% | 0\.27% | 50\.45% | 0\.33% |
| 2  | 89\.35% | 2\.79% | 91\.85% | 3\.05% |
| 3  | 67\.40% | 3\.92% | 76\.55% | 3\.84% |
| 4  | 79\.70% | 1\.88% | 82\.95% | 2\.05% |
| 5  | 49\.75% | 0\.23% | 48\.45% | 0\.35% |
| 6  | 67\.05% | 1\.00% | 68\.10% | 0\.93% |
| 7  | 48\.25% | 0\.26% | 47\.05% | 0\.26% |
| 8  | 85\.45% | 0\.76% | 84\.85% | 0\.91% |
| 9  | 49\.20% | 0\.25% | 52\.30% | 0\.39% |
| 10 | 73\.20% | 1\.69% | 84\.65% | 2\.29% |
| 11 | 92\.20% | 2\.32% | 90\.70% | 3\.01% |
| 12 | 48\.50% | 0\.30% | 47\.90% | 0\.28% |
| 13 | 49\.75% | 0\.28% | 50\.50% | 0\.28% |
| 14 | 60\.50% | 0\.83% | 54\.45% | 1\.06% |
| 15 | 52\.25% | 0\.42% | 47\.60% | 0\.31% |
| 16 | 49\.40% | 0\.48% | 49\.85% | 0\.31% |
| 17 | 57\.25% | 0\.47% | 64\.45% | 0\.61% |
| 18 | 71\.80% | 1\.77% | 90\.15% | 1\.88% |
| 19 | 48\.90% | 0\.32% | 50\.10% | 0\.23% |
| 20 | 49\.65% | 0\.40% | 49\.25% | 0\.35% |
| 21 | 49\.65% | 0\.40% | 50\.80% | 0\.30% |
| 22 | 50\.20% | 0\.29% | 49\.40% | 0\.25% |
| 23 | 57\.05% | 1\.20% | 56\.35% | 0\.73% |

As expected, the performance of the RN was not good; for many SVRT problems, its performance was at the chance level. The performance of the MLP was not good either, but it outperformed the RN in some problems.
