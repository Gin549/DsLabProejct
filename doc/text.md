# Data Science Lab: Process and methods Group project Winter Call

## Abstract


## Problem overview
The group project consists in predicting the position of a particle when it traverses a sensor RSD. The sensor is flat, hence the position is represented by $(x, y)$ values. 
The task has to be performed through a multi-output regression pipeline, one per coordinate. \\

The Resistive Silicon Detector (RSD) sensor is composed by 12 metallic *pads*. They have a asterisk shape and they can be clearly distinguished in Figure \ref{fig:rsd}.\\
Each pad records a signal that it is trasfered and stored by the system. Due to hardware constrains, not all the measures transfered are meaningful, but 6 of the 18 readings are noise. \\

We define as *event* the transit of a particle though the sensor. The dataset is composed by 514,000 events divided in:
- 385,500 in the *development* set
- 128,500 in the *evaluation* set

Each record contains some features extracted from every signal provided by the RSD sensor. They are:
- pmax: the value of the positive peak of the signal in $mV$
- negpmax: the value of the negative peak of the signal in $mV$
- area: the value of the area under the signal
- tmax: the time between a reference moment and the positive peak in $ns$
- rms: the root mean square of the signal
The signal is represented as a number inside the square brackets at the end of each column name. Therefore, "pmax[0]" means the feature pmax obtained by the signal 0. We are going to use the same notation throughout the report.\\
The position of each event is enforced during the experiments and it is also provided in the development set. On the other hand, the evalution set contains only the identifier of the events. \\

The algorithms have to exploit the information provided by development dataset to predict the position of the events present in the evaluation set. Then, the results are submitted to an online platform, where they are evaluted according to the average Euclidean distance (\ref{eq:EuclideanDist}).
\begin{equation}
    \label{eqn:EuclideanDist}
    d=\frac{1}{n} \sum_i \sqrt{(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2}
\end{equation} 


- analysis of the development dataset
- there are no empty/Nan values
- show an example of the pmax. On the pads there are no measures

## Proposed approach
### Preprocessing
- which type of field carry useful information
- triangle
- outlier (explain the two approaches)
- max pmax
- check feature correlation (?)
- advantages

### Model selection
- which models we have tested (possibly link to papers)
- random forest for the interpretability and for the feature selection

### Hyperparameters tuning
- hyperparameters tuning
- table with what we have tried


## Results
- what we got from the tuning and the various steps in the choices  
- mean error for each cell. Maybe some positions are harder to predict
- naive solution


## Discussion
- comparison with a naive solution
- comparison with the two regressors we tune
- what can be done more(use NN) do more gridsearccv tests(explain the time limit)

## Bibliography