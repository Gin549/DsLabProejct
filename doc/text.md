# Data Science Lab: Process and methods Group project Winter Call

## Abstract


## Problem overview
\label{sec:problemOverview}
The group project consists in predicting the position of a particle when it traverses a RSD sensor. The sensor is flat, hence the position is represented by $(x, y)$ values. 
The task has to be performed through a multi-output regression pipeline, one per coordinate. \\

The Resistive Silicon Detector(RSD) sensor is composed by 12 metallic *pads*. They have an asterisk shape and they can be clearly distinguished in Figure \ref{fig:rsd}.\\
Each pad records a signal that it is trasfered and stored by the system. Due to hardware constrains, not all the measures transfered are meaningful, but 6 of the 18 readings are noise. \\

We define as *event* the transit of a particle though the sensor. The dataset is composed by 514,000 events divided in:
- 385,500 in the *development* set
- 128,500 in the *evaluation* set

All the fields have a not null valid value. Each record contains some features of an event extracted from every signal provided by the RSD sensor. They are:
\label{lst:typeFeature}
- pmax: the value of the positive peak of the signal in $mV$
- negpmax: the value of the negative peak of the signal in $mV$
- area: the value of the area under the signal
- tmax: the time between a reference moment and the positive peak in $ns$
- rms: the root mean square of the signal
The signal is represented as a number inside the square brackets at the end of the column name. Therefore, "pmax[0]" means the feature pmax obtained by the signal 0. We are going to use the same notation throughout the report.\\
The position of each event is enforced during the experiments and it is also provided in the development set. On the other hand, the evalution set contains only the identifier of the events. \\

The algorithms have to exploit the information provided by development set to predict the position of the events present in the evaluation set. Then, the results are submitted to an online platform, where they are evaluted according to the average Euclidean distance (\ref{eq:EuclideanDist}).
\begin{equation}
    \label{eqn:EuclideanDist}
    d=\frac{1}{n} \sum_i \sqrt{(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2}
\end{equation} 

The development set can be used to make some analysis, in particular according to the position.\\
A first observation is that for every $(x, y)$ present in the records, there are 100 events. The x and y values present are all the integers in the range $(200, 600)$ with a step 5. Furthermore, not all the possible $(x, y)$ combinations are present as reported in Figure \ref{fig:pmax[5]_heatmap}. In particular, the positions of the pads can be clearly identified by the complete absence of events. Only the shape of a few of them is complete because the positions collected are only of the central zone of the sensor. This is because when a particle hits the sensors, it is not passing through it and the event is not valid. \\%TODO: add citation to a paper)

Figure \ref{fig:pmax[5]_heatmap} represents the mean pmax aggregating by $(x, y)$. Comparing the heatmaps of the average pmax of different signal was useful to discern the noise. An example is the different between the figures \ref{fig:pmax[5]_heatmap} and \ref{fig:pmax[0]_heatmap}. The latter is clearly noise. \\
(put the Figures of pmax[5] and pmax[0])
The same heatmaps were used to identify the pad corresponding to each signal. The mean value of a specif pmax for a given $(x, y)$ is high only if the passing position of the particle is close to the pad. For instance, the pad corresponding to signal 5 is the one surrounded cells encoded with brighter colors in Figure \ref{fig:pmax[5]_heatmap}.
The pad and the corresponding signal number are represented in Figure \ref{fig:rsd_pad_num}.\\

The difference between the noise and the signal can be also observed in the distributions of the values. A typical pmax probability function of a pad is represented in Figure \ref{fig:pmax[5]_distr}. The noise signals have completely different shapes and range of values. Most of the values have a low magnitude. This occurs because most of the time the pad is distant from the passing position.\\

The same analysis were perfomed on the other types presented in the listing \ref{lst:typeFeature}. Similar results were obtained for negpmax and area, and tmax. This confirmed that the features that share the same trailing "[n]" derive from the same signal. The tmax fields give less clear visualizations than the other three. On the other hand, the heatmaps of the rms values do not present any visual pattern and the values are in the order of a few $mV$. The reason is that only a small portion of the signal can be characterized by peaks. Most of time, the signal has a small magnitude of random noise. As a consequence, the latter part prevails in the computation of the rms.\\

Another consideration that can be made is on the distributions of negpmax. Its density functions have an insolit number of outliers with respect to the other features(see Figure \ref{fig:negpmax_boxplots}). 

## Proposed approach
### Preprocessing

We have seen from a visual prospective which are the features that bring most information. These intuitive considerations were confirmed by an analysis of the importances obtained by a Random Forest regressor with default parameters. Only the pmax, negpmax, and the area of the signal derived from the pads were significantly used by the algorithm.\\ %TODO: create/add the feature with the importance by type
An interesting exception is the signal 15. Even if it should be only random noise, the $pmax[15]$ is taken in consideration. Observing Figure \ref{fig:pmax[15]_heatmap}, we see that the values are on average higher on the borders of the sensor and lower near the metal of the pads. \\
For these reasons, we kept only pmax, negpmax, and the area of the signal corresponding to a pad and the feature $pmax[15]$.\\

%TODO: add how we removed the outliers

In Section \ref{sec:problemOverview}, we have seen how the closer the position to a pad, the higher is the corresponding pmax value. For this reason, we introduced a new feature that is the maximum pmax among for each event. Then, we added the normalized value of every pmax by the maximum pmax of that event. This is done because this information provide to the algorithm a way to infer how much the position is near to a pad in comparison to the closest.\\
  
 
- we remove the outliers (how we did it, the consideration to do it only )
- outlier (explain the two approaches)
- triangle ( for the moment I don't put this because the prof put in the report only what he has used at the end). I don't know if it is the right thing to do put here the fact that we discarded the option seeing the results in \ref{results}
- triangles (if we haven't reached the 4 pages with the rest)
- check feature correlation (?)

### Model selection
- which models we have tested (possibly link to papers)
- random forest for the interpretability and for the feature selection (explain the regressor and not the one used for regression)
- advantages of voting regressor (maybe cite somethings)
- what do we expect

### Hyperparameters tuning
- hyperparameters tuning
- table with what we have tried


## Results
- that the addition of feature pmax_normalized improve all the algorithms (I think that we can avoid saying that the removal of the noise improved the results because it is obvious)
- what we got from the tuning and the various steps in the choices  
- mean error for each cell. Maybe some positions are harder to predict(probabily close to the sensors)
- naive solution
- comparison with a perfect random forest (comparison with the uncertainty of the tool that fix the position (we don't know what is, I don't think we can do it)
- comparison with a result with our configurations with a run on the original features without any of our changes. (the evaluation set gives worst results. Locally RT is really bad. Not so much the other two(on 80%), but they give worst performances on the evaluation set)
- both validation performance and public performance
- comparison with a naive solution
- comparison with the two regressors we tune
- comparison with / without the triangle(if we haven't reached the 4 pages with the rest)
- comparison with/without the norm 
## Discussion
- what can be done more(use NN) do more gridsearccv tests(explain the time limit)
- what went well
- consideration on the problem(maybe compare with the uncertainty of the tool used to fix the path of the particles)

## Bibliography