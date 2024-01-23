# Data Science Lab: Process and methods Group project Winter Call

## Abstract
RSD is a sensor that tracks the position of passing particles. Throughout this report, we propose a data science pipeline that uses as input several features extracted by the RSD signals, removes the outliers and the noise, and extracts the maximum positive peak and the relative values with respect to it. The preprocessed data is provided to three regression models. The results exceed a naive baseline and the performance of the same models on the original data.

## Problem overview
\label{sec:problemOverview}
The group project consists in predicting the position of a particle when it traverses an RSD sensor. The sensor is flat, hence the position is represented by $(x, y)$ values. 
The task has to be performed through a multi-output regression pipeline, one per coordinate. \\

The Resistive Silicon Detector(RSD) sensor is composed of 12 metallic *pads*. They have an asterisk-shape and they can be clearly distinguished in Figure \ref{fig:rsd}.\\
Each pad records a signal that is transferred and stored by the system. Due to hardware constraints, not all the measures transferred are meaningful, but 6 of the 18 readings are noise. \\

We define as *event* the transit of a particle through the sensor. The dataset is composed of 514,000 events divided into:
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
The position of each event is enforced during the experiments and it is also provided in the development set. On the other hand, the evaluation set contains only the identifier of the events. \\

The algorithms have to exploit the information provided by the development set to predict the position of the events present in the evaluation set. Then, the results are submitted to an online platform, where they are evaluated according to the average Euclidean distance (\ref{eq:EuclideanDist}).
\begin{equation}
    \label{eqn:EuclideanDist}
    d=\frac{1}{n} \sum_i \sqrt{(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2}
\end{equation} 

The development set can be used to make some analysis, in particular, according to the position.\\
A first observation is that for every $(x, y)$ present in the records, there are 100 events. The x and y values present are all the integers in the range $(200, 600)$ with step 5. Furthermore, not all the possible $(x, y)$ combinations are present as reported in Figure \ref{fig:pmax[5]_heatmap}. In particular, the positions of the pads can be identified by the complete absence of events. Only the shape of a few of them is complete because the positions collected are only of the central zone of the sensor. This is because when a particle hits the sensors, it does not pass through it and the event is not valid. \\%TODO: add a citation to a paper)

Figure \ref{fig:pmax[5]_heatmap} represents the mean pmax aggregating by $(x, y)$. Comparing the heat maps of the average pmax of different signals was useful for discerning the noise. An example is the difference between the figures \ref{fig:pmax[5]_heatmap} and \ref{fig:pmax[0]_heatmap}. The latter is clearly noise. \\
(put the Figures of pmax[5] and pmax[0])
The same heat maps were used to identify the pad corresponding to each signal. The mean value of a specific pmax for a given $(x, y)$ is high only if the passing position of the particle is close to the pad. For instance, the pad corresponding to signal 5 is the one surrounded by cells encoded with brighter colors in Figure \ref{fig:pmax[5]_heatmap}.
The pad and the corresponding signal number are represented in Figure \ref{fig:rsd_pad_num}.\\

The difference between the noise and the signal can be also observed in the distributions of the values. A typical pmax probability function of a pad is represented in Figure \ref{fig:pmax[5]_distr}. The noise signals have completely different shapes and range of values. Most of the values have a low magnitude. This occurs because most of the time the pad is distant from the passing position.\\

The same analyses were performed on the other types presented in the listing \ref{lst:typeFeature}. Similar results were obtained for negpmax, area, and tmax. This confirmed that the features that share the same trailing "[n]" derive from the same signal. The tmax fields give less clear visualizations than the other three. On the other hand, the heat maps of the rms values do not present any visual pattern and the values are in the order of a few $mV$. The reason is that only a small portion of the signal can be characterized by peaks. Most of the time, the signal has a small magnitude of random noise. As a consequence, the latter part prevails in the computation of the rms.\\

Another consideration that can be made is on the distributions of negpmax. Its density functions have an unusual number of outliers compared to the other features(see Figure \ref{fig:negpmax_boxplots}). 

## Proposed approach
### Preprocessing

We have seen from a visual perspective, which are the features that contain most of the information. These intuitive considerations were confirmed by an analysis of the importances obtained by a Random Forest regressor with default parameters. Only the pmax, negpmax, and the area of the signal derived from the pads were significantly used by the algorithm.\\ %TODO: create/add the feature with the importance by type
An interesting exception is the signal 15. Even if it should be only random noise, the $pmax[15]$ is taken into consideration. Observing Figure \ref{fig:pmax[15]_heatmap}, we see that the values are on average higher on the borders of the sensor and lower near the metal of the pads. \\
For these reasons, we kept only pmax, negpmax, and the area of the signal corresponding to a pad and the feature $pmax[15]$.\\

%TODO: add how we removed the outliers

In Section \ref{sec:problemOverview}, we have seen how the closer the position to a pad, the higher the corresponding pmax value. For this reason, we introduced a new feature that is the maximum pmax for each event. Then, we added the normalized value of every pmax by the maximum pmax of that event. This is done because this information provides the algorithm a way to infer how near is the position to a pad in comparison to the closest.\\
  
 
- we remove the outliers (how we did it, the consideration to do it only )
- outlier (explain the two approaches)
- triangle ( for the moment I don't put this because the professor put in the report only what he has used at the end). I don't know if it is the right thing to do to put here the fact that we discarded the option seeing the results in \ref{results}
- triangles (if we haven't reached the 4 pages with the rest)
- check feature correlation (?)

### Model selection
We have tested the following models:
- Random forest(RT): it is an ensemble machine learning technique used for classification and regression. During the training phase, several decision trees are created on different random datasets, sampled with replacement from the original data. For a regression problem, the output of the random forest is the mean of the predictions of the single decision trees. Every split is learned during training considering a criterion, e.g. mean squared error, and possibly on a random subset of the features \cite{paper:randomForest}] \cite{paper:extrRandTree}.\\
The usage of multiple decision trees leads to a model more robust to noise \cite{paper:RandomForest}. The performance and the training times depend on the number of decision trees. The improvement provided by an increase in the number of decision trees is significant up to a certain number \cite{paper:howManyTree}.\\
Even if the model is not interpretable as a decision tree, the overall importance of the features can still be obtained.

- Extra-trees regressor(ET): it is an ensemble machine learning technique. There are only two main differences with the random forest method. The first is that every decision tree is built using the whole learning sample. The second is that the split is randomly selected from a uniform distribution inside the candidate feature's empirical range. Then among all the random splits, the best one is chosen and used to grow the tree. \\
The computational efficiency of the algorithm is an important advantage of this model \cite{paper:extrRandTree}.


- Voting regressor: it is an ensemble machine learning technique. It is based on the simple idea of using the average of different regression models. In this way, the advantages of multiple models can be combined. In this case, we used the mean of the random forest and the extra tree regressors. //TODO: cite a paper and change the name, I don't think that voting regressor is the official one


- advantages of voting regressor (maybe cite something)
- what do we expect

### Hyperparameters tuning
The tuning was performed on the hyperparameters of the RT and the ET. As a consequence, the voting regressor used the best performing configurations of the other two models. \\
The RT and the ET algorithms share all the parameters we considered for the tuning. They are:
- the number of estimators *n_estimators*
- the number of features considered at each split *max_features*
- the maximum depth of each decision tree *max_depth*
- the splitting criterion *criterion*

Over a certain threshold, the increase in the number of decision trees does not improve significantly the performance, while extending considerably the computational time \cite{paper:howManyTree}.// 
We defined that for this problem it is not worth using a number of estimators over 130.\\  
For this reason, we tested using a grid search values of n_estimators near 100.\\

The maximum depth obtained by the RT and the ET with default parameters is respectively 38 and 54. We tested the configurations with max_depth None, 60%, and 80% of the previous values. 

We divided the development dataset 80/20. On the 80%, we performed a cross validation using 4 folds. The metric used was the average Euclidean distance (\ref{eq:EuclideanDist}).\\
The 20% was used to test the best performing models, before using all the labeled data to build the final regressors.


||parameter|| Values RT || Value ET
||n_estimators|| = [90,100,120]
||criterion|| = "squared_error""
||max_features|| = ["sqr",1.0]
||max_depth|| = 22,30,None

## Results
The tuning showed that for RF and ET the best configuration of hyper-parameters is very similar, the only difference was in "max features", 1.0 for ET and "sqrt" for RF.
The best configuration for RF was:
_n_estimators:100
_criterion: "squared_error"
_max_features: "sqrt"
_max_depth:None
The best configuration for RF was:
_n_estimators:100
_criterion: 1.0
_max_features: "sqrt"
_max_depth:None
We did the tuning on the training test using cross-validation, then we tested the results of the two regressors, using the best parameters configuration, on the test set, the 
distance obtained for the RT scirca = , for ET scirca= , for VR the distance was scirca=
With the configuration obtained, we trained the VR on the full development set and used it to label the data contained in the evaluation set, the score obtained on the public
scoreboard was scirca = .

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
- comparison with the baseline on the online platform
- we trained the model on all the dataset at the end
## Discussion
- what can be done more(use NN) do more gridsearccv tests(explain the time limit)
- what went well

## Bibliography
