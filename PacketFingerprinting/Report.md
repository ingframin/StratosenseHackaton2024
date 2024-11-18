# Report
After an intensive week in Alicante, it make sense to write down what went down. This report serves as a logbook, pinpointing parts of the research that should be double checked and realisations that have been made. 

This is the work from Monday 11 November 2024 till Friday 15 November 

## The plan
A big challenge in transmitter localization is the detection of similar packets without overloading a central server with all the data from all the different receivers. The idea is to use neural networks to convert packets received by a receiver into a representation of features, where the goal is to minimise the features, while still be able to detect similar packets. 

To this end, we priorise the true positives, while allowing some False positives. It is important to know that in a real dataset, there are way less positives compared to negatives. 

## Monday 

Today, we stared setting up everything. We set up a slack and a github. 
I started by copying and adapting my ML framework. The prblem is related to modulation classification, but still is a bit different. As I had a ResNet implementation laying around, I choose this as my base.

The idea is to exploit the technique of siamese networks, where we train for the triplet loss. This tripletloss is made for a datapoint existing of an anchor, a positive and a negative. The goals is to make the anchor and the positive close to each other in feature space, while the negative should be far away. 

Except for preparing, nothing much was done this day. At 18h we had a small presentation which we chose to do without slides as there was not a lot to discuss. 

## Tuesday 

Franco made a dataset for me, I spend most of my time debugging and tweaking the network and framework. I did have some progress, but not realy any good results. 
This day was also rather short, as we had a social activity in the afternoon. 


## Wednesday 

I started the training with the dataset of Franco and was amazed by well the model was working. This dataset is a synthetic dataset, therefore I was expecting some high results. After analysing the results I figured out why the accuracy was so high. I USED A DIFFERENT LOSS FOR THE POSITIVE AND NEGATIVE SAMPLES. Therefore acually including the label in the testing. What a shame! 

I fixed the problem, my accuracy dropped and the model was still preforming quite decent. 

After lunch, Matthias provided me with a measured dataset of the 1030MHz. Reading and interfacing with the datset was quite amazing and easily implemented. However, it was clear that there were some issues with training the model. The training loss would go down, but the validation loss would oscilate and not be stable (see slides). I tried out different iterations to see if it was the model itselve that was not capable... I could not figure it out.

Next to the challening model training, I also came up with different ways to determine a loss threshold which is usable at inference to determine weather or not two inputs are similar. This however also proved to be rather challenging. Similarly, a value showing how well the model behaved is also a challenge. 

The day ended quite frustrating as the plug-and-play I hoped for became a big task to tackle. We ended the day with a presentation, where I explained how the triplet loss works. 

[Slides](
https://docs.google.com/presentation/d/1us7NbJh0TX4fuT4sV5_wx8GGSg4WBotxcGRTLB7vLSQ/edit?usp=sharing)


## Thursday

FORGET TRIPLET LOSS! I woke up this morning with the idea that the triplet loss maybe is not that suited for my application. I had the fealing that the entagled loss calculation just added some confusion upon the whole training. 
In the morning, I implemented Contrastive loss. My god and saviour. It worked amazingly. 

Instead of considering triplets, we went back to considering duos. I reused the dataset from the triplets, making 3 duos from each triplet. Therefore, the dataset became bigger, but not as ballanced anymore. 1/3 were positives, 2/3 negatives. But the loss function was amazingly smooth. 

I decided that a for the threshold determination, we should take the loss corresponding to the 99 percentile of the loss of the positives.

I spend the rest of the day training different variations of the models

[Slides](https://docs.google.com/presentation/d/1RsRlqx5a7ytFKGV-bGxe8SEdbXRAME24VwECKUDQVJI/edit?usp=sharing)


## Friday

Two steps today. 

1. Validating the model with a dataset with the same distribution as the real data. In order to compare my compression method against the compression method of Matthias.
2. See if we can compress the features by quantising them.

### Validation
For the validation, Matthias provided me with a dataset of all correlation values between all packets. This lets us figure out some metrics. For the validation, I took the highest preforming model for each number of features.

**Recall:** packets that are considered the same should also be flagged the same by the models, packets that are different, should still be flagged different by the model. Idealy this should be 1.


**Precision:** What is the amount of true positives compared to all positives? This and its inverse gives a nice metric to see what the amount of useful positive there are. This value should be as high as possible, however, the main goal is still for the recall to be one. 

### Compression

I wrote a framework for the compression. The simplest version is to quantize all features with the same amount of bits. These geve us very nice results. We can reduce the bits per feature to 10 bits without losing precision or recall. 

I brievely tried to also do some PCA analysis and do a non uniform quantization: Giving a higher amount of bits to the features with the highst variance, but this did not work out that well. Some more in depth research is needed here. The quantization should incorperate the mean of the features aswell. 


[Slides](https://docs.google.com/presentation/d/14Ubxj8wgLYhL452CZ42AslycgYlqydPuE4LnSzPnWyU/edit?usp=sharing)


# Next steps

- Redo experiments but with different, more lose threshold. The goal is to detect all matches, while allowing (limited) false positives
    - Determin th correctly! not >0.90 correlation, but include flag if considered right or wrong!
    - Instead of 99 percentile of loss, what happes is we go further, what are the other metrics giving us.
    - What is the th for which the recall is as high as possible as well as the precision to be as high as possible. 
- Train with more data, but not balanced. Train on the raw dataset, see if this still provide us with nice results. We have the correlation between a lot of diffent packets. if we sample 100 000 of them to train. maybe this will yield a better solution
- Optimise the model a bit more. The model is fine, but redo some search
- Non uniform feature quantisation
    - Based on variance, but first normalise the features before quantization.
    - This might screw up the threshold! should also be checked with the th of the training data (repeat the same th determination, but with quantized features.)
- Getting a true dataset 
    - Different receivers
    - True labels instead of assumed cuttoff
    - Preform plausability checks before correlation 
        - In a realistic time frame
        - ...
- Find a metric for the *True* compression factor.
    - Packets can be smaller then 512 samples
    - How many samples do we actually need for the correlation? 
    - The bits of the features should add up to a multiple of bytes for transmission. 
