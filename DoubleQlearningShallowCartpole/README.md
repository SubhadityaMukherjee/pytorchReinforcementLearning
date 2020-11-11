# Double Q learning shallow

## What
- Double Q learning
- No Neural network
- Every bit of the code is explained in comments so look at that

## How to run?
```py
python main.py
```

More args are available below

## Arguments available

- n : No of epochs to run
- e : OpenAI gym env name
- a : (ɑ) Learning rate 
- g : (ɣ) Term to choose bw exploration exploitation
- log : How many epochs to log after
- early : Stop if the score isnt improving for et epochs
- et : No of epochs to consider for early stopping

## Environment
- Cartpole problem
- Continuous
- We need to balance a cartpole and prevent it from falling

## Actions
- Move Left or right 

## Parameters
- cart position --- -4.8 - 4.8
- cart velocity --- -inf - inf
- pole angle    --- -41.8 - 41.8
- pole velocity --- -inf - inf
 
