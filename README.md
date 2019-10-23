# mitiBot
A Graph based machine learning approach to bot mitigation systems.

### Training

You can train the model in 2 ways, as it has PHASE 1 (UNSUPERVISED) and PHASE 2 (SUPERVISED)

This will peform both the phases one by one.
```
python3 model.py --train
```

If you want to perform the 2 phases separately
```
python3 model.py --phase1
```

and

```
python3 model.py --phase2
```

Once trained, it creates the pickle files of the model and saves it in the `saved` folder which is then used for the testing.


### Testing

```
python3 model.py --test
```
