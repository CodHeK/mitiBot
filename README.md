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

### DBSCAN + Logistic Regression

- Initially considering `15k` sized test dataset containing `5k bot flows` and `10k non-bot flows`
- Cluster sizes:
  ![cluster_png]('./screenshots/cluster_sizes.png')
- Test run:
  ![dbscan_lr_test]('./screenshots/dbscan_lr_test.png')
- Test time:
  ```
   Avg: 50s-60s
  ```


### Reference

The following code is the implementation of the [paper](https://arxiv.org/pdf/1902.08538.pdf)
with slight modifications.
