# Anime Face Generation

- `input`: random number
- `ouput`: Anime face
- `Implementation requirement`: DCGAN & WGAN & WGAN-GP
- `Target`: generate 1000 anime face images

## Evaluation metrics

`FID (Frechet Inception Distance) score`

- Use another model to create features for real and fake images
- Calculate the Frechet distance between distribution of two features

## Suggested baselines
 
- `Simple`: FID <= 30000, AFD >= 0
- `Medium`: FID <= 12000, AFD >=0.4
- `Strong`: FID <=10000, AFD >=0.5
- `Boss`: FID <= 9000, AFD>=0.6