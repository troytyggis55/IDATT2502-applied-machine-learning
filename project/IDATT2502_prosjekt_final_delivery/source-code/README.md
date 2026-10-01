# Self play training utilzing SB3 PPO with a custom Gymnasium environment 

## Venv setup

### Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Training
Edit the parameters in train_model.py where you see fit.
The default values like total_traningsteps and milestone_every_n_steps are quite small so its recommended to increase them to observe acutal learned behavior.

```bash
python train_model.py
```
The model pool and milestone models will be saved in separate folders.

## Evaluate models in folder
If you want to for example evaluate the models in traned_models/milestone_models for 100 episodes:
```bash
python evaluate_models.py trained_models/milestone_models 200
```
**Warning: Time complexity here is ((number of evaluated models)^2 * episodes). Evaluating is CPU intensive and can take a lot of time.**

### Plot evaluated results
```bash
python visualize_grid.py trained_models/milestone_models/evaluated_results.txt
python visualize_data.py trained_models/milestone_models/evaluated_results.txt
```
Images called GridPlot and LinePlot with the current date will be saved in the same folder.


## Generate heatmap between two models for a given number of episodes
```bash
python collect_heatmap_data.py trained_models/milestone_models/100000.zip trained_models/milestone_models/200000.zip 250
```

### Visualize generated heatmap
```bash
python visualize_heatmap.py heatmap_data_100000_200000.txt
```
Image called heatmap_data_100000_200000_heatmap.png will be saved in the same folder.

## Watch models compete
```bash
python play_environment.py trained_models/milestone_models/100000.zip trained_models/milestone_models/200000.zip
```

## Play against the model
```bash
python play_environment.py trained_models/milestone_models/200000.zip
```
Use arrow keys to move right player.

## Play 2 player
```bash
python play_environment.py
```
Use WASD to move left player and arrow keys to move right player.