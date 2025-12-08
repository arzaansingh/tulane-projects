To use the files, you need the following steps in the terminal:

Step 1:
```
  pip install poke-env
  
  git clone https://github.com/smogon/pokemon-showdown.git
  
  cd pokemon-showdown

  npm install

  cp config/config-example.js config/config.js

  node pokemon-showdown start --no-security
```

Step 2: In a new terminal window
You can run the following after entering their respective folders:
1. Hierarchical Q: ```python ./run_loop_v11.py random```
2. Linear SARSA: ```python ./train_sarsa_orig.py```
3. DQN: ```python ./run_loop.py```
4. Tabular Q:  
