# Persona_leakage_and_defense_in_GPT-2
Code for NAACL 2022 paper "You Don’t Know My Favorite Color: Preventing Dialogue Representations from Revealing Speakers’ Private Personas".


### Environment

* tqdm
* numpy
* pandas
* torch >= 1.7.0
* transformers == 3.4.0


### Data
All data files used are put in the [processed_persona](https://github.com/HKUST-KnowComp/Persona_leakage_and_defense_in_GPT-2/tree/main/processed_persona) folder. You can find more descirptions about data in the folder.


### Training
First, make sure you setup proper path of processed_persona in config.py.

To train the defender GPT-2, run 
```
python training_dialogpt_defense.py
```
Then you can use the defender GPT-2 to train the attacker (or you can just train the attacker on the pre-trained GPT-2 without defense):
```
python training_attacker.py
```

### Evaluation
Run:
```
./eval.sh
```
