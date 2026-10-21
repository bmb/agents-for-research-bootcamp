# Project rules

## Environment
- Install packages with pip or conda into a virtual environment in this folder.

## Data
- Data files live in the data/ folder.
- Never modify raw downloaded files. Save cleaned versions as new files.
- After any filtering step, state how many rows remain.

## Results
- Store results in the results/ folder.

## Analysis
- Use random_state=0 anywhere randomness is involved.
- When reporting a model's accuracy, always also report the accuracy of predicting the most common outcome for every observation.

## Workflow
- Propose a plan before creating or editing files.
- After running a command, show the exact command and summarize the result.