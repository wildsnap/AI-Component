# WildSnap for AI-Component

This repository project uses MLflow to track experiments related to weight processing. It includes scripts to process weighted and unweighted data, along with a Jupyter Notebook to run and visualize the workflow.


## Project Structure


```
code/
├── main.ipynb       # Jupyter notebook for running experiments
├── weight.py        # Script to process weighted data
├── unweight.py      # Script to process unweighted data
├── mlruns/          # MLflow experiment logs (ignored in Git)
```

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```


## How to Run

### 1. Run the Notebook

You can launch the Jupyter notebook and run all cells:

```bash
jupyter notebook code/main.ipynb
```

Make sure MLflow is logging the experiments properly within the notebook.


### 2. Run Python Scripts

If your scripts include MLflow tracking:

```bash
python code/weight.py
python code/unweight.py
```

Each script will run a job and log it under the `mlruns/` folder (if using local tracking).

---

### 3. View MLflow UI (Optional)

To view the MLflow UI and see experiment results:

```bash
mlflow ui
```

Then open your browser and go to:
[http://localhost:5000](http://localhost:5000)


## Notes

* `code/mlruns/` is excluded from version control. Each user should run the scripts to generate their own logs.
* Make sure your MLflow environment is properly configured if using a remote tracking server.
