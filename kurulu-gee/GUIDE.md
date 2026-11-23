# Kurulu Gee – Project Guide Book
### Ensemble Learning Dashboard on Sri Lankan Bird Diversity

> **For Copilot:** This file is the project specification. The project has three code files:
> `bird_ensemble.ipynb` (data + models), `app.py` (Streamlit dashboard) and `requirements.txt`.
> Follow the steps in order. Do not change the train/test split, the feature list or the file
> names in `artifacts/` unless the guide says so. The dashboard depends on them.

---

## 0. The project in one paragraph

We use 1.5 million bird sightings from Sri Lanka (2014–2024). Each sighting already comes with
satellite and climate values (vegetation, rainfall, temperature, elevation and more). We group the
sightings into **one row per district per month** and predict **species richness**, which means
*how many different bird species were seen there that month*. We compare **Bagging**, **Boosting**,
**Stacking** and a **hybrid Random Forest + Neural Network**, explain why each one did better or
worse, and show everything in a **Streamlit dashboard**.

**Story / pitch:** *"Where and when is Sri Lanka's bird life richest, and what drives it? Forest
cover? Rain? Altitude? The migrant season?"* This is useful for eco-tourism, birdwatchers and
conservation planning.

---

## 1. Tonight's timeline (deadline 11:59 pm)

| Time | Task | Guide section |
|---|---|---|
| 19:00 – 19:20 | Set up VS Code, environment and data | 2, 3 |
| 19:20 – 20:30 | Run the notebook cell by cell and understand each step | 4 |
| 20:30 – 21:00 | Write your interpretation in the notebook | 5 |
| 21:00 – 21:30 | Run the dashboard and adjust the text in it | 6 |
| 21:30 – 22:45 | Record the 3-minute video | 8 |
| 22:45 – 23:30 | Final check and submit (optional: deploy) | 7, 9 |
| 23:30 – 23:59 | **Spare time. Submit before this.** | |

---

## 2. Setup (VS Code)

**Yes, VS Code is a good choice.** It runs the notebook (.ipynb) *and* the Streamlit dashboard in one
place, and Copilot works in both. (Google Colab can't easily run a Streamlit app.)

### 2.1 Install VS Code extensions
- **Python** (Microsoft)
- **Jupyter** (Microsoft)
- **GitHub Copilot** + **GitHub Copilot Chat**

### 2.2 Make the project folder

```
kurulu-gee/
├── GUIDE.md               ← this file
├── bird_ensemble.ipynb    ← notebook (submission)
├── app.py                 ← dashboard
├── requirements.txt
├── data/                  ← put the downloaded dataset here
└── artifacts/             ← created automatically by the notebook
```

### 2.3 Create a virtual environment and install packages
Open the folder in VS Code → **Terminal → New Terminal**:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks the activate script, run this first:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.4 Select the kernel
Open `bird_ensemble.ipynb` → click **Select Kernel** (top right) → **Python Environments** → choose
**.venv**.

---

## 3. Get the data

Dataset: **Sri Lanka Bird Diversity Dataset with Environmental and Climate Features**
Link: https://huggingface.co/datasets/Bird-Diversity/Diversity (licence CC BY 4.0)

**Option A: download in the notebook (easiest).** In Step 2 of the notebook, uncomment these two lines:
```python
from datasets import load_dataset
raw = load_dataset("Bird-Diversity/Diversity", split="train").to_pandas()
```
and comment out the `DATA_PATH` lines below them.

**Option B: download by hand.** On the dataset page, open **Files and versions**, download the
data file into `data/`, then set `DATA_PATH` in Step 2 to that file name (`.csv` or `.parquet`
both work).

**Columns you will use**

| Column | Meaning | How we use it |
|---|---|---|
| `verbatimScientificName` | species name | count unique species → **target** |
| `stateProvince` | district (the card says "province", but there are 25, so they are districts) | grouping + feature |
| `eventDate` | sighting date | year and month |
| `decimalLatitude/Longitude` | location | district centre + map |
| `NDVI` | vegetation greenness (0 = bare, 1 = dense forest) | habitat |
| `elevation_meters` | height above sea level | habitat |
| `temp_mean`, `rainfall`, `wind_mean`, `humid_mean`, `shortwave_radiation` | weather | climate |
| `avg_rad` | probably night-time lights (a sign of urbanisation). Check this | human impact |
| `lka_general_2020` | probably 2020 population density. Check this | human impact |
| `Total_Aerosol_Extinction` | air haze / pollution | air quality |

We skip the separate aerosol columns (carbon, dust, SO₂ and so on) because they overlap with
`Total_Aerosol_Extinction`.

---

## 4. The notebook, step by step

Run one cell at a time (**Shift + Enter**). Read the "Why" for each step before running it, because
this is what you'll explain in the video and viva.

### Step 1 – Imports
Loads the libraries. `SEED = 42` makes results repeatable: the same split and the same random
trees every run.

### Step 2 – Load the data
Loads about 1.5 million rows. It may take 30–60 seconds.
✅ **Check:** `raw.shape` shows roughly (1,552,048, 24).

### Step 3 – Quick look
Prints column types, missing values, and the number of districts and species.
✅ **Check:** about 429 species. If the district count is above 25, Step 4 fixes the spelling
differences.

### Step 4 – Clean
- **District names:** "Colombo District" and "Colombo" become the same value.
- **Dates:** some dates are ranges (`2019-05-01/2019-05-31`), so we keep the first date.
- Rows with no district, date or species are dropped.

✅ **Check:** the printed district list has about 25 clean names. If you see odd names
(like provinces), tell Copilot: *"Show me `df['district'].value_counts()` and help me map these to
the 25 districts."*

### Step 5 – Build the modelling table (most important step)
We turn 1.5 million sightings into **about 3,000 rows: one per district per month**.

| New column | Meaning | Why |
|---|---|---|
| `species_richness` | number of unique species | **the target (y)** |
| `n_records` → `log_records` | number of sightings that month | **survey effort**: more birdwatching finds more species. Log is used because the 1,000th sighting adds fewer new species than the 10th |
| `month_sin`, `month_cos` | month as a point on a circle | so the model knows December is next to January |
| `migrant_season` | 1 for Sep–Apr | winter migrant birds come to Sri Lanka in these months |
| environment means | average NDVI, rain and so on for that district-month | habitat and climate |

District-months with fewer than 20 sightings are removed, because their richness is mostly noise.

### Step 6 – Exploratory plots
Three charts: richness by month, richness by district, and effort vs richness.
📸 **Screenshot these for the video.**

### Step 7 – Features, target and split
- **80% train / 20% test**, with `random_state=42`.
- **Every model is tested on the same 20%**, so the comparison is fair.
- The preprocessor **one-hot encodes** the district (one column per district) and **scales**
  the numbers. The neural network needs scaling, and tree models aren't affected by it.

### Step 8 – The models (the core of the assignment)

| Family | Model | Idea in simple words |
|---|---|---|
| Baseline | Mean baseline | always guesses the average. Every model must beat this |
| Baseline | Linear Regression | a straight-line relationship |
| Baseline | Single Decision Tree | one tree. Flexible, but it **overfits** (memorises noise) |
| **Bagging** | Bagging (Trees) | 200 trees, each trained on a **random resample** of the data, then **averaged** |
| **Bagging** | Random Forest | bagging **plus** each split only looks at a random subset of features, so the trees are more different from each other |
| **Boosting** | AdaBoost | trees built one after another. Each one focuses on the rows the previous ones got wrong |
| **Boosting** | LightGBM | gradient boosting: each new tree predicts the **remaining error** of the trees before it |
| Neural net | ANN (MLP) | 2 hidden layers (64 → 32 neurons) that learn smooth non-linear patterns |
| **Stacking** | RF + LightGBM + ANN → Ridge | three different models predict, and a simple "meta-model" learns **how much to trust each one**. It uses 5-fold cross-validation so the meta-model learns from honest predictions |

**Key theory to remember:**
- **Bagging reduces variance.** One tree changes a lot when the data changes, and averaging many
  trees cancels that out.
- **Boosting reduces bias.** It keeps fixing mistakes step by step, but it can overfit on small
  data if it runs too many rounds.
- **Stacking** only helps if the base models make **different kinds of mistakes**.

### Step 9 – Train and evaluate
Trains every model and prints R². This may take 1–3 minutes.

### Step 10 – Hybrid RF + ANN (the "hybridisation" requirement)
Two hybrids:

**A. Residual hybrid.** Random Forest predicts first. Then an ANN learns **what the RF got wrong**
(the residual = actual − RF prediction). Final answer = RF prediction + ANN correction.
- Trees predict in "steps" (flat blocks), while an ANN produces smooth curves, so the ANN can
  smooth out the RF's step-shaped errors.
- We use **out-of-fold** RF predictions (`cross_val_predict`) to compute the residuals. If we used
  the RF's predictions on its own training data, the errors would look tiny (it has memorised
  that data) and the ANN would learn nothing useful. This is called **avoiding data leakage**.

**B. Weighted blend.** Final answer = `w × RF + (1 − w) × ANN`. The best `w` is chosen on
training data (out-of-fold), **never** on the test set.

### Step 11 – Compare
A results table sorted by RMSE, and a coloured bar chart. 📸 **Screenshot for the video.**

| Metric | Meaning | Better |
|---|---|---|
| MAE | average error in "number of species" | lower |
| RMSE | like MAE, but big mistakes count more | lower |
| R² | share of the variation the model explains (1 = perfect, 0 = no better than the average) | higher |

### Step 12 – Predicted vs actual
Dots close to the red diagonal line mean good predictions. Look at where the model misses, for
example whether it under-predicts very rich months.

### Step 13 – Feature importance
**Permutation importance:** shuffle one feature and see how much the test error grows. A bigger
increase means the model relies on that feature more. This is the "why" behind your results.

### Step 14 – Save artifacts
Saves the models and tables to `artifacts/`. **The dashboard reads these files, so run this cell
before starting the dashboard.**

---

## 5. Writing the interpretation (where the marks are)

Fill in the last markdown cell using **your real numbers**. Templates are below. Pick the ones that
match your results and edit them.

**Bagging vs single tree**
> The single decision tree had RMSE ___ while Random Forest had ___. One deep tree memorises
> noise in the training data (high variance). Bagging trains many trees on bootstrap samples and
> averages them, which cancels out their individual errors.

**Random Forest vs plain Bagging**
> RF was ___ than plain bagging because it also picks random features at each split. That makes
> the trees less similar, so averaging them removes more error.

**Boosting vs Bagging**
> *If LightGBM won:* boosting fixes errors step by step, so it captured patterns (e.g. the
> effort × season interaction) that averaging trees missed.
> *If RF won:* our table has only about 3,000 rows. Boosting can overfit small, noisy data,
> while bagging is more stable.

**ANN alone**
> The ANN scored ___. Neural networks usually need more data than tree models, and on about
> 3,000 tabular rows, tree ensembles are typically stronger.

**Stacking**
> Stacking scored ___ vs the best single model ___. It helps only when the base models' errors
> are different. RF and LightGBM are both tree models, so their errors overlap. The ANN adds a
> different kind of model.

**Hybrid RF + ANN**
> The residual hybrid ___ (improved / didn't improve) RF. The blend gave RF a weight of ___, which
> means the data favoured the ___ model.

**Feature importance (connect to Sri Lanka)**
> The most important feature was ___. Survey effort matters because more birdwatching finds more
> species. `migrant_season` / month matters because winter migrants arrive from about September
> to April. NDVI and elevation reflect habitat: forested and hill-country areas hold more endemic
> species.

**Limitation (one line shows maturity)**
> Citizen-science data has sampling bias: popular birding spots (e.g. Sinharaja, Yala) are
> over-recorded, so richness partly reflects where birders go.

---

## 6. The dashboard

In the VS Code terminal (with `.venv` active):
```bash
streamlit run app.py
```
It opens in your browser at http://localhost:8501.

| Tab | What it shows | Video use |
|---|---|---|
| 📊 Overview | key numbers, sightings map, richness by month and by district | the "story" |
| 🏆 Model comparison | bar chart + table, switch between RMSE / MAE / R² | the results |
| 🔮 Predict | pick a district, month and effort level → every model's prediction | **live demo**: show Jan vs Jul, Nuwara Eliya vs Colombo |
| 💡 Insights | feature importance + takeaways | the "why" |

**Edit the "Key takeaways" text in the Insights tab** (bottom of `app.py`) to match your results.

---

## 7. Optional: deploy a live link (only if you have time after the video)

1. Create a GitHub repo and push the folder. Include `artifacts/` (the models file is about 20 MB)
   but **not** `data/` or `.venv/`. Add them to `.gitignore`.
2. Go to https://share.streamlit.io → sign in with GitHub → **New app** → pick the repo →
   main file `app.py` → **Deploy**.
3. Put the app link and the repo link in your submission.

---

## 8. Video script (max 3:00, voice-over)

| Time | Show on screen | Say (adapt it) |
|---|---|---|
| 0:00–0:20 | Title / dashboard home | "Sri Lanka has over 400 bird species and many are found nowhere else. But where and when is our bird life richest, and why? Meet Kurulu Gee." |
| 0:20–0:45 | Overview tab, map | "I used 1.5 million real bird sightings from 2014 to 2024, each with satellite and climate data such as vegetation, rainfall and elevation. I grouped them into district-months and predicted how many species are seen." |
| 0:45–1:30 | Notebook: model list → comparison chart | "I compared bagging (Random Forest), boosting (LightGBM and AdaBoost), stacking, and a hybrid where a neural network corrects the Random Forest's mistakes. The best model was ___ with R² ___." |
| 1:30–2:00 | Comparison tab | "Why? A single tree overfits, and bagging fixes that by averaging. ___ (your boosting / stacking / hybrid finding)." |
| 2:00–2:35 | Predict tab: change the month and the district | "Watch: in January, during migrant season, predicted species go up. Nuwara Eliya vs Colombo shows the habitat effect." |
| 2:35–2:55 | Insights tab | "The biggest drivers are ___, ___ and ___. Survey effort matters too, which shows citizen science shapes what we know." |
| 2:55–3:00 | Dashboard | "Kurulu Gee: ensemble learning for Sri Lanka's birds. Thank you." |

**Recording tips:** use **OBS Studio** or **Xbox Game Bar (Win + G)** on Windows, or **Win + Alt + R**
to record. Write the script first and read it. Do one take per section and join them in Clipchamp
(built into Windows). Keep it under 3:00.

---

## 9. Submission checklist

- [ ] `bird_ensemble.ipynb` runs from top to bottom with **no errors** (Restart → Run All)
- [ ] Outputs (tables and charts) are **visible** in the saved notebook
- [ ] Interpretation cell filled in with **your numbers**
- [ ] **GenAI disclosure** at the top of the notebook (already included, so check the wording)
- [ ] Video ≤ 3:00 with voice-over
- [ ] (Optional) GitHub + Streamlit link
- [ ] Submitted **before 11:59 pm**

**GenAI disclosure (already in the notebook and app):**
> Parts of this notebook's code and the Streamlit dashboard were generated with AI assistants
> (Claude and GitHub Copilot). I reviewed, ran and adjusted all of the code, and the analysis and
> interpretation are my own.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | The kernel isn't `.venv`. Select the kernel again (2.4) or run `pip install -r requirements.txt` |
| Out of memory loading the data | In Step 2 use `pd.read_csv(DATA_PATH, usecols=[...the columns in section 3...])` |
| `KeyError` on a column name | Run `raw.columns` and rename to match, e.g. `raw = raw.rename(columns={...})` |
| District list looks wrong | Ask Copilot to map the names to the 25 districts (see Step 4) |
| Training too slow | In Step 8 reduce `n_estimators` (RF 400 → 200, Bagging 200 → 100) |
| Dashboard: `FileNotFoundError artifacts/...` | Run notebook Step 14 first, and start Streamlit **from the project folder** |
| Stacking very slow | Set `cv=3` in `StackingRegressor` |

---

## 11. How to use Copilot with this guide

Open Copilot Chat in VS Code and attach the files with `#file`:

- *"#file:GUIDE.md #file:bird_ensemble.ipynb Explain Step 5 in simple words before I run it."*
- *"#file:bird_ensemble.ipynb Step 4 shows these district names: ___. Fix the cleaning so there are 25 districts."*
- *"Here are my results table and feature importance: ___. Help me write the interpretation using the templates in section 5 of #file:GUIDE.md."*
- *"#file:app.py Update the Key takeaways text in the Insights tab using these results: ___."*

**Rule for Copilot:** keep the feature list, the split (`random_state=42`, 20% test) and the
`artifacts/` file names unchanged. The dashboard depends on them.
