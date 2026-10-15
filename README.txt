# JobGuard AI — Streamlit deployment

This starter website uses the SimpleRNN model from `fake_job_RNN(1).ipynb`.

## 1. Save your trained model and tokenizer

Open the notebook and run all cells successfully through model training. Then run the code in `save_artifacts_cell.py` as a new notebook cell. This creates:

- `fake_job_rnn.keras` — the trained Keras model
- `tokenizer.pkl` — the fitted text tokenizer

Keep these files in the same folder as `Fake_job.py`. The pickle file must come from a tokenizer you trust; never load pickle files from unknown sources.

**Important:** Keras models are best saved as `.keras`, rather than pickling the whole model. This project uses pickle for the tokenizer.

## 2. Create the project folder

Keep these files together:

```text
fake_job_streamlit_app/
├── Fake_job.py
├── requirements.txt
├── fake_job_rnn.keras
└── tokenizer.pkl
```

## 3. Install and run locally

Open Notepad only to edit `Fake_job.py` if you wish. Save it as **All files** with the exact name `Fake_job.py`, not `Fake_job.py.txt`.

Open Command Prompt in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run Fake_job.py
```

A browser window should open with the website.

## 4. Share it publicly with friends

The local `localhost` link works only on your computer. To give friends a link they can open:

1. Sign in to GitHub and create a **private** repository first if you want to review the files before making them public; Streamlit Community Cloud generally needs repository access and the deployment configuration may require a public repo depending on your setup/account.
2. Upload `app.py`, `requirements.txt`, `fake_job_rnn.keras`, and `tokenizer.pkl` to a GitHub repository that Streamlit Community Cloud can access.
3. Do not upload private data, credentials, or sensitive datasets. Pickle files can execute code when loaded, so only deploy your own trusted tokenizer.
4. Open https://share.streamlit.io/ or https://streamlit.io/cloud, connect GitHub, choose the repository and `app.py`, and deploy.
5. After deployment succeeds, copy the Streamlit app URL and send it to your friends.

The deployed app stays available online according to the hosting service's terms and limits. If you update the code or model files in the connected repository, redeploy/reboot as needed.

## Notes about the supplied notebook

- The notebook currently reads the dataset from a local Windows path. The deployed app does not need that CSV because it uses the saved model and tokenizer.
- The notebook creates `X` by combining company profile, salary, requirements, benefits, location, description and industry. The web app uses those fields too, and also accepts a job title.
- The notebook fits the tokenizer after filtering `df` to fraudulent rows. This may limit vocabulary quality. For a stronger model, fit the tokenizer on training text before filtering/oversampling and split the original dataset before oversampling.
- Make sure class labels match the dataset: `fraudulent = 1` means fraudulent and `0` means legitimate.
- The confidence shown by the website is not a guarantee. Do not use this model as the only basis for accusing a company or deciding whether a job is safe.
