import json
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


@st.cache_resource
def charger_modele():
    m = XGBClassifier()
    m.load_model("modele_xgb.json")
    with open("seuil.json") as f:
        seuil = json.load(f)["seuil"]
    return m, seuil


model, SEUIL = charger_modele()
COLS = model.get_booster().feature_names  # les 114 colonnes, dans le bon ordre

AGE_CHOICES = {f"[{i}-{i+10})": i + 5 for i in range(0, 100, 10)}
DIAG_CHOICES = ["Diabetes", "Circulatory", "Respiratory", "Digestive", "Injury",
                "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Unknown"]


def construire_ligne(age_label, gender, race, time_in_hospital, num_lab, num_proc,
                     num_meds, n_outp, n_emerg, n_inp, n_diag, a1c, max_glu,
                     diag1, diag2, diag3):
    jours = max(int(time_in_hospital), 1)
    row = {c: 0 for c in COLS}

    num = {
        "age": AGE_CHOICES[age_label],
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab,
        "num_procedures": num_proc,
        "num_medications": num_meds,
        "number_outpatient": n_outp,
        "number_emergency": n_emerg,
        "number_inpatient": n_inp,
        "number_diagnoses": n_diag,
        "prior_visits": n_outp + n_emerg + n_inp,
        "meds_per_day": num_meds / jours,
        "labs_per_day": num_lab / jours,
        "severity_score": n_diag + num_proc + time_in_hospital,
        "a1c_tested": int(a1c != "None"),
        "glu_tested": int(max_glu != "None"),
        "n_diag_diabetes": sum(d == "Diabetes" for d in [diag1, diag2, diag3]),
    }
    for k, v in num.items():
        if k in row:
            row[k] = v

    cats = {"gender": gender, "race": race, "A1Cresult": a1c, "max_glu_serum": max_glu,
            "diag_1_group": diag1, "diag_2_group": diag2, "diag_3_group": diag3}
    for c, v in cats.items():
        nom = f"{c}_{v}"
        if nom in row:
            row[nom] = 1

    return pd.DataFrame([row], columns=COLS)


st.set_page_config(page_title="Prédiction diabète", page_icon="🩺")
st.title("Prédiction : traitement antidiabétique")
st.caption("Outil pédagogique. Ne remplace pas un avis médical.")

with st.form("formulaire"):
    col1, col2 = st.columns(2)
    with col1:
        age_label = st.selectbox("Tranche d'âge", list(AGE_CHOICES), index=5)
        gender = st.selectbox("Genre", ["Female", "Male"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic",
                                     "Asian", "Other", "Unknown"])
        time_in_hospital = st.number_input("Jours d'hospitalisation", 1, 14, 3)
        num_lab = st.number_input("Nb d'examens de laboratoire", 0, 200, 40)
        num_proc = st.number_input("Nb de procédures", 0, 10, 1)
        num_meds = st.number_input("Nb de médicaments", 0, 100, 15)
    with col2:
        n_outp = st.number_input("Visites ambulatoires (an passé)", 0, 50, 0)
        n_emerg = st.number_input("Visites aux urgences (an passé)", 0, 50, 0)
        n_inp = st.number_input("Hospitalisations (an passé)", 0, 50, 0)
        n_diag = st.number_input("Nb de diagnostics", 1, 16, 8)
        a1c = st.selectbox("Résultat HbA1c", ["None", "Norm", ">7", ">8"])
        max_glu = st.selectbox("Glucose sérique max", ["None", "Norm", ">200", ">300"])
        diag1 = st.selectbox("Diagnostic principal", DIAG_CHOICES, index=0)
        diag2 = st.selectbox("Diagnostic 2", DIAG_CHOICES, index=1)
        diag3 = st.selectbox("Diagnostic 3", DIAG_CHOICES, index=8)

    envoyer = st.form_submit_button("Prédire")

if envoyer:
    X = construire_ligne(age_label, gender, race, time_in_hospital, num_lab, num_proc,
                         num_meds, n_outp, n_emerg, n_inp, n_diag, a1c, max_glu,
                         diag1, diag2, diag3)
    p = model.predict_proba(X)[0, 1]
    st.metric("Probabilité de traitement", f"{p:.1%}")
    if p >= SEUIL:
        st.success(f"Prédiction : Oui (seuil = {SEUIL})")
    else:
        st.info(f"Prédiction : Non (seuil = {SEUIL})")