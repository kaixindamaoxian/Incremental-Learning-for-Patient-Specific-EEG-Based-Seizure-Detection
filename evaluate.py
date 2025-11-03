from pathlib import Path
import re
import numpy as np
import pandas as pd
from epilepsy2bids.annotations import Annotations, SeizureType
from timescoring import annotations, scoring
from timescoring import visualization
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

FS = 256


def toMask(annotations):
    mask = np.zeros(int(annotations.events[0]["recordingDuration"] * FS))   # 2625 * 256 = 672000, nd(672000,)
    for event in annotations.events:
        if event["eventType"].value != "bckg":
            mask[
                round(event["onset"] * FS): round(event["onset"] + event["duration"])
                * FS
            ] = 1
    return mask


def computeScores(tp, fp, refTrue, duration):
    # Sensitivity
    if refTrue > 0:
        sensitivity = tp / refTrue
    else:
        sensitivity = np.nan  # no ref event

    # Precision
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan  # no hyp event

    # F1 Score
    if np.isnan(sensitivity) or np.isnan(precision):
        f1 = np.nan
    elif (sensitivity + precision) == 0:  # No overlap ref & hyp
        f1 = 0
    else:
        f1 = 2 * sensitivity * precision / (sensitivity + precision)

    # FP Rate
    fpRate = fp / (duration / 3600 / 24)  # FP per day

    return sensitivity, precision, f1, fpRate


def extract_patient_session(filename):
    match = re.search(r"sub-(\d+)_ses-(\d+)", filename) # re.search(r"sub-(\d+)_ses-(\d+)", filename)
    return int(match.group(1)), int(match.group(2)) # patient_num, session_num

def extract_patient(filename):
    match = re.search(r"sub-(\d+)", filename) # re.search(r"sub-(\d+)_ses-(\d+)", filename)
    return int(match.group(1)) # patient_num, session_num


def sz_evaluate(refFolder: str, hypFolder=None):
    DATASET = "BIDS_Siena"
    results = {
        "dataset": [],
        "subject": [],
        # "session": [],   # 新增session字段
        # "fold": [],      # 新增fold字段
        "file": [],
        "duration": [],
        "tp_sample": [],
        "fp_sample": [],
        "refTrue_sample": [],
        "tp_event": [],
        "fp_event": [],
        "refTrue_event": [],
    }

    patient_session_data = {}

    for refTsv in sorted(Path(refFolder).glob("*_label.tsv"), key=lambda f: extract_patient(f.name)):   # extract_patient_session(f.name)
        ref = Annotations.loadTsv(refTsv)
        ref = annotations.Annotation(toMask(ref), FS)

        hypTsv = Path(str(refTsv)[:-10] + "_output.tsv")
        if hypTsv.exists():
            hyp = Annotations.loadTsv(hypTsv)
            hyp = annotations.Annotation(toMask(hyp), FS)
        else:
            hyp = annotations.Annotation(np.zeros_like(ref.mask), ref.fs)

        # plotIndividualEvents
        # visualization.plotIndividualEvents(ref, hyp)

        subject = refTsv.name.split("_")[0]
        # session = refTsv.name.split("_")[1]
        # fold = refTsv.name.split("_")[2]

        sampleScore = scoring.SampleScoring(ref, hyp)
        eventScore = scoring.EventScoring(ref, hyp)

        # key = f"{subject}_{session}"
        key = f"{subject}"
        if key not in patient_session_data:
            patient_session_data[key] = {
                "tp_sample": 0,
                "fp_sample": 0,
                "refTrue_sample": 0,
                "tp_event": 0,
                "fp_event": 0,
                "refTrue_event": 0,
                "duration": 0
            }

        patient_session_data[key]["tp_sample"] += sampleScore.tp
        patient_session_data[key]["fp_sample"] += sampleScore.fp
        patient_session_data[key]["refTrue_sample"] += sampleScore.refTrue
        patient_session_data[key]["tp_event"] += eventScore.tp
        patient_session_data[key]["fp_event"] += eventScore.fp
        patient_session_data[key]["refTrue_event"] += eventScore.refTrue
        patient_session_data[key]["duration"] += len(ref.mask) / ref.fs

        results["dataset"].append(DATASET)
        results["subject"].append(subject)
        # results["session"].append(session)
        # results["fold"].append(fold)
        results["file"].append(refTsv.name)
        results["duration"].append(len(ref.mask) / ref.fs)
        results["tp_sample"].append(sampleScore.tp)
        results["fp_sample"].append(sampleScore.fp)
        results["refTrue_sample"].append(sampleScore.refTrue)
        results["tp_event"].append(eventScore.tp)
        results["fp_event"].append(eventScore.fp)
        results["refTrue_event"].append(eventScore.refTrue)

    # 计算并打印最终结果
    for key, values in patient_session_data.items():
        # subject, session = key.split("_")
        subject = key

        # Sample-level scoring
        sensitivity_sample, precision_sample, f1_sample, fpRate_sample = computeScores(
            values["tp_sample"],
            values["fp_sample"],
            values["refTrue_sample"],
            values["duration"],
        )
        print(
            f"# Sample scoring for Subject {subject}, \n"  # Session {session}
            f"- Sensitivity : {sensitivity_sample:.3f} \n"
            f"- Precision   : {precision_sample:.3f} \n"
            f"- F1-score    : {f1_sample:.3f} \n"
            f"- FP/24h      : {fpRate_sample:.3f} \n"
        )

        # Event-level scoring
        sensitivity_event, precision_event, f1_event, fpRate_event = computeScores(
            values["tp_event"],
            values["fp_event"],
            values["refTrue_event"],
            values["duration"],
        )
        print(
            f"# Event scoring for Subject {subject}, \n"
            f"- Sensitivity : {sensitivity_event:.3f} \n"
            f"- Precision   : {precision_event:.3f} \n"
            f"- F1-score    : {f1_event:.3f} \n"
            f"- FP/24h      : {fpRate_event:.3f} \n"
        )

    # 保存详细的结果到CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)


def ss_evaluate(all_preds, test_labels):
    report = classification_report(test_labels, all_preds, output_dict=True, zero_division=0)
    formatted_report = {}
    for key, value in report.items():
        if isinstance(value, dict):
            formatted_report[key] = {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in value.items()}
        elif isinstance(value, float):
            formatted_report[key] = f"{value:.3f}"
        else:
            formatted_report[key] = value
    kappa = cohen_kappa_score(test_labels, all_preds)
    cm = confusion_matrix(test_labels, all_preds)

    print(
        f"- Accuracy (Acc)                  : {report['accuracy']:.3f} \n"
        f"- Macro-averaged Precision (MP)   : {report['macro avg']['precision']:.3f} \n"
        f"- Macro-averaged Recall (MR)      : {report['macro avg']['recall']:.3f} \n"
        f"- Macro-averaged F1-score (MF1)   : {report['macro avg']['f1-score']:.3f} \n"    # 'macro avg' and 'weighted avg'
        f"- Cohen Kappa (K)                 : {kappa:.3f} \n"
        # f"- Confusion Matrix                : \n{cm}"
    )
    return {
        "Accuracy": report["accuracy"],
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Cohen Kappa": kappa
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluation code to compare annotations from a seizure detection algorithm to ground truth annotations."
    )
    # parser.add_argument("ref", help="Path to the root folder containing the reference annotations.")
    # parser.add_argument("hyp", help="Path to the root folder containing the hypothesis annotations.")
    parser.add_argument("--root_path", type=str, default="/home/z/dzw/BIDS_CHB-MIT_Preprocess/output/",
                        help="Root path of ref and hyp files.")

    args = parser.parse_args()
    sz_evaluate(args.root_path)
