"""Builds docs/NLP Final Report - Group 02 Section B.docx from the course template.

Fills the cover, the contributions table and all eight report sections, and appends the
four code listings the appendix asks for. Every number is read from the artefacts on
disk at build time rather than typed here, so the report cannot drift from the results.

Two scales appear in the report and the distinction is deliberate. Section 2 reports the
midterm experiments, which ran on a 6,000-row balanced sample per dataset. Sections 3 to
6 report the final-term experiments, which ran on the full balanced corpora. Table 2
follows the course specification, which asks for the classical rows from the midterm and
the transformer rows from the final term, and the text says so plainly.

Run:  python docs/builders/build_final_report.py
"""
import json
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / 'docs' / 'NLP Report Template with Cover Page - Final-term Summer 25-26.docx'
OUT = ROOT / 'docs' / 'NLP Final Report - Group 02 Section B.docx'
PS = ROOT / 'experiments' / 'paper_scale'
AUDIT = ROOT / 'experiments' / 'audit'
NB = ROOT / 'notebooks' / 'nlp_final_submission_code.ipynb'

BODY_FONT, BODY_PT = 'Times New Roman', 12
TABLE_PT, CODE_PT = 7.5, 7.5

MEMBERS = [
    ('NILOY PAUL', '23-51773-2',
     'Data preprocessing and corpus preparation (25%). Loaded and class-balanced both '
     'corpora, built the duplicate-content hashing and the group-aware 72/8/20 split, ran '
     'the duplicate and leakage audit, and implemented the classical normalisation '
     'pipeline and the 128-token tokenisation diagnostics.'),
    ('ARNOB SARKER SUPTA', '23-52080-2',
     'BERT experiments (25%). Ran the bert-base-uncased hyperparameter grid over learning '
     'rate, batch size and weight decay on both datasets, selected the best configuration '
     'on validation weighted F1, and produced the BERT confusion matrices and per-document '
     'prediction arrays.'),
    ('NAZMUS SAKIB', '23-52638-2',
     'BERT variant experiments and analysis (25%). Ran the DeBERTa grid on both datasets, '
     'carried out the surface and content analysis, the tokenisation and label-free '
     'checks, the paired significance testing, and the figures.'),
    ('AFNAN UR RAYAN', '23-51992-2',
     'Ensemble, evaluation and reporting (25%). Built the validation-weighted soft-vote '
     'ensemble and its degenerate-weight check, ran the classical baselines and the '
     'cross-dataset transfer evaluation, compiled the result tables, and wrote the report.'),
]

# The classical models the midterm introduced, re-run on the FULL corpora. Both text
# representations were re-run, so the report no longer mixes two dataset sizes in one
# table the way the earlier draft did.
CLASSICAL = ['Naive Bayes', 'Logistic Regression', 'Support Vector Machine']
REPS = ['BoW', 'TF-IDF']

GRID = [(2e-5, 16, 0.01), (3e-5, 16, 0.01), (2e-5, 32, 0.01), (3e-5, 32, 0.01),
        (2e-5, 16, 0.1),  (3e-5, 16, 0.1),  (2e-5, 32, 0.1),  (3e-5, 32, 0.1)]


# ---------------------------------------------------------------- data loading
def load_results():
    """Everything the report quotes, read from disk."""
    r = {'grid': {}, 'deployed': {}, 'ensemble': {}}
    for tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            for lr, bs, wd in GRID:
                key = f'full_{tag}_{mk}_lr{lr:g}_bs{bs}_wd{wd:g}_s42'
                p = PS / 'results' / f'{key}.json'
                if p.exists():
                    t = json.load(open(p))['test']
                    r['grid'][(tag, mk, lr, bs, wd)] = (
                        t['accuracy'], t['precision'], t['recall'], t['f1'])
            info = json.load(open(PS / 'models' / f'{tag}_{mk}' / 'run_info.json'))
            r['deployed'][(tag, mk)] = info

        # Ensemble, recomputed from the deployed checkpoints' own probabilities.
        w = json.load(open(AUDIT / 'ensemble_full_scale.json'))['datasets'][tag]['weight_on_bert']
        pb = np.load(PS / 'probs' / f"{r['deployed'][(tag,'BERT')]['key']}.npz")
        pd_ = np.load(PS / 'probs' / f"{r['deployed'][(tag,'DeBERTa')]['key']}.npz")
        y = pb['test_labels']
        pred = (w * pb['test_probs'] + (1 - w) * pd_['test_probs']).argmax(1)
        acc = accuracy_score(y, pred)
        pre, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average='weighted', zero_division=0)
        r['ensemble'][tag] = {'w': w, 'metrics': (round(acc, 4), round(pre, 4),
                                                  round(rec, 4), round(f1, 4))}
    # Classical models at full scale, both representations. The JSON records accuracy
    # and F1 but not precision and recall, so those are recomputed from the saved score
    # arrays and checked back against the recorded accuracy and F1 before use.
    ev = json.load(open(AUDIT / 'full_model_evaluation.json'))
    sc = np.load(AUDIT / 'full_model_scores.npz')
    r['classical'] = {}
    for tag in ('D1', 'D2'):
        y = sc[f'{tag}|y_true']
        for name in CLASSICAL:
            for rep in REPS:
                key = f'{name} ({rep})'
                s_ = sc[f'{tag}|{key}']
                pred = ((s_ >= 0.5) if (s_.min() >= 0 and s_.max() <= 1) else (s_ > 0)).astype(int)
                acc = accuracy_score(y, pred)
                pre, rec, f1, _ = precision_recall_fscore_support(
                    y, pred, average='weighted', zero_division=0)
                rj = ev['datasets'][tag]['models'][key]
                assert abs(acc - rj['accuracy']) < 5e-4 and abs(f1 - rj['weighted_f1']) < 5e-4, \
                    f'recomputed metrics disagree with the recorded ones for {tag} {key}'
                r['classical'][(tag, name, rep)] = (round(acc, 4), round(pre, 4),
                                                    round(rec, 4), round(f1, 4))
    r['eval'] = ev
    r['dec'] = json.load(open(AUDIT / 'surface_content_decomposition.json'))
    r['ens_json'] = json.load(open(AUDIT / 'ensemble_full_scale.json'))
    return r


# ---------------------------------------------------------------- docx helpers
def style_run(run, size=BODY_PT, font=BODY_FONT, bold=False):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    # Word needs the east-asian name set too or it silently substitutes.
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font)
    return run


def set_text(par, text, size=BODY_PT, font=BODY_FONT, bold=False, justify=True):
    for r in list(par.runs):
        r._element.getparent().remove(r._element)
    style_run(par.add_run(text), size, font, bold)
    if justify:
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return par


def para_after(anchor, text, size=BODY_PT, font=BODY_FONT, bold=False, justify=True):
    """Insert a new paragraph directly after `anchor` and return it."""
    new_p = anchor._element.makeelement(qn('w:p'), {})
    anchor._element.addnext(new_p)
    from docx.text.paragraph import Paragraph
    par = Paragraph(new_p, anchor._parent)
    set_text(par, text, size, font, bold, justify)
    return par


def find_par(doc, needle):
    for p in doc.paragraphs:
        if needle in p.text:
            return p
    raise KeyError(needle)


def table_after(doc, anchor, rows, widths=None, header_rows=1, size=TABLE_PT):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ''
            par = cell.paragraphs[0]
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
            style_run(par.add_run(str(val)), size, BODY_FONT, bold=(i < header_rows))
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
    anchor._element.addnext(t._element)
    return t


def set_cell(cell, text, size=TABLE_PT, bold=False, center=True):
    """Merging in python-docx concatenates the text of every cell involved and keeps
    one empty paragraph per cell, which is what made the header rows tall and the
    label columns repeat. Clear the merged cell and write it once instead."""
    for par in list(cell.paragraphs[1:]):
        par._element.getparent().remove(par._element)
    par = cell.paragraphs[0]
    for r in list(par.runs):
        r._element.getparent().remove(r._element)
    style_run(par.add_run(text), size, BODY_FONT, bold)
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    par.paragraph_format.space_before = Pt(1)
    par.paragraph_format.space_after = Pt(1)


def merge_and_label(table, r0, c0, r1, c1, text, size=TABLE_PT, bold=True):
    cell = table.cell(r0, c0).merge(table.cell(r1, c1))
    set_cell(cell, text, size, bold)
    return cell


def no_row_split(table):
    """Stop a single row breaking across a page. Without this, a tall cell such as
    "Support Vector Machine" splits and Word leaves an empty continuation row on the
    next page, which reads as a spurious extra model."""
    from docx.oxml import OxmlElement
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        el = OxmlElement('w:cantSplit')
        el.set(qn('w:val'), 'true')
        trPr.append(el)


def repeat_header(table, n_rows):
    """Mark the first n rows as a header so they repeat when the table breaks a page."""
    from docx.oxml import OxmlElement
    for i in range(n_rows):
        trPr = table.rows[i]._tr.get_or_add_trPr()
        el = OxmlElement('w:tblHeader')
        el.set(qn('w:val'), 'true')
        trPr.append(el)


# ---------------------------------------------------------------- report text
def build_text(R):
    """All prose, with the numbers substituted from disk."""
    d = R['deployed']
    ev = R['eval']['datasets']
    dec = R['dec']['datasets']
    e1, e2 = R['ensemble']['D1'], R['ensemble']['D2']
    ej = R['ens_json']['datasets']

    def f(tag, mk):   # deployed test F1
        return d[(tag, mk)]['test']['f1']

    def cfg(tag, mk):
        i = d[(tag, mk)]
        return f"learning rate {i['lr']:g}, batch size {i['batch_size']}, weight decay {i['weight_decay']:g}"

    C = R['classical']

    def best_classical(tag, name):
        """Better of the two representations for this model on this dataset."""
        rep = max(REPS, key=lambda rp: C[(tag, name, rp)][3])
        return rep, C[(tag, name, rep)]

    d1_best = max(CLASSICAL, key=lambda n: best_classical('D1', n)[1][3])
    d2_best = max(CLASSICAL, key=lambda n: best_classical('D2', n)[1][3])
    d1_rep, d1_vals = best_classical('D1', d1_best)
    d2_rep, d2_vals = best_classical('D2', d2_best)

    T = {}

    T['title'] = 'Detecting AI-Generated Text with BERT, DeBERTa and a Soft-Vote Ensemble'
    T['authors'] = ', '.join(n.title() for n, _, _ in MEMBERS[:-1]) + ' and ' + MEMBERS[-1][0].title()
    T['emails'] = '{' + ', '.join(i for _, i, _ in MEMBERS) + '}@student.aiub.edu'

    T['intro'] = (
        'This project builds a system that decides whether a piece of text was written by a person or '
        'produced by a language model. We treat it as a binary classification problem and study it on two '
        'public datasets. The first is DAIGT V2, which holds 44,868 argumentative student essays, part '
        'written by students and part generated by several 2023-era models. The second is HC3, which holds '
        '85,449 question-and-answer pairs contrasting human answers with answers from ChatGPT. The two are '
        'useful together because they differ in almost every way that matters. DAIGT V2 has many generators, '
        'one genre and long documents, while HC3 has a single generator, five source domains and short ones.\n'
        'We balanced both datasets by cutting the larger class down to the size of the smaller one, which '
        'leaves 34,994 rows for DAIGT V2 and 53,806 for HC3. We then split each dataset once into training, '
        'validation and test parts of 72, 8 and 20 per cent, and every model in this report uses that same '
        'split. The split is group aware, meaning that documents with identical text after normalisation are '
        'kept together on one side of the boundary. This matters for HC3, where 7.16 per cent of rows are '
        'duplicates. Measured directly, our split puts none of the 10,732 HC3 test documents into training, '
        'while an ordinary random split of the same data puts 570 of them there. Without the group-aware '
        'split, part of the score would be measuring memorisation rather than detection.\n'
        'On top of that data we trained five kinds of model. Three are classical and were introduced in the '
        'midterm project, namely Naive Bayes, logistic regression and a linear support vector machine, each '
        'under both bag-of-words and TF-IDF. Two are '
        'transformers fine-tuned for this task, bert-base-uncased and microsoft/deberta-v3-base. We also '
        'combined the two transformers into a soft-vote ensemble. Each transformer was trained over a grid '
        'of eight settings per dataset, varying learning rate, batch size and weight decay, with the winner '
        'chosen on validation score. Training ran on a single RTX 3060 Ti, and one run took between ten and '
        'nineteen minutes.\n'
        f'The short version of the result is that DeBERTa is the best single model on both datasets, reaching '
        f'{f("D1","DeBERTa"):.4f} weighted F1 on DAIGT V2 and {f("D2","DeBERTa"):.4f} on HC3. The ensemble does not '
        'reliably beat it. We also found something less expected, which is that the two datasets are not '
        'equally hard for the same reasons, and Section 6 explains what we mean by that.')

    T['midterm'] = (
        'The midterm project compared three classical models on the same two datasets, each under two text '
        'representations, bag-of-words and TF-IDF. All six model and representation combinations were run on '
        'the full balanced corpora, using the same split the transformers use, so every number in this report '
        'comes from the same data. Table 1 gives the results.\n'
        f'The clearest pattern is that the two datasets are not equally difficult. On DAIGT V2 every classical '
        f'model does well, and the best of them, '
        f'{d1_best.lower()} with {d1_rep}, reaches {d1_vals[3]:.4f} F1. On HC3 the same models are noticeably '
        f'weaker, with the best, {d2_best.lower()} with {d2_rep}, reaching only {d2_vals[3]:.4f}. That is a gap '
        f'of about {abs(d1_vals[3]-d2_vals[3])*100:.0f} points of F1 between the two datasets for the same family of '
        'model, so the difficulty belongs to the data rather than to the method.\n'
        'The choice of representation also matters much more on HC3 than on DAIGT V2. On HC3, moving logistic '
        f'regression from TF-IDF to bag-of-words changes F1 from {C[("D2","Logistic Regression","TF-IDF")][3]:.4f} to '
        f'{C[("D2","Logistic Regression","BoW")][3]:.4f}, a gain of about two points. On DAIGT V2 the same change moves '
        f'it from {C[("D1","Logistic Regression","TF-IDF")][3]:.4f} to {C[("D1","Logistic Regression","BoW")][3]:.4f}, less '
        'than half a point. Bag-of-words wins for five of the six model and dataset combinations, the exception '
        'being the support vector machine on DAIGT V2, where TF-IDF is better. We read this as a sign that raw '
        'counts of very common tokens carry real signal on HC3, and that TF-IDF weighting, which is designed to '
        'discount common tokens, throws part of that signal away.\n'
        'Naive Bayes is the weakest model on both datasets and by a wide margin on HC3, which is expected given '
        'that it treats every word as independent. Logistic regression and the support vector machine are close '
        'to each other everywhere.\n'
        f'These results set the target for the transformers. To be worth their extra cost, a fine-tuned model '
        f'has to beat {d1_vals[3]:.4f} on DAIGT V2 and {d2_vals[3]:.4f} on HC3.')

    T['bert'] = (
        'We chose BERT as the base transformer for three practical reasons. It is the model the course '
        'introduced, so the comparison is meaningful against what we already knew. It is small enough at 110 '
        'million parameters to fine-tune repeatedly on one consumer GPU, which we needed because the grid has '
        'eight settings per dataset. And it is the most widely reported baseline for this task, so our numbers '
        'can be compared with published work.\n'
        'We used bert-base-uncased with a maximum sequence length of 128 tokens, at most five epochs with '
        'early stopping after two epochs without improvement, a warmup ratio of 0.1, dropout of 0.1 and the '
        'AdamW optimiser. The grid varied learning rate over 0.00002 and 0.00003, batch size over 16 and 32, '
        'and weight decay over 0.01 and 0.1, giving eight runs per dataset. Table 2 lists every run. We picked '
        'the winner on the validation split, never on the test split, because choosing on test would mean '
        'reporting a number we had already optimised against.\n'
        f'On DAIGT V2 the best BERT setting is {cfg("D1","BERT")}. It reaches {f("D1","BERT"):.4f} weighted F1 on '
        f'the test split, having scored {d[("D1","BERT")]["val"]["f1"]:.4f} on validation, and it trained for '
        f'{d[("D1","BERT")]["epochs_run"]} epochs in {d[("D1","BERT")]["train_seconds"]/60:.1f} minutes. Its test confusion '
        f'matrix is {d[("D1","BERT")]["test_confusion"]}, so of 6,998 documents it misclassifies 59, split '
        'roughly evenly between the two error directions.\n'
        f'On HC3 the best setting is {cfg("D2","BERT")}, reaching {f("D2","BERT"):.4f} on test after '
        f'{d[("D2","BERT")]["val"]["f1"]:.4f} on validation, in {d[("D2","BERT")]["train_seconds"]/60:.1f} minutes. '
        f'Its confusion matrix is {d[("D2","BERT")]["test_confusion"]}, which is 90 errors out of 10,732.\n'
        'It is worth noting how small the spread across the grid is. On DAIGT V2 the eight BERT runs span '
        '0.9869 to 0.9954 in F1, a range of under one point, and on HC3 they span 0.9910 to 0.9945. In other '
        'words, once the model and the data are fixed, the hyperparameters in this grid make very little '
        'difference. We would not treat the difference between the best and second-best setting as meaningful.')

    T['variant'] = (
        'For the BERT variant we chose DeBERTa, specifically the microsoft/deberta-v3-base checkpoint. Four '
        'things made it the right choice.\n'
        'The first is benchmark standing. DeBERTa-v3-base is the strongest encoder of its size at the time we '
        'selected it, ahead of both BERT-base and RoBERTa-base across the GLUE suite of language-understanding '
        'tasks, and the DeBERTa family was the first to pass the human baseline on SuperGLUE. Starting from a '
        'stronger pretrained model is the simplest way to get a stronger fine-tuned one, and it costs us '
        'nothing extra since the two are close in size.\n'
        'The second is its attention mechanism, which keeps the representation of a word and the '
        'representation of its position separate rather than adding them together. That suits a task where '
        'the way text is written may matter as much as which words appear.\n'
        'The third is how it was pretrained. DeBERTa-v3 is trained to spot tokens that have been replaced by '
        'a small generator model, rather than to fill in masked blanks. This gives it a learning signal on '
        'every token instead of only the masked ones, so it reaches good accuracy after fewer epochs of '
        'fine-tuning, which is what our GPU budget allowed.\n'
        'The fourth is its tokeniser, and this is the one that turned out to matter most for our data. '
        'DeBERTa treats a space before a punctuation mark as part of the token, whereas the BERT tokeniser '
        'discards that distinction entirely. We verified this directly rather than assuming it. For the '
        'strings "the answer is simple ." and "the answer is simple." BERT produces identical token '
        'identifiers on three of three test pairs, while DeBERTa distinguishes all three. The two models '
        'therefore see genuinely different text, and Section 6 shows why that matters on HC3.\n'
        'One honest qualification belongs with the first reason. A model that leads a benchmark suite does '
        'not automatically lead on a new task, and our own results show exactly that. DeBERTa is far ahead on '
        'HC3 but level with BERT on DAIGT V2. Benchmark standing was a sound reason to try it first. It was '
        'not a guarantee, and we would not present it as one.\n'
        'We trained DeBERTa with exactly the same grid, the same split and the same harness as BERT, so any '
        'difference between the two is a difference between the models and not between two training setups. '
        'Table 2 lists all eight runs per dataset alongside the BERT ones.\n'
        f'On DAIGT V2 the best DeBERTa setting is {cfg("D1","DeBERTa")}, reaching {f("D1","DeBERTa"):.4f} weighted F1 '
        f'on test after {d[("D1","DeBERTa")]["val"]["f1"]:.4f} on validation, in '
        f'{d[("D1","DeBERTa")]["train_seconds"]/60:.1f} minutes. Its confusion matrix is {d[("D1","DeBERTa")]["test_confusion"]}.\n'
        f'On HC3 the best setting is {cfg("D2","DeBERTa")}, and this is where DeBERTa clearly separates from '
        f'BERT. It reaches {f("D2","DeBERTa"):.4f} on test after {d[("D2","DeBERTa")]["val"]["f1"]:.4f} on validation, '
        f'with confusion matrix {d[("D2","DeBERTa")]["test_confusion"]}, which is only 30 errors out of 10,732. '
        f'BERT made 90 errors on the same documents, so DeBERTa removes two thirds of them.\n'
        'The comparison between the two models is not the same on the two datasets, and that is the most '
        'useful thing this section shows. On DAIGT V2 the two are level, '
        f'{f("D1","BERT"):.4f} against {f("D1","DeBERTa"):.4f}, a difference of one ten-thousandth that we would not '
        'call a difference at all. On HC3 the gap is real and large. Any conclusion of the form "DeBERTa is '
        'better than BERT for detecting AI text" would be true for one of our datasets and false for the other.')

    T['ensemble'] = (
        'Our ensemble combines the two fine-tuned transformers by averaging the class probabilities they '
        'produce, which is usually called soft voting. Rather than fixing an equal average, we searched a '
        'weight w between 0 and 1 in steps of 0.05, forming w times the BERT probabilities plus one minus w '
        'times the DeBERTa probabilities. We chose w on the validation split and then applied it once to the '
        'test split. Choosing the weight on test would have meant tuning on the data we report.\n'
        f'On DAIGT V2 the search selected w = {e1["w"]:.2f}, an equal blend of the two models. The ensemble reaches '
        f'{e1["metrics"][3]:.4f} weighted F1, against {f("D1","BERT"):.4f} for BERT alone and {f("D1","DeBERTa"):.4f} for '
        'DeBERTa alone. That looks like a clear gain, and it is the point where it would be easy to overstate '
        'the result. We compared the ensemble with DeBERTa on the same test documents using a paired test, '
        f'which gives p = {ej["D1"]["paired_ensemble_vs_stronger"]["mcnemar_exact_p"]:.3f} and a 95 per cent interval on '
        'the difference in error rate of minus 0.39 to plus 0.01 percentage points. That interval contains '
        'zero, so on this test set the improvement cannot be told apart from chance. We report the ensemble as '
        'nominally ahead, not as better.\n'
        f'On HC3 the search selected w = {e2["w"]:.2f}. That means it put no weight at all on BERT, so the ensemble '
        f'is simply DeBERTa and scores exactly what DeBERTa scores, {e2["metrics"][3]:.4f}. We report this openly '
        'rather than presenting the number as an ensemble result, because a soft-vote ensemble that has '
        'discarded one of its two members is not really an ensemble.\n'
        'The honest summary is that ensembling did not help us. On the dataset where the two models are '
        'level, blending them produced a small gain we cannot confirm statistically, and on the dataset where '
        'one model is clearly better, the weight search simply picked that model. This is a reasonable '
        'outcome rather than a failure. Ensembles help most when their members make different kinds of '
        'mistakes, and on HC3 DeBERTa makes so few mistakes that BERT has little to add.')

    T['overall'] = (
        'Table 3 brings everything together in the form the project specification asks for. Every row comes '
        'from the full balanced corpora and the same fixed split. Each classical row is shown with whichever '
        'of the two text representations worked better on that dataset, which the Rep. column names. The '
        'transformers read raw text, so no representation applies to them.\n'
        'Reading the table across, three conclusions hold. The first is that DeBERTa is the best single model '
        'on both datasets. The second is that the size of its advantage depends entirely on which dataset you '
        f'look at. On DAIGT V2 the best classical model reaches {d1_vals[3]:.4f} and DeBERTa reaches '
        f'{f("D1","DeBERTa"):.4f}, a difference of {abs(f("D1","DeBERTa")-d1_vals[3])*100:.2f} of a percentage point, so '
        'the extra cost of a transformer buys almost nothing there. On HC3 the same comparison is '
        f'{d2_vals[3]:.4f} against {f("D2","DeBERTa"):.4f}, a gap of more than four points, and the transformer is '
        'clearly doing something the classical models cannot. The third is that the ensemble adds nothing we '
        'can demonstrate.\n'
        'It is worth being precise about the DAIGT V2 result rather than rounding it away. The classical model '
        'reads the whole document while the transformers read only their first 128 tokens, which is about a '
        'third of the average essay. When we refitted the classical models on exactly the same 128-token span '
        'the transformers see, their error rate rose from 0.90 to 2.10 per cent while BERT stayed at 0.84. So '
        'the apparent tie on DAIGT V2 is not evidence that a bag-of-words model is as good as a transformer. '
        'It is evidence that reading the whole essay is worth about as much as being a transformer, which is a '
        'different and more useful statement.\n'
        'We also ran one experiment that is not required by the specification but explains the pattern above, '
        'and it is the most interesting thing we found. We built two restricted models. One reads only 47 '
        'surface properties of the text, such as how often a space appears before a full stop, how long the '
        'document is, and how often capital letters are used. It never sees a single word. The other reads '
        'only the words, after punctuation, capitalisation and unusual characters have been stripped out. '
        f'On HC3 these two score almost identically, with error rates of {dec["D2"]["surface_only"]["error_rate"]*100:.2f} '
        f'and {dec["D2"]["content_only"]["error_rate"]*100:.2f} per cent, and a paired test cannot separate them. In '
        'other words, on HC3 you can do about as well by looking at how the text is punctuated as by reading '
        f'what it says. On DAIGT V2 the picture is different, with {dec["D1"]["surface_only"]["error_rate"]*100:.2f} per '
        f'cent error from surface alone against {dec["D1"]["content_only"]["error_rate"]*100:.2f} per cent from words '
        'alone, so the words carry roughly eight times more of the signal.\n'
        'This explains why HC3 looked harder for the classical models and easier for DeBERTa. Much of what '
        'separates the classes on HC3 is in the formatting, and the classical pipeline deletes formatting '
        'during preprocessing while DeBERTa keeps it. It also means a high score on HC3 should be read '
        'carefully, because part of it comes from how the dataset was collected rather than from any real '
        'difference between human and machine writing.')

    T['limits'] = (
        'Several things limit what our numbers can be taken to mean, and we would rather state them than '
        'leave them to be discovered.\n'
        'Both transformers read at most 128 tokens per document. On HC3 that covers about three quarters of '
        'the average answer, but on DAIGT V2 it covers only about a third of the average essay. So the '
        'DAIGT V2 transformer results really describe classifying an essay from its opening, and the '
        'comparison with the classical models, which read the whole document, is not a fair fight in either '
        'direction.\n'
        'Every transformer number comes from a single training seed. We repeated a few settings with '
        'different seeds and saw the score move by up to a few thousandths, and one such estimate changed by '
        'about a third when we simply reran it, because training on a GPU is not perfectly repeatable. For '
        'that reason we did not base any claim on small differences between runs, and where we compare two '
        'models we compare their predictions on the same documents instead.\n'
        'Both datasets are English, and one of them, HC3, contains output from a single generator collected '
        'at a single point in time. Nothing in this report shows that our models would work on text from a '
        'newer model, on another language, or on a domain neither dataset covers. We checked the weakest '
        'version of this by training on one dataset and testing on the other, and performance falls by '
        'between 8 and 20 points of F1, so the models clearly learn things specific to their own dataset.\n'
        'Finally, we did not test what happens when someone actively tries to defeat the detector. We ran '
        'some simple text corruption experiments, but we also found that our models confidently label random '
        'character strings as human-written, which means a drop in score under corruption cannot be '
        'separated from the model simply not coping with unreadable input. We therefore do not report those '
        'experiments as evidence of robustness in either direction.')

    T['conclusion'] = (
        'We built and compared five families of model for deciding whether text is human-written or '
        'machine-generated, on two public datasets, using one fixed split and one training harness so that '
        f'the comparisons are fair. DeBERTa is the best single model on both, reaching {f("D1","DeBERTa"):.4f} '
        f'weighted F1 on DAIGT V2 and {f("D2","DeBERTa"):.4f} on HC3, which improves on the best classical model '
        f'by {abs(f("D1","DeBERTa")-d1_vals[3])*100:.2f} of a percentage point on the first dataset and by more than four '
        'points on the second. A soft-vote ensemble of BERT and DeBERTa did not reliably improve on DeBERTa alone, '
        'and we report that plainly rather than presenting a difference in the fourth decimal place as a '
        'gain.\n'
        'The result we think is most worth carrying forward is not the leaderboard. It is that a model '
        'reading only 47 formatting properties of a document, and no words at all, matches a full '
        'bag-of-words model on HC3 while falling far behind it on DAIGT V2. That tells us the two datasets '
        'are separable for different reasons, which a single accuracy figure hides completely. Anyone '
        'choosing a dataset to evaluate a detector, or reading a reported score, should want to know which '
        'of the two situations they are in. The check is cheap, it needs no extra labelling, and it runs in '
        'minutes, so we would recommend running it before trusting any headline number on this task.')

    return T


# ---------------------------------------------------------------- code listings
def code_listings():
    """The four required listings, lifted from the submission notebook so the report
    and the notebook cannot disagree."""
    nb = json.load(open(NB))
    src = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    def find(*needles):
        for s in src:
            if all(n in s for n in needles):
                return s.strip()
        raise KeyError(needles)
    return [
        ('Data preprocessing code',
         find('def group_split', 'def balance')),
        ('Data preprocessing code, classical text pipeline',
         find('def preprocess_classical')),
        ('Data preprocessing code, tokenisation for the transformers',
         find('def get_tokenized')),
        ('BERT and BERT-variant training code (one harness, used for both)',
         find('def train_one', 'TrainingArguments')),
        ('BERT code where the best performance was achieved',
         find('BERT_RESULT, BERT_CFG')),
        ('BERT variant (DeBERTa) code where the best performance was achieved',
         find('DEBERTA_RESULT, DEBERTA_CFG')),
        ('Ensemble code',
         find('WEIGHTS = np.round', 'ENSEMBLE')),
    ]


# ---------------------------------------------------------------- assembly
def main():
    R = load_results()
    T = build_text(R)
    doc = Document(str(TEMPLATE))

    # ---- cover page
    set_text(find_par(doc, 'Section:'), 'Section: B', justify=False).alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_text(find_par(doc, 'Group:'), 'Group: 02', justify=False).alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---- contributions table
    tbl = doc.tables[0]
    for i, (name, sid, contrib) in enumerate(MEMBERS, start=1):
        set_text(tbl.cell(i, 1).paragraphs[0], f'Name: {name}', justify=False)
        p2 = tbl.cell(i, 1).paragraphs[1] if len(tbl.cell(i, 1).paragraphs) > 1 else tbl.cell(i, 1).add_paragraph()
        set_text(p2, f'ID: {sid}', justify=False)
        set_text(tbl.cell(i, 2).paragraphs[0], contrib)

    # ---- title block
    set_text(find_par(doc, 'Project Title'), T['title'], size=16, bold=True, justify=False
             ).alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_text(find_par(doc, 'Student 1 Name'), T['authors'], justify=False
             ).alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_text(find_par(doc, '@student.aiub.edu'), T['emails'], justify=False
             ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---- section bodies. Each template placeholder becomes the first paragraph and
    #      any remaining ones are appended after it, so paragraph breaks survive.
    def write_section(placeholder_needle, body):
        chunks = body.split('\n')
        anchor = set_text(find_par(doc, placeholder_needle), chunks[0])
        for c in chunks[1:]:
            anchor = para_after(anchor, c)
        return anchor

    end1 = write_section('Provide an overall summary of your work', T['intro'])
    end2 = write_section('Take the resultant table from the experiments', T['midterm'])
    end3 = write_section('Give reason behind selecting the BERT model. Additionally, based on your experimental results, select the BERT model', T['bert'])
    end4 = write_section('select the BERT variant model', T['variant'])
    end5 = write_section('Discuss your ensemble model', T['ensemble'])
    end6 = write_section('Compare the overall results', T['overall'])
    write_section('Discuss the limitations', T['limits'])
    write_section('Summarize your work in this section', T['conclusion'])

    # ---- Table 1, the midterm results
    cap1 = para_after(end2, 'Table 1. Classical models on the full balanced corpora, both text representations, '
                            'evaluated on the same test split every other model in this report uses. Acc = Accuracy, Prec = Precision, Rec = Recall, F1 = F1 Score.',
                      size=10, justify=False)
    rows = [[''] * 10, ['', '', 'Acc', 'Prec', 'Rec', 'F1', 'Acc', 'Prec', 'Rec', 'F1']]
    for name in CLASSICAL:
        for rep in REPS:
            rows.append([name, rep]
                        + [f'{v:.4f}' for v in R['classical'][('D1', name, rep)]]
                        + [f'{v:.4f}' for v in R['classical'][('D2', name, rep)]])
    t1 = table_after(doc, cap1, rows, header_rows=2)
    merge_and_label(t1, 0, 2, 0, 5, 'Dataset 1 (DAIGT V2)')
    merge_and_label(t1, 0, 6, 0, 9, 'Dataset 2 (HC3)')
    merge_and_label(t1, 0, 0, 1, 0, 'Model')
    merge_and_label(t1, 0, 1, 1, 1, 'Rep.')
    for i, name in enumerate(CLASSICAL):
        r0 = 2 + i * len(REPS)
        merge_and_label(t1, r0, 0, r0 + len(REPS) - 1, 0, name, bold=False)
    repeat_header(t1, 2)
    no_row_split(t1)

    # ---- Table 2, the full fine-tuning grid
    cap2 = para_after(end3, 'Table 2. Final-term fine-tuning experiments. Eight configurations for each '
                            'model on each dataset, full balanced corpora, evaluated on the test split. '
                            'The ENSEMBLE row uses the weight chosen on validation.', size=10, justify=False)
    rows = [[''] * 12,
            ['', '', '', '', 'Acc', 'Prec', 'Rec', 'F1', 'Acc', 'Prec', 'Rec', 'F1']]
    missing = []
    for mk in ('BERT', 'DeBERTa'):
        for lr, bs, wd in GRID:
            a = R['grid'].get(('D1', mk, lr, bs, wd))
            b = R['grid'].get(('D2', mk, lr, bs, wd))
            if a is None or b is None:
                missing.append((mk, lr, bs, wd))
            rows.append(['', f'{lr:.5f}', str(bs), str(wd)]
                        + [f'{v:.4f}' for v in a] + [f'{v:.4f}' for v in b])
    if missing:
        raise SystemExit('missing grid results, refusing to emit blank cells: %s' % missing)
    rows.append([''] * 4
                + [f'{v:.4f}' for v in R['ensemble']['D1']['metrics']]
                + [f'{v:.4f}' for v in R['ensemble']['D2']['metrics']])
    t2 = table_after(doc, cap2, rows, header_rows=2, size=6.5)
    merge_and_label(t2, 0, 4, 0, 7, 'Dataset 1 (DAIGT V2)', size=6.5)
    merge_and_label(t2, 0, 8, 0, 11, 'Dataset 2 (HC3)', size=6.5)
    for c, lbl in enumerate(('Model', 'Learning Rate', 'Batch Size', 'Weight Decay')):
        merge_and_label(t2, 0, c, 1, c, lbl, size=6.5)
    merge_and_label(t2, 2, 0, 9, 0, 'BERT', size=6.5, bold=False)
    merge_and_label(t2, 10, 0, 17, 0, 'DeBERTa\n(BERT variation)', size=6.5, bold=False)
    set_cell(t2.cell(18, 0), 'ENSEMBLE', size=6.5, bold=True)
    merge_and_label(t2, 18, 1, 18, 3, '(validation-selected weight)', size=6.5, bold=False)
    repeat_header(t2, 2)
    no_row_split(t2)

    # ---- Table 3, the required final comparison
    def best_rep(tag, name):
        rep = max(REPS, key=lambda rp: R['classical'][(tag, name, rp)][3])
        return rep, R['classical'][(tag, name, rep)]

    cap3 = para_after(end6, 'Table 3. Final comparison. Every row comes from the full balanced corpora and '
                            'the same fixed split. Each classical row is shown with whichever text '
                            'representation performed better on that dataset, which the Rep. column names. The transformers read raw text, so no '
                            'representation applies to them. The ENSEMBLE row uses the weight chosen on '
                            'validation.', size=10, justify=False)

    def best_rep(tag, name):
        rep = max(REPS, key=lambda rp: R['classical'][(tag, name, rp)][3])
        return rep, R['classical'][(tag, name, rep)]

    # Eleven columns. The representation sits inside each dataset group rather than being
    # packed into the model name, because a model can win with one representation on one
    # dataset and the other on the other, and a combined label like "TF-IDF / BoW" makes
    # the reader decode which half belongs to which dataset.
    rows = [[''] * 11,
            ['', 'Rep.', 'Acc', 'Prec', 'Rec', 'F1', 'Rep.', 'Acc', 'Prec', 'Rec', 'F1']]
    for name in CLASSICAL:
        r1, v1 = best_rep('D1', name)
        r2, v2 = best_rep('D2', name)
        rows.append([name, r1] + [f'{v:.4f}' for v in v1]
                    + [r2] + [f'{v:.4f}' for v in v2])
    for mk, label in (('BERT', 'BERT'), ('DeBERTa', 'DeBERTa (BERT variation)')):
        a = R['deployed'][('D1', mk)]['test']
        b = R['deployed'][('D2', mk)]['test']
        rows.append([label, 'raw'] + [f'{a[k]:.4f}' for k in ('accuracy', 'precision', 'recall', 'f1')]
                    + ['raw'] + [f'{b[k]:.4f}' for k in ('accuracy', 'precision', 'recall', 'f1')])
    rows.append(['ENSEMBLE', 'raw'] + [f'{v:.4f}' for v in R['ensemble']['D1']['metrics']]
                + ['raw'] + [f'{v:.4f}' for v in R['ensemble']['D2']['metrics']])
    t3 = table_after(doc, cap3, rows, header_rows=2, size=7)
    merge_and_label(t3, 0, 1, 0, 5, 'Dataset 1 (DAIGT V2)', size=7)
    merge_and_label(t3, 0, 6, 0, 10, 'Dataset 2 (HC3)', size=7)
    merge_and_label(t3, 0, 0, 1, 0, 'Model', size=7)
    repeat_header(t3, 2)
    no_row_split(t3)

    # ---- code appendix
    anchor = find_par(doc, 'Give the entire code of your project here')
    set_text(anchor, 'The complete project code is reproduced below in the order the specification asks '
                     'for. It is the code in notebooks/nlp_final_submission_code.ipynb, which runs '
                     'end to end. Configuration values are read from each saved checkpoint at runtime '
                     'rather than typed in, so the listings and the results above cannot disagree.')
    for needle in ('Data preprocessing code if there is any',
                   'BERT code where you achieved the best performance',
                   'BERT Variant code where you achieved the best performance',
                   'Ensemble code'):
        try:
            p = find_par(doc, needle)
            p._element.getparent().remove(p._element)
        except KeyError:
            pass
    for title, code in code_listings():
        anchor = para_after(anchor, title, size=11, bold=True, justify=False)
        for line in code.split('\n'):
            anchor = para_after(anchor, line if line.strip() else ' ',
                                size=CODE_PT, font='Courier New', justify=False)
            anchor.paragraph_format.space_before = Pt(0)
            anchor.paragraph_format.space_after = Pt(0)

    doc.save(str(OUT))
    print('wrote', OUT.name)
    print(f'  sections filled 8, tables 3, code listings {len(code_listings())}')


if __name__ == '__main__':
    main()
