import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# Ensure necessary NLTK data packages are downloaded
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'stopwords']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

FILLER_WORDS = set(['um', 'uh', 'er', 'ah', 'like', 'you know', 'well', 'i mean', 'basically', 'actually', 'sort of', 'kind of'])
SUBORDINATING_CONJUNCTIONS = set(['because', 'although', 'since', 'while', 'whereas', 'if', 'unless', 'which', 'that', 'whom', 'whose', 'where', 'when', 'after', 'before', 'though', 'even though'])
COORDINATING_CONJUNCTIONS = set(['and', 'but', 'or', 'so', 'yet', 'nor', 'for'])
MODAL_VERBS = set(['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'])

def extract_single_text_features(text):
    """
    Extracts comprehensive linguistic, grammatical, syntactic, and fluency metrics from a text transcript.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'text_len': 0,
            'word_count': 0,
            'sentence_count': 0,
            'is_empty_text': 1.0
        }

    clean_text = text.strip()
    words = re.findall(r'\b[a-zA-Z]+\b', clean_text.lower())
    total_words = len(words)
    
    if total_words == 0:
        return {'text_len': len(clean_text), 'word_count': 0, 'sentence_count': 0, 'is_empty_text': 1.0}

    features = {
        'text_len': len(clean_text),
        'word_count': total_words,
        'is_empty_text': 0.0
    }

    # 1. Lexical Richness & Diversity
    unique_words = set(words)
    total_unique = len(unique_words)
    word_lengths = [len(w) for w in words]
    
    features['avg_word_length'] = float(np.mean(word_lengths))
    features['std_word_length'] = float(np.std(word_lengths))
    features['long_words_count'] = sum(1 for w in words if len(w) >= 6)
    features['long_words_ratio'] = features['long_words_count'] / total_words
    
    # Type-Token Ratio (TTR) & Root TTR (Guiraud's Index)
    features['ttr'] = total_unique / total_words
    features['root_ttr'] = total_unique / np.sqrt(total_words)
    
    # Hapax Legomena (words appearing exactly once)
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    hapax_count = sum(1 for w, c in word_freq.items() if c == 1)
    features['hapax_ratio'] = hapax_count / total_words

    # 2. Hesitation, Fillers & Repetitions (Speech Fluency & Disfluency)
    filler_count = 0
    lower_text = clean_text.lower()
    for filler in FILLER_WORDS:
        filler_count += len(re.findall(r'\b' + re.escape(filler) + r'\b', lower_text))
    features['filler_count'] = filler_count
    features['filler_ratio'] = filler_count / total_words

    # Adjacent word repetitions (e.g., "I I", "the the", "very very")
    repetition_count = 0
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            repetition_count += 1
    features['word_repetition_count'] = repetition_count
    features['word_repetition_ratio'] = repetition_count / total_words

    # Incomplete sentence patterns
    features['ellipsis_count'] = len(re.findall(r'\.\.\.', clean_text))
    features['dash_disfluency'] = len(re.findall(r'--', clean_text))

    # 3. Sentences & Syntactic Complexity
    try:
        sentences = sent_tokenize(clean_text)
    except Exception:
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]
    
    num_sentences = max(1, len(sentences))
    features['sentence_count'] = num_sentences
    sent_lengths = [len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in sentences]
    features['avg_words_per_sentence'] = float(np.mean(sent_lengths))
    features['std_words_per_sentence'] = float(np.std(sent_lengths))
    features['max_words_per_sentence'] = float(np.max(sent_lengths))
    features['min_words_per_sentence'] = float(np.min(sent_lengths))

    # Clause markers & complex syntax connectors
    subord_count = sum(1 for w in words if w in SUBORDINATING_CONJUNCTIONS)
    coord_count = sum(1 for w in words if w in COORDINATING_CONJUNCTIONS)
    modal_count = sum(1 for w in words if w in MODAL_VERBS)

    features['subordinating_clause_count'] = subord_count
    features['subordinating_clause_ratio'] = subord_count / total_words
    features['subord_per_sentence'] = subord_count / num_sentences

    features['coordinating_conjunction_count'] = coord_count
    features['coordinating_conjunction_ratio'] = coord_count / total_words

    features['modal_verb_count'] = modal_count
    features['modal_verb_ratio'] = modal_count / total_words

    # 4. Readability & Textstat Indices
    try:
        features['flesch_reading_ease'] = float(textstat.flesch_reading_ease(clean_text))
        features['flesch_kincaid_grade'] = float(textstat.flesch_kincaid_grade(clean_text))
        features['gunning_fog'] = float(textstat.gunning_fog(clean_text))
        features['coleman_liau_index'] = float(textstat.coleman_liau_index(clean_text))
        features['automated_readability_index'] = float(textstat.automated_readability_index(clean_text))
        features['dale_chall_score'] = float(textstat.dale_chall_readability_score(clean_text))
        features['difficult_words'] = float(textstat.difficult_words(clean_text))
        features['difficult_words_ratio'] = features['difficult_words'] / total_words
    except Exception:
        features['flesch_reading_ease'] = 50.0
        features['flesch_kincaid_grade'] = 8.0
        features['gunning_fog'] = 10.0
        features['coleman_liau_index'] = 8.0
        features['automated_readability_index'] = 8.0
        features['dale_chall_score'] = 7.0
        features['difficult_words'] = 0.0
        features['difficult_words_ratio'] = 0.0

    # 5. POS (Part-of-Speech) Tagging Distributions
    try:
        tokens = word_tokenize(clean_text)
        tags = pos_tag(tokens)
        tag_counts = {}
        for _, tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        n_tags = max(1, len(tags))
        
        # Nouns: NN, NNS, NNP, NNPS
        noun_count = sum(tag_counts.get(t, 0) for t in ['NN', 'NNS', 'NNP', 'NNPS'])
        features['pos_noun_ratio'] = noun_count / n_tags

        # Verbs: VB, VBD, VBG, VBN, VBP, VBZ
        verb_count = sum(tag_counts.get(t, 0) for t in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        features['pos_verb_ratio'] = verb_count / n_tags
        
        # Verb tenses (Past vs Present vs Participle)
        past_verbs = sum(tag_counts.get(t, 0) for t in ['VBD', 'VBN'])
        present_verbs = sum(tag_counts.get(t, 0) for t in ['VB', 'VBP', 'VBZ', 'VBG'])
        features['pos_past_verb_ratio'] = past_verbs / max(1, verb_count)
        features['pos_present_verb_ratio'] = present_verbs / max(1, verb_count)

        # Adjectives: JJ, JJR, JJS
        adj_count = sum(tag_counts.get(t, 0) for t in ['JJ', 'JJR', 'JJS'])
        features['pos_adj_ratio'] = adj_count / n_tags

        # Adverbs: RB, RBR, RBS
        adv_count = sum(tag_counts.get(t, 0) for t in ['RB', 'RBR', 'RBS'])
        features['pos_adv_ratio'] = adv_count / n_tags

        # Prepositions / Subordinating conjunctions: IN
        prep_count = tag_counts.get('IN', 0)
        features['pos_prep_ratio'] = prep_count / n_tags

        # Pronouns: PRP, PRP$
        pron_count = sum(tag_counts.get(t, 0) for t in ['PRP', 'PRP$'])
        features['pos_pronoun_ratio'] = pron_count / n_tags

        # Determiners: DT
        dt_count = tag_counts.get('DT', 0)
        features['pos_determiner_ratio'] = dt_count / n_tags

    except Exception as e:
        for p in ['noun', 'verb', 'past_verb', 'present_verb', 'adj', 'adv', 'prep', 'pronoun', 'determiner']:
            features[f'pos_{p}_ratio'] = 0.0

    return features


def extract_all_text_features(train_transcripts_csv, test_transcripts_csv, out_train_csv, out_test_csv, n_tfidf_components=16):
    """
    Extracts text/grammar features and SVD-reduced TF-IDF embeddings from transcripts.
    """
    train_df = pd.read_csv(train_transcripts_csv)
    test_df = pd.read_csv(test_transcripts_csv)

    print("Extracting NLP & Grammar features for train transcripts...")
    train_feats = [extract_single_text_features(t) for t in tqdm(train_df['transcript'])]
    train_feat_df = pd.DataFrame(train_feats)
    train_feat_df['filename'] = train_df['filename']
    if 'label' in train_df.columns:
        train_feat_df['label'] = train_df['label']

    print("Extracting NLP & Grammar features for test transcripts...")
    test_feats = [extract_single_text_features(t) for t in tqdm(test_df['transcript'])]
    test_feat_df = pd.DataFrame(test_feats)
    test_feat_df['filename'] = test_df['filename']

    # TF-IDF & Truncated SVD for lexical semantic representation
    all_texts = train_df['transcript'].fillna("").tolist() + test_df['transcript'].fillna("").tolist()
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    tfidf_mat = tfidf.fit_transform(all_texts)
    
    svd = TruncatedSVD(n_components=n_tfidf_components, random_state=42)
    svd_mat = svd.fit_transform(tfidf_mat)
    
    train_svd = svd_mat[:len(train_df)]
    test_svd = svd_mat[len(train_df):]

    for i in range(n_tfidf_components):
        train_feat_df[f'tfidf_svd_{i}'] = train_svd[:, i]
        test_feat_df[f'tfidf_svd_{i}'] = test_svd[:, i]

    # Reorder columns with filename first
    cols_train = ['filename'] + [c for c in train_feat_df.columns if c not in ['filename', 'label']]
    if 'label' in train_feat_df.columns:
        cols_train.append('label')
    train_feat_df = train_feat_df[cols_train].fillna(0.0)

    cols_test = ['filename'] + [c for c in test_feat_df.columns if c not in ['filename', 'label']]
    test_feat_df = test_feat_df[cols_test].fillna(0.0)

    train_feat_df.to_csv(out_train_csv, index=False)
    test_feat_df.to_csv(out_test_csv, index=False)

    print(f"Saved {len(train_feat_df)} train feature records to {out_train_csv}")
    print(f"Saved {len(test_feat_df)} test feature records to {out_test_csv}")
    return train_feat_df, test_feat_df


if __name__ == "__main__":
    transcripts_train = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features\transcripts_train.csv"
    transcripts_test = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features\transcripts_test.csv"
    
    out_train_txt_feats = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features\text_features_train.csv"
    out_test_txt_feats = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features\text_features_test.csv"
    
    extract_all_text_features(transcripts_train, transcripts_test, out_train_txt_feats, out_test_txt_feats)
