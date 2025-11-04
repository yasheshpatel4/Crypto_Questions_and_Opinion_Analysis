import argparse
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

LEVEL1_MAP = {0: "noise", 1: "objective", 2: "subjective", "0": "noise", "1": "objective", "2": "subjective", "0.0": "noise", "1.0": "objective", "2.0": "subjective"}
LEVEL2_SUBJECTIVE = {0: "neutral", 1: "negative", 2: "positive", "0": "neutral", "1": "negative", "2": "positive", "0.0": "neutral", "1.0": "negative", "2.0": "positive"}
LEVEL3_NEUTRAL = {0: "neutral_sentiments", 1: "questions", 2: "advertisements", 3: "misc", "0": "neutral_sentiments", "1": "questions", "2": "advertisements", "3": "misc", "0.0": "neutral_sentiments", "1.0": "questions", "2.0": "advertisements", "3.0": "misc"}

RANDOM_STATE = 42
MAX_WORDS = 40000
MAX_LEN = 160
EMBED_DIM = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 2


def build_text(df: pd.DataFrame) -> pd.Series:
	candidates = ["title", "MAIN", "selftext", "text", "content", "body"]
	primary = None
	for c in candidates:
		if c in df.columns:
			primary = c
			break
	secondary = None
	for c in candidates:
		if c in df.columns and c != primary:
			secondary = c
			break
	if primary is None:
		primary = df.columns[0]
	if secondary is None:
		secondary = primary
	texts = df[primary].astype(str).fillna("") + " " + df[secondary].astype(str).fillna("")
	return texts


def tokenize_fit(texts: pd.Series):
	tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
	tok.fit_on_texts(texts.tolist())
	return tok


def texts_to_padded(tok: Tokenizer, texts: pd.Series):
	seqs = tok.texts_to_sequences(texts.tolist())
	return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")


def pad_single(tok: Tokenizer, text: str):
	seq = tok.texts_to_sequences([text])
	return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res


def build_transformer_model(num_classes: int):
	inputs = Input(shape=(MAX_LEN,))
	x = Embedding(MAX_WORDS, EMBED_DIM)(inputs)
	for _ in range(NUM_LAYERS):
		x = transformer_encoder(x, EMBED_DIM // NUM_HEADS, NUM_HEADS, FF_DIM, dropout=0.1)
	x = GlobalAveragePooling1D()(x)
	x = Dropout(0.1)(x)
	x = Dense(64, activation="relu")(x)
	x = Dropout(0.1)(x)
	outputs = Dense(num_classes, activation="softmax")(x)
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	return model


def train_quiet(model, x_train, y_train, x_val, y_val, class_weights=None):
	es = EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64, callbacks=[es], verbose=0, class_weight=class_weights)
	preds = np.argmax(model.predict(x_val, verbose=0), axis=1)
	acc = accuracy_score(y_val, preds)
	return model, acc, preds


def predict_all_levels_path(sentence: str) -> str:
	import pickle
	with open("final_tokenizer.pkl", "rb") as f:
		tok = pickle.load(f)
	X = pad_single(tok, sentence)
	classes_l1 = ["noise","objective","subjective"]
	for cand in ["final_transformer_level1.keras"]:
		if os.path.exists(cand):
			model_l1 = load_model(cand, custom_objects={'transformer_encoder': transformer_encoder})
			label1 = classes_l1[int(np.argmax(model_l1.predict(X, verbose=0)[0]))]
			if label1 != "subjective":
				return label1
			# Level 2
			if os.path.exists("final_transformer_level2_level1.keras"):
				classes_l2 = ["neutral","negative","positive"]
				model_l2 = load_model("final_transformer_level2_level1.keras", custom_objects={'transformer_encoder': transformer_encoder})
				label2 = classes_l2[int(np.argmax(model_l2.predict(X, verbose=0)[0]))]
				if label2 != "neutral":
					return f"subjective->{label2}"
				# Level 3
				if os.path.exists("final_transformer_level3_level2.keras"):
					model_l3 = load_model("final_transformer_level3_level2.keras", custom_objects={'transformer_encoder': transformer_encoder})
					classes_l3 = ["neutral_sentiments","questions","advertisements","misc"]
					label3 = classes_l3[int(np.argmax(model_l3.predict(X, verbose=0)[0]))]
					return f"subjective->neutral->{label3}"
				return "subjective->neutral"
			return "subjective"
	raise FileNotFoundError("Models not found. Train first.")


def main():
	parser = argparse.ArgumentParser(description="Transformer-based hierarchical training and predict_text path output")
	parser.add_argument("--mode", choices=["train","predict_text"], default="train")
	parser.add_argument("--csv", default="sample_data.csv")
	parser.add_argument("--label_l1", default="Level 1")
	parser.add_argument("--label_l2", default=None)
	parser.add_argument("--label_l3", default=None)
	parser.add_argument("--text", default=None)
	args = parser.parse_args()

	if args.mode == "predict_text":
		if not args.text:
			raise ValueError("--text is required")
		print(predict_all_levels_path(args.text))
		return

	# TRAIN ALL LEVELS IN ONE RUN
	df = pd.read_csv(args.csv)
	texts = build_text(df)
	y1 = df[args.label_l1].map(LEVEL1_MAP).fillna(df[args.label_l1]).astype(str)
	classes_l1 = ["noise","objective","subjective"]
	l1_to_idx = {c:i for i,c in enumerate(classes_l1)}
	y1_idx = y1.map(l1_to_idx).astype(int).values

	tok = tokenize_fit(texts)
	X = texts_to_padded(tok, texts)
	X_tr, X_va, y1_tr, y1_va = train_test_split(X, y1_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y1_idx)

	# class weights for imbalanced Level 1
	classes = np.unique(y1_tr)
	weights = compute_class_weight(class_weight='balanced', classes=classes, y=y1_tr)
	class_weights_l1 = {int(c): float(w) for c, w in zip(classes, weights)}

	# Use transformer for Level 1
	m = build_transformer_model(len(classes_l1))
	m, acc, preds = train_quiet(m, X_tr, y1_tr, X_va, y1_va, class_weights=class_weights_l1)
	m.save("final_transformer_level1.keras")
	with open("final_tokenizer.pkl", "wb") as f:
		import pickle
		pickle.dump(tok, f)

	# Prepare masks on validation set for deeper levels
	l1_true = y1_va
	l1_pred = preds

	# Level 2
	acc2, preds2, y2_va_final = None, None, None
	if args.label_l2 and args.label_l2 in df.columns:
		subj_mask_all = (y1 == "subjective").values
		texts_l2_all = texts[subj_mask_all]
		y2_raw_all = df.loc[subj_mask_all, args.label_l2]
		y2_all = y2_raw_all.map(LEVEL2_SUBJECTIVE).fillna(y2_raw_all).astype(str)
		label2 = ["neutral","negative","positive"]
		l2_to_idx = {c:i for i,c in enumerate(label2)}
		valid_all = y2_all.isin(label2).values
		X2_all = texts_to_padded(tok, texts_l2_all[valid_all])
		y2_all_idx = y2_all[valid_all].map(l2_to_idx).astype(int).values
		if len(y2_all_idx) > 5 and len(np.unique(y2_all_idx)) > 1:
			X2_tr, X2_va, y2_tr, y2_va = train_test_split(X2_all, y2_all_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y2_all_idx)
			# class weights level2
			classes2 = np.unique(y2_tr)
			weights2 = compute_class_weight(class_weight='balanced', classes=classes2, y=y2_tr)
			cw2 = {int(c): float(w) for c, w in zip(classes2, weights2)}
			m2 = build_transformer_model(len(label2))
			m2, acc2, preds2 = train_quiet(m2, X2_tr, y2_tr, X2_va, y2_va, class_weights=cw2)
			m2.save("final_transformer_level2_level1.keras")
			y2_va_final = y2_va

	# Level 3
	acc3, preds3, y3_va_final = None, None, None
	if args.label_l3 and args.label_l3 in df.columns and args.label_l2 and args.label_l2 in df.columns:
		subj_mask_all = (y1 == "subjective").values
		y2_raw_all = df.loc[subj_mask_all, args.label_l2]
		y2_all = y2_raw_all.map(LEVEL2_SUBJECTIVE).fillna(y2_raw_all).astype(str)
		neutral_mask_all = (y2_all == "neutral").values
		texts_l3_all = texts[subj_mask_all][neutral_mask_all]
		if len(texts_l3_all) > 0:
			y3_raw_all = df.loc[subj_mask_all, args.label_l3][neutral_mask_all]
			y3_all = y3_raw_all.map(LEVEL3_NEUTRAL).fillna(y3_raw_all).astype(str)
			label3 = ["neutral_sentiments","questions","advertisements","misc"]
			l3_to_idx = {c:i for i,c in enumerate(label3)}
			valid3_all = y3_all.isin(label3).values
			X3_all = texts_to_padded(tok, texts_l3_all[valid3_all])
			y3_all_idx = y3_all[valid3_all].map(l3_to_idx).astype(int).values
			if len(y3_all_idx) > 5 and len(np.unique(y3_all_idx)) > 1:
				X3_tr, X3_va, y3_tr, y3_va = train_test_split(X3_all, y3_all_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y3_all_idx)
				# class weights level3
				classes3 = np.unique(y3_tr)
				weights3 = compute_class_weight(class_weight='balanced', classes=classes3, y=y3_tr)
				cw3 = {int(c): float(w) for c, w in zip(classes3, weights3)}
				m3 = build_transformer_model(len(label3))
				m3, acc3, preds3 = train_quiet(m3, X3_tr, y3_tr, X3_va, y3_va, class_weights=cw3)
				m3.save("final_transformer_level3_level2.keras")
				y3_va_final = y3_va

	# Compute single overall hierarchical accuracy
	overall_correct = (preds == y1_va).astype(int)
	if y2_va_final is not None and preds2 is not None:
		overall_correct = np.concatenate([overall_correct, (preds2 == y2_va_final).astype(int)])
	if y3_va_final is not None and preds3 is not None:
		overall_correct = np.concatenate([overall_correct, (preds3 == y3_va_final).astype(int)])
	accuracy = float(np.mean(overall_correct)) if overall_correct.size > 0 else 0.0
	print(f"accuracy: {accuracy:.4f}")


if __name__ == "__main__":
	main()
