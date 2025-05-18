import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_inference_time_varying_sizes(train_model):
    """異なるデータサイズでの推論時間を計測"""
    model, X_test, _ = train_model

    # データサイズごとの推論時間を計測
    sizes = [0.25, 0.5, 0.75, 1.0]
    times = []

    for size in sizes:
        n_samples = int(len(X_test) * size)
        X_subset = X_test.iloc[:n_samples]

        # 推論時間の計測（複数回の平均）
        n_runs = 5
        subset_times = []

        for _ in range(n_runs):
            start_time = time.time()
            model.predict(X_subset)
            end_time = time.time()
            subset_times.append(end_time - start_time)

        avg_time = sum(subset_times) / n_runs
        times.append(avg_time)

        # 大きなデータセットでも5秒以内に推論できることを確認
        assert (
            avg_time < 5.0
        ), f"{n_samples}サンプルの推論時間が長すぎます: {avg_time}秒"

    # データサイズと推論時間の関係が概ね線形であることを確認
    # (完全に線形でない場合もあるため、ゆるい検証)
    for i in range(len(sizes) - 1):
        ratio_size = sizes[i + 1] / sizes[i]
        ratio_time = times[i + 1] / max(times[i], 1e-6)  # ゼロ除算防止

        # 時間の増加率がデータサイズの増加率の2倍以下であることを確認
        assert (
            ratio_time < ratio_size * 2
        ), f"推論時間の増加が著しく非線形です: サイズ比={ratio_size}, 時間比={ratio_time}"


def test_model_detailed_metrics(train_model):
    """モデルの詳細な評価指標を検証"""
    model, X_test, y_test = train_model

    # 予測
    y_pred = model.predict(X_test)

    # 各種評価指標の計算
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 混同行列の計算
    cm = confusion_matrix(y_test, y_pred)

    # 評価指標の下限値を設定
    assert accuracy >= 0.75, f"精度が低すぎます: {accuracy}"
    assert precision >= 0.7, f"適合率が低すぎます: {precision}"
    assert recall >= 0.6, f"再現率が低すぎます: {recall}"
    assert f1 >= 0.65, f"F1スコアが低すぎます: {f1}"

    # クラス間のバランスを確認（片方のクラスだけを予測していないか）
    assert (
        cm[0, 0] > 0 and cm[1, 1] > 0
    ), f"片方のクラスのみを予測しています。混同行列: {cm}"


def test_model_probability_calibration(train_model):
    """確率予測の品質を検証"""
    model, X_test, y_test = train_model

    # 確率予測
    y_proba = model.predict_proba(X_test)

    # 確率の範囲が[0,1]であることを確認
    assert np.all(y_proba >= 0) and np.all(y_proba <= 1), "確率値が[0,1]の範囲外です"

    # 確率の合計が1になることを確認
    assert np.allclose(
        np.sum(y_proba, axis=1), 1.0
    ), "各サンプルの確率合計が1になっていません"

    # 高確率(>0.8)の予測の精度を検証
    high_conf_indices = np.where(np.max(y_proba, axis=1) > 0.8)[0]

    if len(high_conf_indices) > 0:
        high_conf_preds = np.argmax(y_proba[high_conf_indices], axis=1)
        high_conf_true = y_test.iloc[high_conf_indices].values
        high_conf_acc = accuracy_score(high_conf_true, high_conf_preds)

        # 高確率予測の精度が通常よりも高いことを確認
        assert high_conf_acc >= 0.8, f"高確率予測の精度が低すぎます: {high_conf_acc}"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"
