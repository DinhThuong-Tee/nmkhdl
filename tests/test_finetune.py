import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "model")))

from finetune_cobia import finetune_model as finetune_cobia_model
from finetune_oyster import finetune_model as finetune_oyster_model

class TestFinetuneCobia(unittest.TestCase):
    @patch("finetune_cobia.os.path.exists")
    def test_cobia_base_model_not_found(self, mock_exists):
        mock_exists.side_effect = lambda path: path != "base.pkl"
        
        result = finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    def test_cobia_new_data_not_found(self, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), (["DO_lag1"], ["DO"])]
        mock_prepare.side_effect = FileNotFoundError("File not found")
        
        with self.assertRaises(FileNotFoundError):
            finetune_cobia_model(
                base_model_path="base.pkl",
                new_data_path="new_data.csv",
                output_path="out.pkl",
                features_list=["DO"]
            )

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    def test_cobia_metadata_not_found(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), Exception("Metadata not found")]
        
        result = finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    def test_cobia_metadata_corrupted(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), ValueError("Corrupted data")]
        
        result = finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    def test_cobia_prepare_data_returns_none(self, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), (["DO_lag1"], ["DO"])]
        mock_prepare.return_value = (None, [])
        
        result = finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    def test_cobia_prepare_data_returns_empty(self, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), (["DO_lag1"], ["DO"])]
        mock_prepare.return_value = (pd.DataFrame(), [])
        
        result = finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_success_flow(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1"], ["DO"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        
        self.assertTrue(mock_exists.called)
        self.assertTrue(mock_load.called)
        self.assertTrue(mock_prepare.called)
        self.assertTrue(mock_split.called)
        self.assertTrue(mock_estimator.fit.called)
        self.assertTrue(mock_dump.called)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_success_multiple_estimators(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        
        mock_est1 = MagicMock()
        mock_est1.get_booster.return_value = MagicMock()
        
        mock_est2 = MagicMock()
        mock_est2.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_est1, mock_est2]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1", "pH_lag1"], ["DO", "pH"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0],
            "pH": [7.0, 8.0],
            "pH_lag1": [6.9, 7.8]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1", "pH_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = np.array([[5.1, 7.1], [5.9, 7.9]])
        
        finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO", "pH"]
        )
        
        self.assertEqual(mock_est1.fit.call_count, 1)
        self.assertEqual(mock_est2.fit.call_count, 1)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_set_params_propagation(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1"], ["DO"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        
        mock_estimator.set_params.assert_called_with(
            learning_rate=0.005,
            n_estimators=500,
            early_stopping_rounds=30
        )

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_predict_valid_handling(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1"], ["DO"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        
        self.assertTrue(mock_dump.called)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_path_conversions(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        from pathlib import Path
        mock_exists.return_value = True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1"], ["DO"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_cobia_model(
            base_model_path=Path("base.pkl"),
            new_data_path=Path("new_data.csv"),
            output_path=Path("out.pkl"),
            features_list=["DO"]
        )
        
        self.assertTrue(mock_dump.called)


class TestFinetuneOyster(unittest.TestCase):
    @patch("finetune_oyster.os.path.exists")
    def test_oyster_base_model_not_found(self, mock_exists):
        mock_exists.side_effect = lambda path: path != "base.pkl"
        
        result = finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    def test_oyster_new_data_not_found(self, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), (["DO_lag1"], ["DO"])]
        mock_prepare.side_effect = FileNotFoundError("File not found")
        
        with self.assertRaises(FileNotFoundError):
            finetune_oyster_model(
                base_model_path="base.pkl",
                new_data_path="new_data.csv",
                output_path="out.pkl",
                features_list=["DO"]
            )

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    def test_oyster_metadata_not_found(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), Exception("Metadata not found")]
        
        result = finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    def test_oyster_metadata_corrupted(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), ValueError("Corrupted data")]
        
        result = finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    def test_oyster_prepare_data_returns_none(self, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), (["DO_lag1"], ["DO"])]
        mock_prepare.return_value = (None, [])
        
        result = finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    def test_oyster_prepare_data_returns_empty(self, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_load.side_effect = [MagicMock(), (["DO_lag1"], ["DO"])]
        mock_prepare.return_value = (pd.DataFrame(), [])
        
        result = finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertIsNone(result)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_success_flow(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1"], ["DO"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        
        self.assertTrue(mock_exists.called)
        self.assertTrue(mock_load.called)
        self.assertTrue(mock_prepare.called)
        self.assertTrue(mock_split.called)
        self.assertTrue(mock_estimator.fit.called)
        self.assertTrue(mock_dump.called)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_success_multiple_estimators(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        
        mock_est1 = MagicMock()
        mock_est1.get_booster.return_value = MagicMock()
        
        mock_est2 = MagicMock()
        mock_est2.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_est1, mock_est2]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1", "pH_lag1"], ["DO", "pH"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0],
            "pH": [7.0, 8.0],
            "pH_lag1": [6.9, 7.8]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1", "pH_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = np.array([[5.1, 7.1], [5.9, 7.9]])
        
        finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO", "pH"]
        )
        
        self.assertEqual(mock_est1.fit.call_count, 1)
        self.assertEqual(mock_est2.fit.call_count, 1)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_set_params_propagation(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1"], ["DO"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        
        mock_estimator.set_params.assert_called_with(
            learning_rate=0.005,
            n_estimators=500,
            early_stopping_rounds=30
        )

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_predict_valid_handling(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1"], ["DO"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        
        self.assertTrue(mock_dump.called)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_path_conversions(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        from pathlib import Path
        mock_exists.return_value = True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (["DO_lag1"], ["DO"])
        ]
        
        mock_df = pd.DataFrame({
            "DO": [5.0, 6.0],
            "DO_lag1": [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_oyster_model(
            base_model_path=Path("base.pkl"),
            new_data_path=Path("new_data.csv"),
            output_path=Path("out.pkl"),
            features_list=["DO"]
        )
        
        self.assertTrue(mock_dump.called)


class TestFinetuneEdgeCases(unittest.TestCase):
    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_empty_estimators_list(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.estimators_ = []
        mock_load.side_effect = [mock_model, ([], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [5.0], "DO_lag1": [4.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        mock_model.predict.return_value = np.array([]).reshape(0, 0)

        finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=[]
        )
        self.assertTrue(mock_dump.called)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_empty_estimators_list(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.estimators_ = []
        mock_load.side_effect = [mock_model, ([], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [5.0], "DO_lag1": [4.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        mock_model.predict.return_value = np.array([]).reshape(0, 0)

        finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=[]
        )
        self.assertTrue(mock_dump.called)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_extremely_large_rmse(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [mock_model, (["DO_lag1"], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [1.0, 1000.0], "DO_lag1": [0.0, 999.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        mock_model.predict.return_value = np.array([[100000.0], [-100000.0]])

        finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertTrue(mock_dump.called)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_extremely_large_rmse(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [mock_model, (["DO_lag1"], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [1.0, 1000.0], "DO_lag1": [0.0, 999.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        mock_model.predict.return_value = np.array([[100000.0], [-100000.0]])

        finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertTrue(mock_dump.called)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_prediction_zero_rmse(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [mock_model, (["DO_lag1"], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [5.0, 6.0], "DO_lag1": [4.0, 5.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        mock_model.predict.return_value = np.array([[5.0], [6.0]])

        finetune_cobia_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertTrue(mock_dump.called)

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_prediction_zero_rmse(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [mock_model, (["DO_lag1"], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [5.0, 6.0], "DO_lag1": [4.0, 5.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        mock_model.predict.return_value = np.array([[5.0], [6.0]])

        finetune_oyster_model(
            base_model_path="base.pkl",
            new_data_path="new_data.csv",
            output_path="out.pkl",
            features_list=["DO"]
        )
        self.assertTrue(mock_dump.called)

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_fit_raises_exception(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        mock_estimator.fit.side_effect = RuntimeError("Training crashed")
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [mock_model, (["DO_lag1"], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [5.0, 6.0], "DO_lag1": [4.0, 5.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)

        with self.assertRaises(RuntimeError):
            finetune_cobia_model(
                base_model_path="base.pkl",
                new_data_path="new_data.csv",
                output_path="out.pkl",
                features_list=["DO"]
            )

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_fit_raises_exception(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        mock_estimator.fit.side_effect = RuntimeError("Training crashed")
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [mock_model, (["DO_lag1"], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [5.0, 6.0], "DO_lag1": [4.0, 5.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)

        with self.assertRaises(RuntimeError):
            finetune_oyster_model(
                base_model_path="base.pkl",
                new_data_path="new_data.csv",
                output_path="out.pkl",
                features_list=["DO"]
            )

    @patch("finetune_cobia.os.path.exists")
    @patch("finetune_cobia.joblib.load")
    @patch("finetune_cobia.prepare_time_series_data")
    @patch("finetune_cobia.temporal_train_val_split")
    @patch("finetune_cobia.joblib.dump")
    def test_cobia_dump_raises_exception(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [mock_model, (["DO_lag1"], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [5.0, 6.0], "DO_lag1": [4.0, 5.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        mock_model.predict.return_value = [[5.1], [5.9]]
        mock_dump.side_effect = IOError("Disk full")

        with self.assertRaises(IOError):
            finetune_cobia_model(
                base_model_path="base.pkl",
                new_data_path="new_data.csv",
                output_path="out.pkl",
                features_list=["DO"]
            )

    @patch("finetune_oyster.os.path.exists")
    @patch("finetune_oyster.joblib.load")
    @patch("finetune_oyster.prepare_time_series_data")
    @patch("finetune_oyster.temporal_train_val_split")
    @patch("finetune_oyster.joblib.dump")
    def test_oyster_dump_raises_exception(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [mock_model, (["DO_lag1"], ["DO"])]
        
        mock_df = pd.DataFrame({"DO": [5.0, 6.0], "DO_lag1": [4.0, 5.0]})
        mock_prepare.return_value = (mock_df, ["DO_lag1"])
        mock_split.return_value = (mock_df, mock_df)
        mock_model.predict.return_value = [[5.1], [5.9]]
        mock_dump.side_effect = IOError("Disk full")

        with self.assertRaises(IOError):
            finetune_oyster_model(
                base_model_path="base.pkl",
                new_data_path="new_data.csv",
                output_path="out.pkl",
                features_list=["DO"]
            )

if __name__ == "__main__":
    unittest.main()
