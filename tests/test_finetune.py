import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "model")))

from finetune_cobia import finetune_model as finetune_cobia_model
from finetune_oyster import finetune_model as finetune_oyster_model

class TestFinetune(unittest.TestCase):
    @patch('os.path.exists')
    def test_finetune_cobia_base_model_missing(self, mock_exists):
        mock_exists.return_value = False
        
        result = finetune_cobia_model(
            base_model_path='base.pkl',
            new_data_path='new_data.csv',
            output_path='out.pkl',
            features_list=['DO']
        )
        self.assertIsNone(result)

    @patch('os.path.exists')
    def test_finetune_oyster_base_model_missing(self, mock_exists):
        mock_exists.return_value = False
        
        result = finetune_oyster_model(
            base_model_path='base.pkl',
            new_data_path='new_data.csv',
            output_path='out.pkl',
            features_list=['DO']
        )
        self.assertIsNone(result)

    @patch('os.path.exists')
    @patch('joblib.load')
    @patch('finetune_cobia.prepare_time_series_data')
    @patch('finetune_cobia.temporal_train_val_split')
    @patch('joblib.dump')
    def test_finetune_cobia_success_flow(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.side_effect = lambda path: True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (['DO_lag1'], ['DO'])
        ]
        
        import pandas as pd
        mock_df = pd.DataFrame({
            'DO': [5.0, 6.0],
            'DO_lag1': [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ['DO_lag1'])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_cobia_model(
            base_model_path='base.pkl',
            new_data_path='new_data.csv',
            output_path='out.pkl',
            features_list=['DO']
        )
        
        self.assertTrue(mock_exists.called)
        self.assertTrue(mock_load.called)
        self.assertTrue(mock_prepare.called)
        self.assertTrue(mock_split.called)
        self.assertTrue(mock_estimator.fit.called)
        self.assertTrue(mock_dump.called)

    @patch('os.path.exists')
    @patch('joblib.load')
    @patch('finetune_oyster.prepare_time_series_data')
    @patch('finetune_oyster.temporal_train_val_split')
    @patch('joblib.dump')
    def test_finetune_oyster_success_flow(self, mock_dump, mock_split, mock_prepare, mock_load, mock_exists):
        mock_exists.side_effect = lambda path: True
        
        mock_estimator = MagicMock()
        mock_estimator.get_booster.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_model.estimators_ = [mock_estimator]
        
        mock_load.side_effect = [
            mock_model,
            (['DO_lag1'], ['DO'])
        ]
        
        import pandas as pd
        mock_df = pd.DataFrame({
            'DO': [5.0, 6.0],
            'DO_lag1': [4.0, 5.0]
        })
        
        mock_prepare.return_value = (mock_df, ['DO_lag1'])
        mock_split.return_value = (mock_df, mock_df)
        
        mock_model.predict.return_value = [[5.1], [5.9]]
        
        finetune_oyster_model(
            base_model_path='base.pkl',
            new_data_path='new_data.csv',
            output_path='out.pkl',
            features_list=['DO']
        )
        
        self.assertTrue(mock_exists.called)
        self.assertTrue(mock_load.called)
        self.assertTrue(mock_prepare.called)
        self.assertTrue(mock_split.called)
        self.assertTrue(mock_estimator.fit.called)
        self.assertTrue(mock_dump.called)

if __name__ == '__main__':
    unittest.main()
