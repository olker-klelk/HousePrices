import numpy as np
import pandas as pd
import logging
import yaml
from typing import Tuple, Optional, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split
from preprocess import DataPreprocessor

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataLoader')

class DataLoader:
    def __init__(self, config_path: str = 'config.yml'):
        """
        初始化數據加載器

        Args:
                config_path: 配置文件路徑
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data_paths']['data_dir'])
        self.raw_data_path = self.data_dir / self.config['data_paths']['raw_train_data']
        self.processed_data_path = self.data_dir / self.config['data_paths']['processed_data']
        
        # 數據相關屬性
        self.data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def _load_config(self, config_path: str) -> Dict:
        """
        加載配置文件
        
        Args:
                config_path: YAML配置文件路徑
                
        Returns:
                配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_raw_data(self) -> pd.DataFrame:
        """
        加載原始數據
        
        Returns:
                加載的DataFrame
        """
        try:
            logger.info(f"Loading raw data from {self.raw_data_path}")
            self.data = pd.read_csv(self.raw_data_path)
            logger.info(f"Successfully loaded {len(self.data)} rows of data")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def check_data_quality(self) -> Dict:
        """
        檢查數據質量
        
        Returns:
                數據質量報告字典
        """
        if self.data is None:
                self.load_raw_data()
                
        quality_report = {
                'total_rows': len(self.data),
                'missing_values': self.data.isnull().sum().to_dict(),
                'duplicates': self.data.duplicated().sum(),
                'dtypes': self.data.dtypes.to_dict()
        }
        
        logger.info("Data quality check completed")
        return quality_report
    def split_data(self, 
                                     target_column: str,
                                     test_size: float = 0.2,
                                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分割訓練集和測試集
        
        Args:
                target_column: 目標變量的列名
                test_size: 測試集比例
                random_state: 隨機種子
                
        Returns:
                訓練集和測試集的元組
        """
        if self.data is None:
            self.load_raw_data()
                
        try:
            X = self.data.drop(columns=[target_column]).drop(columns=["Id"])
            y = self.data[target_column]
                
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state
            )
                
            self.train_data = pd.concat([X_train, y_train], axis=1)
            self.test_data = pd.concat([X_test, y_test], axis=1)
            logger.info(f"Data split completed. Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")
            return self.train_data, self.test_data
                
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
        
    def save_processed_data(self) -> None:
        """
        保存處理後的數據
        """
        if self.train_data is not None and self.test_data is not None:
            try:
                # 創建處理後數據的目錄
                self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)

                # 保存訓練集和測試集
                train_path = self.processed_data_path.parent / 'train.csv'
                test_path = self.processed_data_path.parent / 'test.csv'

                self.train_data.to_csv(train_path, index=False)
                self.test_data.to_csv(test_path, index=False)

                logger.info(f"Processed data saved to {self.processed_data_path.parent}")

            except Exception as e:
                logger.error(f"Error saving processed data: {str(e)}")
                raise
        else:
            logger.warning("No processed data to save. Please split the data first.")

    def get_feature_info(self) -> Dict:
        """
        獲取特徵信息
        
        Returns:
                特徵信息字典
        """
        if self.data is None:
            self.load_raw_data()
                
        feature_info = {
            'numerical_features': list(self.data.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_features': list(self.data.select_dtypes(include=['object']).columns),
            'feature_descriptions': {
                column: {
                    'dtype': str(self.data[column].dtype),
                    'unique_values': len(self.data[column].unique()),
                    'missing_values': self.data[column].isnull().sum()
                }
                for column in self.data.columns
            }
        }
        
        return feature_info

if __name__ == "__main__":

    # 顯示所有列
    #pd.set_option('display.max_columns', None)
    # 顯示所有行
    #pd.set_option('display.max_rows', None)

    loader = DataLoader('config.yml')
    loader.load_raw_data()
    # 檢查數據質量
    quality_report = loader.check_data_quality()
    # print("Data Quality Report:")
    # for report in quality_report:
    #     print(report, quality_report[report], "\n");

    train_data, test_data = loader.split_data(target_column='SalePrice')

    preprocessor = DataPreprocessor('config.yml')
    loader.train_data = preprocessor.handle_missing_values(train_data)
    loader.train_data = preprocessor.handle_outliers(loader.train_data)
    loader.train_data = preprocessor.scale_numerical_features(loader.train_data, True)
    loader.train_data = preprocessor.encode_categorical_features(loader.train_data, fit=True)
    #print(train_data);
    loader.save_processed_data()
    feature_info = loader.get_feature_info()
    #print("Feature Information:")
    #for report in feature_info:
    #    print(report, feature_info[report], "\n");
