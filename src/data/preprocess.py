import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
import yaml
import joblib

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Preprocessor')

class DataPreprocessor:
    def __init__(self, config_path: str = 'config.yml'):
        """
        初始化預處理器
        
        Args:
            config_path: 配置文件路徑
        """
        self.config = self._load_config(config_path)
        
        # 初始化轉換器
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.one_hot_encoders: Dict[str, OneHotEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}

        
    def _load_config(self, config_path: str) -> Dict:
        """加載配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def handle_missing_values(self, 
                            data: pd.DataFrame,
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        處理缺失值
        
        Args:
            data: 輸入數據框
            strategy: 每個特徵的缺失值處理策略
            
        Returns:
            處理後的數據框
        """
        if strategy is None:
            strategy = {
                'numerical': 'mean',
                'categorical': 'most_frequent'
            }
        
        df = data.copy()
        # 處理數值型特徵的缺失值
        for feature in list(df.select_dtypes(include=['int64', 'float64']).columns):

            if feature not in self.imputers:
                self.imputers[feature] = SimpleImputer(
                    strategy=strategy['numerical']
                )
            
            df[feature] = self.imputers[feature].fit_transform(
                df[[feature]]
            )
            # Save na strategy_mean
            #print("\n ", feature,"mean : ", self.imputers[feature].statistics_[0]);
        # 處理類別型特徵的缺失值
        for feature in list(df.select_dtypes(include=['object']).columns):
            if feature not in self.imputers:
                self.imputers[feature] = SimpleImputer(
                    strategy=strategy['categorical']
                )
            df[feature] = self.imputers[feature].fit_transform(
                df[[feature]]
            ).ravel()
                        # Save na strategy_most_frequent
            #print("\n ", feature,"mean : ", self.imputers[feature].statistics_[0]);
        logger.info("Missing values handled")
        return df
    
    def scale_numerical_features(self, 
                               data: pd.DataFrame,
                               fit: bool = True) -> pd.DataFrame:
        """
        標準化數值特徵
        
        Args:
            data: 輸入數據框
            fit: 是否需要擬合轉換器
            
        Returns:
            標準化後的數據框
        """
        df = data.copy()
        
        for feature in list(df.select_dtypes(include=['int64', 'float64']).columns):
            if fit and feature not in self.scalers:
                self.scalers[feature] = StandardScaler()
                df[feature] = self.scalers[feature].fit_transform(
                    df[[feature]]
                )
            else:
                df[feature] = self.scalers[feature].transform(
                    df[[feature]]
                )

        logger.info("Numerical features scaled")
        return df
    
    def encode_categorical_features(self,
                                  data: pd.DataFrame,
                                  encoding_type: str = 'onehot',
                                  fit: bool = True) -> pd.DataFrame:
        """
        編碼類別特徵
        
        Args:
            data: 輸入數據框
            encoding_type: 編碼類型 ('label' 或 'onehot')
            fit: 是否需要擬合轉換器
            
        Returns:
            編碼後的數據框
        """
        df = data.copy()
        
        if encoding_type == 'label':
            for feature in list(df.select_dtypes(include=['object']).columns):
                if fit and feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    df[feature] = self.label_encoders[feature].fit_transform(
                        df[feature]
                    )
                else:
                    df[feature] = self.label_encoders[feature].transform(
                        df[feature]
                    )
        
        elif encoding_type == 'onehot':
            for feature in list(df.select_dtypes(include=['object']).columns):
                if fit and feature not in self.one_hot_encoders:
                    self.one_hot_encoders[feature] = OneHotEncoder(
                        sparse_output=False,
                        handle_unknown='ignore'
                    )
                    # 轉換並創建新列
                    encoded_features = self.one_hot_encoders[feature].fit_transform(
                        df[[feature]]
                    )
                    encoded_feature_names = [
                        f"{feature}_{i}" for i in range(encoded_features.shape[1])
                    ]
                    encoded_df = pd.DataFrame(
                        encoded_features,
                        columns=encoded_feature_names,
                        index=df.index
                    )
                    # 刪除原始列並添加編碼後的列
                    df = df.drop(columns=[feature])
                    df = pd.concat([df, encoded_df], axis=1)
                else:
                    encoded_features = self.one_hot_encoders[feature].transform(
                        df[[feature]]
                    )
                    encoded_feature_names = [
                        f"{feature}_{i}" for i in range(encoded_features.shape[1])
                    ]
                    encoded_df = pd.DataFrame(
                        encoded_features,
                        columns=encoded_feature_names,
                        index=df.index
                    )
                    df = df.drop(columns=[feature])
                    df = pd.concat([df, encoded_df], axis=1)
        
        logger.info(f"Categorical features encoded using {encoding_type} encoding")
        return df
    
    def handle_outliers(self,
                       data: pd.DataFrame,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        處理異常值
        
        Args:
            data: 輸入數據框
            method: 異常值處理方法 ('iqr' 或 'zscore')
            threshold: 閾值
            
        Returns:
            處理後的數據框
        """
        df = data.copy()
        
        for feature in list(df.select_dtypes(include=['int64', 'float64']).columns):
            if method == 'iqr':
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                # print("\n", feature , f"lower :{lower_bound} upper {upper_bound}")
                # print("\n lower : ", len(df[feature][df[feature] < lower_bound]))
                # print("\n upper : ", len(df[feature][df[feature] > upper_bound]))
                df[feature] = df[feature].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                mean = df[feature].mean()
                std = df[feature].std()
                df[feature] = df[feature].clip(
                    mean - threshold * std,
                    mean + threshold * std
                )

        
        logger.info(f"Outliers handled using {method} method")
        return df
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        創建新特徵
        
        Args:
            data: 輸入數據框
            
        Returns:
            添加新特徵後的數據框
        """
        df = data.copy()
        
        # 這裡添加特徵工程的邏輯
        # 例如：組合現有特徵、創建交互特徵等
        
        logger.info("New features created")
        return df
    
    def save_preprocessor(self, filepath: Union[str, Path]):
        """保存預處理器的狀態"""
        preprocessor_state = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'one_hot_encoders': self.one_hot_encoders,
            'imputers': self.imputers,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(preprocessor_state, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: Union[str, Path]):
        """加載預處理器的狀態"""
        preprocessor_state = joblib.load(filepath)
        
        self.scalers = preprocessor_state['scalers']
        self.label_encoders = preprocessor_state['label_encoders']
        self.one_hot_encoders = preprocessor_state['one_hot_encoders']
        self.imputers = preprocessor_state['imputers']
        self.numerical_features = preprocessor_state['numerical_features']
        self.categorical_features = preprocessor_state['categorical_features']
        
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def preprocess_data(self,
                       data: pd.DataFrame,
                       fit: bool = True) -> pd.DataFrame:
        """
        執行完整的預處理流程
        
        Args:
            data: 輸入數據框
            fit: 是否需要擬合轉換器
            
        Returns:
            預處理後的數據框
        """
        df = data.copy()
        
        # 1. 處理缺失值
        df = self.handle_missing_values(df)
        
        # 2. 處理異常值
        df = self.handle_outliers(df)
        
        # 3. 特徵工程
        df = self.create_features(df)
        
        # 4. 標準化數值特徵
        df = self.scale_numerical_features(df, fit)
        
        # 5. 編碼類別特徵
        df = self.encode_categorical_features(df, fit=fit)
        
        logger.info("Full preprocessing pipeline completed")
        return df

# 使用示例
if __name__ == "__main__":
    # 配置文件示例 (config.yml):
    """
    paths:
      model_dir: "models"
      
    features:
      numerical_features:
        - "age"
        - "income"
        - "credit_score"
      categorical_features:
        - "education"
        - "occupation"
        - "marital_status"
      target_column: "loan_status"
    """
    

    
    # 初始化預處理器
    #preprocessor = DataPreprocessor('config.yml')
    
    # 執行預處理
    #processed_data = preprocessor.preprocess_data(data)
    
    # 保存預處理器
    # preprocessor.save_preprocessor('models/preprocessor.joblib')
    
    # print("Processed data shape:", processed_data.shape)
    # print("\nProcessed data head:")
    # print(processed_data.head())
