import os
import json
import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessor for recommendation models
    """
    
    def __init__(self, db_host, db_port, db_name, db_user, db_password):
        """Initialize the data preprocessor with database connection parameters"""
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'dbname': db_name,
            'user': db_user,
            'password': db_password
        }
    
    def _connect_db(self):
        """Create a new database connection"""
        try:
            connection = psycopg2.connect(**self.db_config)
            return connection
        except psycopg2.Error as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise
            
    def get_user_data(self):
        """
        Get user data from PostgreSQL
        Returns a DataFrame with user information
        """
        try:
            connection = self._connect_db()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Query to get user data
            query = """
            SELECT 
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.created_at,
                MAX(o.created_at) as last_order_date,
                COUNT(DISTINCT o.order_id) as order_count,
                SUM(o.total_amount) as total_spent
            FROM 
                users u
            LEFT JOIN 
                orders o ON u.user_id = o.user_id
            GROUP BY 
                u.user_id
            ORDER BY 
                u.user_id
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(rows)
            
            if df.empty:
                logger.warning("No user data found")
                return pd.DataFrame()
            
            logger.info(f"Loaded data for {len(df)} users")
            return df
            
        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            raise
    
    def get_product_data(self):
        """
        Get product data from PostgreSQL
        Returns a DataFrame with product information
        """
        try:
            connection = self._connect_db()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Query to get product data with sales metrics
            query = """
            SELECT 
                p.product_id,
                p.name,
                p.description,
                p.price,
                p.category_id,
                c.name as category_name,
                COUNT(DISTINCT oi.order_id) as order_count,
                SUM(oi.quantity) as total_quantity_sold
            FROM 
                products p
            LEFT JOIN 
                categories c ON p.category_id = c.category_id
            LEFT JOIN 
                order_items oi ON p.product_id = oi.product_id
            LEFT JOIN 
                orders o ON oi.order_id = o.order_id AND o.status = 'COMPLETED'
            GROUP BY 
                p.product_id, c.name
            ORDER BY 
                p.product_id
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(rows)
            
            if df.empty:
                logger.warning("No product data found")
                return pd.DataFrame()
            
            logger.info(f"Loaded data for {len(df)} products")
            return df
            
        except Exception as e:
            logger.error(f"Error getting product data: {e}")
            raise
    
    def get_transaction_data(self, start_date=None, end_date=None):
        """
        Get transaction data from PostgreSQL
        
        Args:
            start_date (str): Optional start date filter (YYYY-MM-DD)
            end_date (str): Optional end date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with order and order item information
        """
        try:
            connection = self._connect_db()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Build query with optional date filters
            query = """
            SELECT 
                o.order_id,
                o.user_id,
                o.created_at as order_date,
                o.total_amount,
                o.status,
                oi.order_item_id,
                oi.product_id,
                p.name as product_name,
                p.category_id,
                c.name as category_name,
                oi.quantity,
                oi.price as unit_price,
                (oi.quantity * oi.price) as item_total
            FROM 
                orders o
            JOIN 
                order_items oi ON o.order_id = oi.order_id
            JOIN 
                products p ON oi.product_id = p.product_id
            LEFT JOIN 
                categories c ON p.category_id = c.category_id
            WHERE 
                1=1
            """
            
            params = []
            
            if start_date:
                query += " AND o.created_at >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND o.created_at <= %s"
                params.append(end_date)
            
            query += """
            ORDER BY 
                o.order_id, oi.order_item_id
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(rows)
            
            if df.empty:
                logger.warning("No transaction data found")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} order items from {df['order_id'].nunique()} orders")
            return df
            
        except Exception as e:
            logger.error(f"Error getting transaction data: {e}")
            raise
    
    def get_user_item_matrix(self):
        """
        Create a user-item matrix for collaborative filtering
        
        Returns:
            DataFrame where rows are users, columns are products, and values are interactions (e.g., quantity purchased)
        """
        try:
            connection = self._connect_db()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Query to get user-product interactions
            query = """
            SELECT 
                o.user_id, 
                oi.product_id,
                SUM(oi.quantity) as total_quantity
            FROM 
                order_items oi
            JOIN 
                orders o ON oi.order_id = o.order_id
            WHERE 
                o.status = 'COMPLETED'
            GROUP BY 
                o.user_id, oi.product_id
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            connection.close()
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(rows)
            
            if df.empty:
                logger.warning("No user-item interaction data found")
                return pd.DataFrame()
            
            # Create a pivot table (user-item matrix)
            user_item_matrix = df.pivot_table(
                index='user_id',
                columns='product_id',
                values='total_quantity',
                aggfunc='sum',
                fill_value=0
            )
            
            logger.info(f"Created user-item matrix with {user_item_matrix.shape[0]} users and {user_item_matrix.shape[1]} products")
            return user_item_matrix
            
        except Exception as e:
            logger.error(f"Error creating user-item matrix: {e}")
            raise
    
    def scale_features(self, df, columns, method='standard'):
        """
        Scale numerical features in a DataFrame
        
        Args:
            df (DataFrame): Input DataFrame
            columns (list): List of column names to scale
            method (str): Scaling method ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        scaled_df = df.copy()
        
        # Select only numerical columns that exist in the DataFrame
        valid_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not valid_cols:
            logger.warning("No valid numerical columns to scale")
            return scaled_df
        
        # Select the scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}, using standard scaling")
            scaler = StandardScaler()
        
        # Scale the features
        scaled_values = scaler.fit_transform(scaled_df[valid_cols])
        
        # Update the DataFrame with scaled values
        for i, col in enumerate(valid_cols):
            scaled_df[f"{col}_scaled"] = scaled_values[:, i]
        
        logger.info(f"Scaled {len(valid_cols)} features using {method} scaling")
        return scaled_df
    
    def clean_product_data(self, df):
        """
        Clean and preprocess product data
        
        Args:
            df (DataFrame): Product DataFrame
            
        Returns:
            DataFrame with cleaned product data
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Fill missing values
        if 'description' in cleaned_df.columns:
            cleaned_df['description'] = cleaned_df['description'].fillna('')
        
        # Numerical columns to clean
        numerical_cols = ['price', 'order_count', 'total_quantity_sold']
        for col in numerical_cols:
            if col in cleaned_df.columns:
                # Replace NaN with 0 for numerical columns
                cleaned_df[col] = cleaned_df[col].fillna(0)
        
        # Calculate popularity score based on order count and quantity sold
        if 'order_count' in cleaned_df.columns and 'total_quantity_sold' in cleaned_df.columns:
            cleaned_df['popularity_score'] = (
                (cleaned_df['order_count'] / cleaned_df['order_count'].max() if cleaned_df['order_count'].max() > 0 else 0) * 0.5 +
                (cleaned_df['total_quantity_sold'] / cleaned_df['total_quantity_sold'].max() if cleaned_df['total_quantity_sold'].max() > 0 else 0) * 0.5
            )
        
        logger.info(f"Cleaned product data with {len(cleaned_df)} records")
        return cleaned_df
    
    def clean_user_data(self, df):
        """
        Clean and preprocess user data
        
        Args:
            df (DataFrame): User DataFrame
            
        Returns:
            DataFrame with cleaned user data
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Convert date columns to datetime
        date_cols = ['created_at', 'last_order_date']
        for col in date_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
        
        # Fill missing values for numerical columns
        numerical_cols = ['order_count', 'total_spent']
        for col in numerical_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(0)
        
        # Calculate days since last order
        if 'last_order_date' in cleaned_df.columns:
            cleaned_df['days_since_last_order'] = (pd.Timestamp.now() - cleaned_df['last_order_date']).dt.days
            # Fill missing values (for users with no orders)
            cleaned_df['days_since_last_order'] = cleaned_df['days_since_last_order'].fillna(999999)
        
        # Calculate days since registration
        if 'created_at' in cleaned_df.columns:
            cleaned_df['days_since_registration'] = (pd.Timestamp.now() - cleaned_df['created_at']).dt.days
        
        logger.info(f"Cleaned user data with {len(cleaned_df)} records")
        return cleaned_df
    
    def prepare_data_for_models(self):
        """
        Prepare all necessary data for recommendation models
        
        Returns:
            dict with various preprocessed DataFrames
        """
        # Get and preprocess data
        user_data = self.get_user_data()
        product_data = self.get_product_data()
        transaction_data = self.get_transaction_data()
        user_item_matrix = self.get_user_item_matrix()
        
        # Clean and transform data
        cleaned_user_data = self.clean_user_data(user_data)
        cleaned_product_data = self.clean_product_data(product_data)
        
        # Scale relevant features
        if not cleaned_product_data.empty:
            scaled_product_data = self.scale_features(
                cleaned_product_data, 
                ['price', 'order_count', 'total_quantity_sold'], 
                method='minmax'
            )
        else:
            scaled_product_data = cleaned_product_data
        
        if not cleaned_user_data.empty:
            scaled_user_data = self.scale_features(
                cleaned_user_data, 
                ['order_count', 'total_spent', 'days_since_last_order', 'days_since_registration'], 
                method='minmax'
            )
        else:
            scaled_user_data = cleaned_user_data
        
        # Return all preprocessed data
        return {
            'user_data': scaled_user_data,
            'product_data': scaled_product_data,
            'transaction_data': transaction_data,
            'user_item_matrix': user_item_matrix
        } 