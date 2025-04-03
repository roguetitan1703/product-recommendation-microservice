import os
import json
import logging
import redis
from .apriori_model import AprioriModel
from .collaborative_filtering import CollaborativeFilteringModel
from .data_preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating and managing recommendation models
    """
    
    def __init__(self, db_config, redis_config):
        """
        Initialize the model factory with configuration
        
        Args:
            db_config (dict): Database configuration with host, port, dbname, user, password
            redis_config (dict): Redis configuration with host, port, db
        """
        self.db_config = db_config
        self.redis_config = redis_config
        self.redis_client = None
        self.models = {}
        
        # Initialize Redis connection
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_config.get('host', 'localhost'),
                port=self.redis_config.get('port', 6379),
                db=self.redis_config.get('db', 0),
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get_data_preprocessor(self):
        """
        Get a data preprocessor instance
        
        Returns:
            DataPreprocessor: A data preprocessor for preparing data for recommendation models
        """
        return DataPreprocessor(
            db_host=self.db_config.get('host', 'localhost'),
            db_port=self.db_config.get('port', 5432),
            db_name=self.db_config.get('dbname', 'quickcommerce'),
            db_user=self.db_config.get('user', 'postgres'),
            db_password=self.db_config.get('password', '')
        )
    
    def get_apriori_model(self):
        """
        Get an Apriori model instance
        
        Returns:
            AprioriModel: An Apriori model for product recommendations
        """
        if 'apriori' not in self.models:
            self.models['apriori'] = AprioriModel(
                db_host=self.db_config.get('host', 'localhost'),
                db_port=self.db_config.get('port', 5432),
                db_name=self.db_config.get('dbname', 'quickcommerce'),
                db_user=self.db_config.get('user', 'postgres'),
                db_password=self.db_config.get('password', ''),
                redis_client=self.redis_client
            )
        
        return self.models['apriori']
    
    def get_collaborative_filtering_model(self):
        """
        Get a collaborative filtering model instance
        
        Returns:
            CollaborativeFilteringModel: A collaborative filtering model for recommendations
        """
        if 'collaborative' not in self.models:
            self.models['collaborative'] = CollaborativeFilteringModel(
                db_host=self.db_config.get('host', 'localhost'),
                db_port=self.db_config.get('port', 5432),
                db_name=self.db_config.get('dbname', 'quickcommerce'),
                db_user=self.db_config.get('user', 'postgres'),
                db_password=self.db_config.get('password', ''),
                redis_client=self.redis_client
            )
        
        return self.models['collaborative']
    
    def train_all_models(self):
        """
        Train all recommendation models
        
        Returns:
            dict: Results of training each model
        """
        results = {}
        
        # Train Apriori model
        try:
            apriori_model = self.get_apriori_model()
            apriori_results = apriori_model.generate_rules()
            results['apriori'] = apriori_results
            logger.info(f"Apriori model trained: {apriori_results}")
        except Exception as e:
            logger.error(f"Error training Apriori model: {e}")
            results['apriori'] = {'status': 'error', 'message': str(e)}
        
        # Train collaborative filtering model
        try:
            cf_model = self.get_collaborative_filtering_model()
            cf_results = cf_model.train_model()
            results['collaborative'] = cf_results
            logger.info(f"Collaborative filtering model trained: {cf_results}")
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
            results['collaborative'] = {'status': 'error', 'message': str(e)}
        
        return results
    
    def get_product_recommendations(self, product_id, limit=5):
        """
        Get product recommendations for a given product
        
        Args:
            product_id (int): The product ID to get recommendations for
            limit (int): Maximum number of recommendations to return
            
        Returns:
            dict: Recommendations from different models
        """
        recommendations = {}
        
        # Get recommendations from Apriori model
        try:
            apriori_model = self.get_apriori_model()
            apriori_recs = apriori_model.get_recommendations(product_id, limit=limit)
            recommendations['apriori'] = apriori_recs
        except Exception as e:
            logger.error(f"Error getting Apriori recommendations: {e}")
            recommendations['apriori'] = []
        
        # Get similar items from collaborative filtering model
        try:
            cf_model = self.get_collaborative_filtering_model()
            cf_recs = cf_model.get_similar_items(product_id, limit=limit)
            recommendations['collaborative'] = cf_recs
        except Exception as e:
            logger.error(f"Error getting collaborative filtering recommendations: {e}")
            recommendations['collaborative'] = []
        
        return recommendations
    
    def get_user_recommendations(self, user_id, limit=10):
        """
        Get product recommendations for a user
        
        Args:
            user_id (int): The user ID to get recommendations for
            limit (int): Maximum number of recommendations to return
            
        Returns:
            dict: Recommendations from different models
        """
        recommendations = {}
        
        # Get recommendations from collaborative filtering model
        try:
            cf_model = self.get_collaborative_filtering_model()
            cf_recs = cf_model.get_item_recommendations(user_id, limit=limit)
            recommendations['collaborative'] = cf_recs
        except Exception as e:
            logger.error(f"Error getting collaborative filtering recommendations: {e}")
            recommendations['collaborative'] = []
        
        return recommendations
    
    def clear_all_models(self):
        """
        Clear all model data from Redis
        
        Returns:
            dict: Results of clearing each model
        """
        results = {}
        
        # Clear Apriori model
        try:
            apriori_model = self.get_apriori_model()
            apriori_model.clear_rules()
            results['apriori'] = {'status': 'success'}
            logger.info("Apriori model cleared")
        except Exception as e:
            logger.error(f"Error clearing Apriori model: {e}")
            results['apriori'] = {'status': 'error', 'message': str(e)}
        
        # Clear collaborative filtering model
        try:
            cf_model = self.get_collaborative_filtering_model()
            cf_model.clear_model_data()
            results['collaborative'] = {'status': 'success'}
            logger.info("Collaborative filtering model cleared")
        except Exception as e:
            logger.error(f"Error clearing collaborative filtering model: {e}")
            results['collaborative'] = {'status': 'error', 'message': str(e)}
        
        return results 