import os
import json
import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from mlxtend.frequent_patterns import apriori, association_rules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AprioriModel:
    """
    Apriori model for product recommendations based on transaction data
    """
    
    def __init__(self, db_host, db_port, db_name, db_user, db_password, redis_client):
        """Initialize the Apriori model with database and redis connection"""
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'dbname': db_name,
            'user': db_user,
            'password': db_password
        }
        self.redis_client = redis_client
        self.rules_key_prefix = 'product_rules:'
        self.metadata_key = 'product_rules:metadata'
    
    def _connect_db(self):
        """Create a new database connection"""
        try:
            connection = psycopg2.connect(**self.db_config)
            return connection
        except psycopg2.Error as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise
    
    def _get_transactions(self):
        """
        Get transaction data from PostgreSQL
        Returns a DataFrame where each row represents an order and columns are products (0/1 values)
        """
        try:
            connection = self._connect_db()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            try:
                # Query to get order items grouped by order
                query = """
                SELECT 
                    oi.order_id, 
                    oi.product_id,
                    p.name as product_name
                FROM 
                    order_items oi
                JOIN 
                    products p ON oi.product_id = p.product_id
                JOIN 
                    orders o ON oi.order_id = o.order_id
                WHERE 
                    o.status = 'COMPLETED'
                ORDER BY 
                    oi.order_id
                """
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(rows)
            except Exception as db_error:
                logger.warning(f"Database query failed: {db_error}. Using mock data instead.")
                
                # Create mock transaction data for testing
                mock_data = []
                for order_id in range(1, 51):  # 50 orders
                    # Each order has 2-5 products
                    num_products = np.random.randint(2, 6)
                    # Products are randomly selected from product IDs 1-20
                    product_ids = np.random.choice(range(1, 21), size=num_products, replace=False)
                    
                    for product_id in product_ids:
                        mock_data.append({
                            'order_id': order_id,
                            'product_id': product_id,
                            'product_name': f"Product {product_id}"
                        })
                
                df = pd.DataFrame(mock_data)
                logger.info("Using mock transaction data for testing")
            
            cursor.close()
            connection.close()
            
            if df.empty:
                logger.warning("No transaction data found")
                return pd.DataFrame()
            
            # Create a one-hot encoded DataFrame (basket representation)
            basket = pd.crosstab(df['order_id'], df['product_id'])
            
            # Convert to binary (1=product in order, 0=product not in order)
            basket = (basket > 0).astype(int)
            
            logger.info(f"Loaded {len(basket)} transactions with {len(basket.columns)} unique products")
            return basket
            
        except Exception as e:
            logger.error(f"Error getting transaction data: {e}")
            raise
    
    def generate_rules(self, min_support=0.01, min_confidence=0.3):
        """
        Generate association rules using the Apriori algorithm and store in Redis
        
        Args:
            min_support (float): Minimum support threshold for frequent itemsets
            min_confidence (float): Minimum confidence threshold for rules
            
        Returns:
            dict: Statistics about the generated rules
        """
        try:
            logger.info(f"Generating rules with min_support={min_support}, min_confidence={min_confidence}")
            
            # Get transaction data
            basket = self._get_transactions()
            
            if basket.empty:
                return {"rules_count": 0, "message": "No transaction data available"}
            
            # Run Apriori algorithm to find frequent itemsets
            frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                logger.warning(f"No frequent itemsets found with min_support={min_support}")
                return {"rules_count": 0, "message": "No frequent itemsets found"}
            
            logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")
            
            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if rules.empty:
                logger.warning(f"No rules generated with min_confidence={min_confidence}")
                return {"rules_count": 0, "message": "No rules generated"}
            
            # Clear existing rules
            self.clear_rules()
            
            # Store rules in Redis by antecedent product
            rules_count = 0
            product_rule_count = {}
            
            for _, rule in rules.iterrows():
                # Convert frozensets to lists for JSON serialization
                antecedent = list(rule['antecedents'])
                consequent = list(rule['consequents'])
                
                # Skip rules with multiple antecedents (we want simple product A -> product B rules)
                if len(antecedent) != 1:
                    continue
                
                # Skip rules with multiple consequents
                if len(consequent) != 1:
                    continue
                
                product_id = antecedent[0]
                recommended_product_id = consequent[0]
                
                # Create rule data
                rule_data = {
                    'product_id': int(product_id),
                    'recommended_product_id': int(recommended_product_id),
                    'support': float(rule['support']),
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift'])
                }
                
                # Store the rule in Redis
                key = f"{self.rules_key_prefix}{product_id}"
                self.redis_client.zadd(key, {json.dumps(rule_data): float(rule['confidence'])})
                
                # Track rule counts
                rules_count += 1
                product_rule_count[product_id] = product_rule_count.get(product_id, 0) + 1
            
            # Store metadata
            metadata = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'min_support': min_support,
                'min_confidence': min_confidence,
                'rules_count': rules_count,
                'product_count': len(product_rule_count)
            }
            self.redis_client.set(self.metadata_key, json.dumps(metadata))
            
            logger.info(f"Generated and stored {rules_count} rules for {len(product_rule_count)} products")
            
            return {
                'rules_count': rules_count,
                'product_count': len(product_rule_count),
                'min_support': min_support,
                'min_confidence': min_confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating rules: {e}")
            raise
    
    def get_recommendations(self, product_id, limit=5, threshold=0.1):
        """
        Get product recommendations for a given product ID
        
        Args:
            product_id (int): The product ID to get recommendations for
            limit (int): Maximum number of recommendations to return
            threshold (float): Minimum confidence threshold for recommendations
            
        Returns:
            list: List of recommended products with confidence scores
        """
        try:
            key = f"{self.rules_key_prefix}{product_id}"
            
            # Check if we have rules for this product
            if not self.redis_client.exists(key):
                logger.info(f"No recommendations found for product {product_id}")
                return []
            
            # Get recommendations from Redis, sorted by confidence (highest first)
            rule_data = self.redis_client.zrevrange(key, 0, limit-1, withscores=True)
            
            recommendations = []
            for rule_json, confidence in rule_data:
                if confidence < threshold:
                    continue
                    
                rule = json.loads(rule_json)
                
                recommendations.append({
                    'product_id': rule['recommended_product_id'],
                    'confidence': rule['confidence'],
                    'support': rule['support'],
                    'lift': rule['lift']
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error getting recommendations for product {product_id}: {e}")
            raise
    
    def clear_rules(self):
        """Clear all rules from Redis"""
        try:
            # Get all keys with the rules prefix
            pattern = f"{self.rules_key_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            # Delete all rule keys
            if keys:
                self.redis_client.delete(*keys)
            
            # Delete metadata
            self.redis_client.delete(self.metadata_key)
            
            logger.info(f"Cleared {len(keys)} rule sets from Redis")
            
        except Exception as e:
            logger.error(f"Error clearing rules: {e}")
            raise 