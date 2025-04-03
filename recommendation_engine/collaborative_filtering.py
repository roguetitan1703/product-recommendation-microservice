import os
import json
import logging
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollaborativeFilteringModel:
    """
    Collaborative filtering model for product recommendations based on user ratings
    and purchase history
    """
    
    def __init__(self, db_host, db_port, db_name, db_user, db_password, redis_client):
        """Initialize the collaborative filtering model with database and redis connection"""
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'dbname': db_name,
            'user': db_user,
            'password': db_password
        }
        self.redis_client = redis_client
        self.user_similarity_key_prefix = 'user_similarity:'
        self.item_similarity_key_prefix = 'item_similarity:'
        self.user_product_ratings_key = 'user_product_ratings'
        self.product_info_key = 'product_info'
        self.metadata_key = 'cf_model:metadata'
        
        # In-memory cache for similarity matrices
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.user_index_map = None
        self.item_index_map = None
        self.reverse_user_index = None
        self.reverse_item_index = None
    
    def _connect_db(self):
        """Create a new database connection"""
        try:
            connection = psycopg2.connect(**self.db_config)
            return connection
        except psycopg2.Error as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise
    
    def _get_interaction_data(self):
        """
        Get user-item interaction data from PostgreSQL
        Returns a DataFrame with columns [user_id, product_id, rating]
        """
        try:
            connection = self._connect_db()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            interactions_df = pd.DataFrame()
            product_df = pd.DataFrame()
            
            try:
                # Query to get order items and product ratings
                query = """
                -- Order-based interactions (purchase = implicit rating of 1.0)
                SELECT 
                    o.user_id, 
                    oi.product_id,
                    1.0 as rating,
                    o.created_at as interaction_date
                FROM 
                    order_items oi
                JOIN 
                    orders o ON oi.order_id = o.order_id
                WHERE 
                    o.status = 'COMPLETED'
                    
                UNION ALL
                
                -- Explicit ratings (if available)
                SELECT 
                    r.user_id,
                    r.product_id,
                    r.rating,
                    r.created_at as interaction_date
                FROM 
                    product_ratings r
                
                ORDER BY 
                    user_id, product_id
                """
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                # Also get product information
                product_query = """
                SELECT 
                    product_id,
                    name,
                    category_id,
                    price
                FROM 
                    products
                """
                
                cursor.execute(product_query)
                product_rows = cursor.fetchall()
                
                # Convert to pandas DataFrame
                interactions_df = pd.DataFrame(rows)
                product_df = pd.DataFrame(product_rows)
                
            except Exception as db_error:
                logger.warning(f"Database query failed: {db_error}. Using mock data instead.")
                
                # Create mock interaction data for testing
                mock_interactions = []
                
                # Create 50 users
                user_ids = range(1, 51)
                # 20 products (same as in apriori model)
                product_ids = range(1, 21)
                
                # Generate random interactions
                np.random.seed(42)  # For reproducibility
                
                # Each user interacts with 3-10 products
                for user_id in user_ids:
                    # Number of products this user has interacted with
                    n_interactions = np.random.randint(3, 11)
                    # Sample random products for this user
                    user_products = np.random.choice(product_ids, size=n_interactions, replace=False)
                    
                    for product_id in user_products:
                        # Generate a rating between 3 and 5
                        rating = np.random.uniform(3, 5)
                        # Add to interactions
                        mock_interactions.append({
                            'user_id': user_id,
                            'product_id': product_id,
                            'rating': rating,
                            'interaction_date': pd.Timestamp.now()
                        })
                
                # Create mock product data
                mock_products = []
                for product_id in product_ids:
                    mock_products.append({
                        'product_id': product_id,
                        'name': f'Product {product_id}',
                        'category_id': np.random.randint(1, 6),  # 5 categories
                        'price': np.random.uniform(10, 100)  # Random price between $10-$100
                    })
                
                interactions_df = pd.DataFrame(mock_interactions)
                product_df = pd.DataFrame(mock_products)
                
                logger.info("Using mock interaction data for testing")
            
            cursor.close()
            connection.close()
            
            if interactions_df.empty:
                logger.warning("No interaction data found")
                return pd.DataFrame(), pd.DataFrame()
            
            # Aggregate interactions (take the max rating if multiple interactions)
            interactions_df = interactions_df.groupby(['user_id', 'product_id']).agg({'rating': 'max'}).reset_index()
            
            logger.info(f"Loaded {len(interactions_df)} user-product interactions with {interactions_df['user_id'].nunique()} users and {interactions_df['product_id'].nunique()} products")
            
            return interactions_df, product_df
            
        except Exception as e:
            logger.error(f"Error getting interaction data: {e}")
            raise
    
    def _create_user_item_matrix(self, interactions_df):
        """
        Create a user-item matrix from the interactions DataFrame
        
        Args:
            interactions_df (DataFrame): DataFrame with columns [user_id, product_id, rating]
            
        Returns:
            tuple: (user_item_matrix, user_index_map, item_index_map)
        """
        # Create mappings from user_id/product_id to matrix indices
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        user_index_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        item_index_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        # Create reverse mappings (matrix index to user_id/product_id)
        reverse_user_index = {idx: user_id for user_id, idx in user_index_map.items()}
        reverse_item_index = {idx: item_id for item_id, idx in item_index_map.items()}
        
        # Create matrix indices
        user_indices = interactions_df['user_id'].map(user_index_map).values
        item_indices = interactions_df['product_id'].map(item_index_map).values
        ratings = interactions_df['rating'].values
        
        # Create sparse matrix
        user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                                    shape=(len(unique_users), len(unique_items)))
        
        return (user_item_matrix, user_index_map, item_index_map, 
                reverse_user_index, reverse_item_index)
    
    def _calculate_similarity_matrices(self, user_item_matrix):
        """
        Calculate user-user and item-item similarity matrices
        
        Args:
            user_item_matrix (csr_matrix): Sparse user-item matrix
            
        Returns:
            tuple: (user_similarity_matrix, item_similarity_matrix)
        """
        # Calculate item-item similarity matrix
        # Transpose the user-item matrix to get item-user matrix
        item_user_matrix = user_item_matrix.T.tocsr()
        
        # Calculate cosine similarity between items
        item_similarity_matrix = cosine_similarity(item_user_matrix, dense_output=False)
        
        # Calculate user-user similarity matrix
        user_similarity_matrix = cosine_similarity(user_item_matrix, dense_output=False)
        
        return user_similarity_matrix, item_similarity_matrix
    
    def train_model(self, min_interactions=1):
        """
        Train the collaborative filtering model
        
        Args:
            min_interactions (int): Minimum number of interactions for a user to be included
            
        Returns:
            dict: Statistics about the trained model
        """
        try:
            logger.info("Training collaborative filtering model")
            
            # Get interaction data
            interactions_df, product_df = self._get_interaction_data()
            
            if interactions_df.empty:
                return {"status": "error", "message": "No interaction data available"}
            
            # Filter users with sufficient interactions
            user_counts = interactions_df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_interactions].index
            filtered_df = interactions_df[interactions_df['user_id'].isin(valid_users)]
            
            logger.info(f"Filtered to {len(filtered_df)} interactions with {len(valid_users)} users having at least {min_interactions} interactions")
            
            # Create user-item matrix and mappings
            (self.user_item_matrix, 
             self.user_index_map, 
             self.item_index_map,
             self.reverse_user_index,
             self.reverse_item_index) = self._create_user_item_matrix(filtered_df)
            
            # Calculate similarity matrices
            self.user_similarity_matrix, self.item_similarity_matrix = self._calculate_similarity_matrices(self.user_item_matrix)
            
            # Store data in Redis
            self._store_model_data(filtered_df, product_df)
            
            # Store metadata
            metadata = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'n_users': len(self.user_index_map),
                'n_items': len(self.item_index_map),
                'n_interactions': len(filtered_df),
                'min_interactions': min_interactions
            }
            self.redis_client.set(self.metadata_key, json.dumps(metadata))
            
            logger.info(f"Model trained with {len(self.user_index_map)} users and {len(self.item_index_map)} items")
            
            return {
                'status': 'success',
                'n_users': len(self.user_index_map),
                'n_items': len(self.item_index_map),
                'n_interactions': len(filtered_df)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _store_model_data(self, interactions_df, product_df):
        """
        Store model data in Redis
        
        Args:
            interactions_df (DataFrame): DataFrame with user-item interactions
            product_df (DataFrame): DataFrame with product information
        """
        # Store user-product ratings
        ratings_data = {}
        for user_id, group in interactions_df.groupby('user_id'):
            user_ratings = {}
            for _, row in group.iterrows():
                user_ratings[int(row['product_id'])] = float(row['rating'])
            ratings_data[int(user_id)] = user_ratings
        
        self.redis_client.set(self.user_product_ratings_key, json.dumps(ratings_data))
        
        # Store product info
        product_info = {}
        for _, row in product_df.iterrows():
            product_info[int(row['product_id'])] = {
                'name': row['name'],
                'category_id': int(row['category_id']) if row['category_id'] is not None else None,
                'price': float(row['price'])
            }
        
        self.redis_client.set(self.product_info_key, json.dumps(product_info))
        
        # Store item similarity for each item
        for item_idx in range(self.item_similarity_matrix.shape[0]):
            item_id = self.reverse_item_index[item_idx]
            
            # Get similar items
            similar_items = []
            
            # Get the row from the similarity matrix
            sim_scores = self.item_similarity_matrix[item_idx].toarray().flatten()
            
            # Create (item_idx, similarity) tuples and sort by similarity
            item_scores = [(i, sim_scores[i]) for i in range(len(sim_scores)) if i != item_idx and sim_scores[i] > 0]
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Convert to original item IDs and store
            for similar_idx, score in item_scores[:50]:  # Store top 50 similar items
                similar_id = self.reverse_item_index[similar_idx]
                similar_items.append({
                    'product_id': int(similar_id),
                    'similarity': float(score)
                })
            
            # Store in Redis
            if similar_items:
                key = f"{self.item_similarity_key_prefix}{item_id}"
                self.redis_client.set(key, json.dumps(similar_items))
    
    def get_item_recommendations(self, user_id, limit=10, include_rated=False):
        """
        Get item recommendations for a user using item-based collaborative filtering
        
        Args:
            user_id (int): The user ID to get recommendations for
            limit (int): Maximum number of recommendations to return
            include_rated (bool): Whether to include items the user has already rated
            
        Returns:
            list: List of recommended product IDs with scores
        """
        try:
            # Check if ratings data exists in Redis
            if not self.redis_client.exists(self.user_product_ratings_key):
                logger.warning("No ratings data in Redis, cannot generate recommendations")
                return []
            
            # Get user ratings data
            ratings_data = json.loads(self.redis_client.get(self.user_product_ratings_key))
            
            # Convert user_id to string for JSON dict key lookup
            user_id_str = str(user_id)
            
            # Check if user exists in ratings data
            if user_id_str not in ratings_data:
                logger.info(f"User {user_id} not found in ratings data")
                return []
            
            user_ratings = ratings_data[user_id_str]
            
            # Get product info
            product_info = json.loads(self.redis_client.get(self.product_info_key))
            
            # Calculate recommendations
            candidate_items = {}
            
            # For each item the user has rated
            for product_id, rating in user_ratings.items():
                # Get similar items
                key = f"{self.item_similarity_key_prefix}{product_id}"
                if not self.redis_client.exists(key):
                    continue
                
                similar_items = json.loads(self.redis_client.get(key))
                
                # For each similar item
                for item in similar_items:
                    sim_product_id = item['product_id']
                    similarity = item['similarity']
                    
                    # Skip items the user has already rated if include_rated is False
                    if not include_rated and str(sim_product_id) in user_ratings:
                        continue
                    
                    # Calculate weighted rating
                    weighted_rating = similarity * rating
                    
                    # Update candidate items
                    if sim_product_id not in candidate_items:
                        candidate_items[sim_product_id] = {
                            'weighted_sum': weighted_rating,
                            'similarity_sum': similarity
                        }
                    else:
                        candidate_items[sim_product_id]['weighted_sum'] += weighted_rating
                        candidate_items[sim_product_id]['similarity_sum'] += similarity
            
            # Calculate final scores and sort
            recommendations = []
            for product_id, scores in candidate_items.items():
                product_id = int(product_id)
                
                # Calculate normalized score
                if scores['similarity_sum'] > 0:
                    normalized_score = scores['weighted_sum'] / scores['similarity_sum']
                else:
                    normalized_score = 0
                
                # Add to recommendations with product info
                if str(product_id) in product_info:
                    recommendations.append({
                        'product_id': product_id,
                        'score': float(normalized_score),
                        'name': product_info[str(product_id)]['name'],
                        'price': product_info[str(product_id)]['price'],
                        'category_id': product_info[str(product_id)].get('category_id')
                    })
            
            # Sort by score and limit
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {e}")
            raise
    
    def get_similar_items(self, product_id, limit=10):
        """
        Get similar items for a product using item-based collaborative filtering
        
        Args:
            product_id (int): The product ID to get similar items for
            limit (int): Maximum number of similar items to return
            
        Returns:
            list: List of similar product IDs with similarity scores
        """
        try:
            # Check if similarity data exists in Redis
            key = f"{self.item_similarity_key_prefix}{product_id}"
            if not self.redis_client.exists(key):
                logger.info(f"No similarity data found for product {product_id}")
                return []
            
            # Get similar items from Redis
            similar_items = json.loads(self.redis_client.get(key))
            
            # Get product info
            product_info = json.loads(self.redis_client.get(self.product_info_key))
            
            # Add product info to similar items
            results = []
            for item in similar_items[:limit]:
                item_id = item['product_id']
                if str(item_id) in product_info:
                    results.append({
                        'product_id': item_id,
                        'similarity': item['similarity'],
                        'name': product_info[str(item_id)]['name'],
                        'price': product_info[str(item_id)]['price'],
                        'category_id': product_info[str(item_id)].get('category_id')
                    })
                else:
                    results.append({
                        'product_id': item_id,
                        'similarity': item['similarity']
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar items for product {product_id}: {e}")
            raise
    
    def clear_model_data(self):
        """Clear all model data from Redis"""
        try:
            # Delete user-product ratings
            self.redis_client.delete(self.user_product_ratings_key)
            
            # Delete product info
            self.redis_client.delete(self.product_info_key)
            
            # Delete item similarity data
            pattern = f"{self.item_similarity_key_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            
            # Delete user similarity data
            pattern = f"{self.user_similarity_key_prefix}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            
            # Delete metadata
            self.redis_client.delete(self.metadata_key)
            
            logger.info("Cleared all collaborative filtering model data from Redis")
            
        except Exception as e:
            logger.error(f"Error clearing model data: {e}")
            raise 