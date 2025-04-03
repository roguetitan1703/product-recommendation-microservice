import os
import json
import logging
import redis
import threading
import time
import requests
from datetime import datetime
from flask import Flask, jsonify, request
import colorlog

# Try to import Flask-CORS, but don't fail if it's not available
try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False
    print("WARNING: flask_cors is not installed. CORS support is disabled.")

from dotenv import load_dotenv

# Import recommendation engine models
try:
    from recommendation_engine.model_factory import ModelFactory
    from recommendation_engine.apriori_model import AprioriModel
    from recommendation_engine.collaborative_filtering import CollaborativeFilteringModel
    models_available = True
except ImportError as e:
    models_available = False
    print(f"WARNING: Could not import recommendation models: {e}")
    print("Recommendation functionality will be limited.")

# Load environment variables
load_dotenv()

# Configure colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Initialize Flask app
app = Flask(__name__)
if cors_available:
    CORS(app)  # Enable CORS for all routes
    logger.info("CORS support enabled")
else:
    logger.warning("CORS support disabled")

# Initialize Redis connection (default to localhost if not specified)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_db = int(os.getenv('REDIS_DB', 0))

try:
    redis_client = redis.Redis(
        host=redis_host, 
        port=redis_port, 
        db=redis_db, 
        decode_responses=True
    )
    redis_client.ping()  # Test connection
    logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
    redis_available = True
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}")
    logger.warning("Recommendations will not be persisted")
    redis_client = None
    redis_available = False

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'dbname': os.getenv('DB_NAME', 'snapquickcommerce'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'root')
}

# Redis configuration
redis_config = {
    'host': redis_host,
    'port': redis_port,
    'db': redis_db
}

# Initialize model factory
if models_available:
    try:
        model_factory = ModelFactory(db_config, redis_config)
        # Get model instances
        apriori_model = model_factory.get_apriori_model()
        cf_model = model_factory.get_collaborative_filtering_model()
        logger.info("Recommendation models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        apriori_model = None
        cf_model = None
else:
    apriori_model = None
    cf_model = None
    logger.warning("Recommendation models not available")

# Configure model parameters
APRIORI_MIN_SUPPORT = float(os.getenv('APRIORI_MIN_SUPPORT', '0.01'))
APRIORI_MIN_CONFIDENCE = float(os.getenv('APRIORI_MIN_CONFIDENCE', '0.3'))

# Fetch products from API and create fake relations
def fetch_products_and_create_relations():
    try:
        products = []
        
        # Try to fetch products from API
        try:
            response = requests.get('http://localhost:8081/api/products', timeout=2)
            if response.status_code == 200:
                products = response.json()
                logger.info(f"Successfully fetched {len(products)} products from API")
        except Exception as api_error:
            logger.warning(f"Could not fetch products from API: {api_error}. Using mock products.")
            
            # Generate fake products if API is not accessible
            import random
            
            # List of fake product categories
            categories = ["Electronics", "Clothing", "Home & Kitchen", "Beauty", "Books", "Toys", "Food"]
            
            # Generate 20 fake products
            for i in range(1, 21):
                category = random.choice(categories)
                price = round(random.uniform(10, 500), 2)
                
                product = {
                    'id': i,
                    'name': f"{category} Product {i}",
                    'description': f"This is a description for product {i}",
                    'price': price,
                    'category': category,
                    'imageUrl': f"https://example.com/images/product{i}.jpg"
                }
                products.append(product)
            
            logger.info(f"Generated {len(products)} mock products")
        
        # Store products in Redis for future reference
        redis_client.set('product_list', json.dumps(products))
        
        # Create fake relation table (product-to-product recommendations)
        # For each product, recommend 3-5 random other products
        import random
        relations = {}
        product_ids = [p['id'] for p in products]
        
        for product in products:
            prod_id = product['id']
            # Exclude current product from potential recommendations
            potential_recommendations = [p for p in product_ids if p != prod_id]
            # Select 3-5 random products as recommendations
            num_recommendations = min(random.randint(3, 5), len(potential_recommendations))
            recommended_ids = random.sample(potential_recommendations, num_recommendations)
            
            # Create relation entries with random confidence scores
            recommendations = []
            for rec_id in recommended_ids:
                # Find the recommended product details
                rec_product = next((p for p in products if p['id'] == rec_id), None)
                if rec_product:
                    recommendations.append({
                        'product_id': rec_id,
                        'name': rec_product.get('name', ''),
                        'confidence': round(random.uniform(0.3, 0.9), 2),
                        'support': round(random.uniform(0.1, 0.3), 2),
                        'lift': round(random.uniform(1.0, 3.0), 2),
                        'price': rec_product.get('price', 0)
                    })
            
            # Store in Redis
            relations[str(prod_id)] = recommendations
            redis_client.set(f"product_rules:{prod_id}", json.dumps(recommendations))
        
        # Store the complete relation mapping
        redis_client.set('product_relations', json.dumps(relations))
        
        # Store last update time in Redis
        redis_client.set('recommendation_rules:last_update', datetime.now().isoformat())
        
        logger.info(f"Created fake relations for {len(relations)} products and stored in Redis")
        return True
    except Exception as e:
        logger.error(f"Error creating fake relations: {e}")
        return False

# Generate mock order transactions for Apriori algorithm
def generate_mock_order_transactions():
    try:
        # Get product list from Redis
        if not redis_client.exists('product_list'):
            logger.error("Product list not found in Redis")
            return False
            
        products = json.loads(redis_client.get('product_list'))
        product_ids = [p['id'] for p in products]
        
        # Generate 100 mock orders
        import random
        import uuid
        from datetime import datetime, timedelta
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Generate orders over the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Create mock orders
        orders = []
        order_items = []
        
        for order_id in range(1, 101):  # 100 orders
            # Generate order date (random date between start_date and end_date)
            days_ago = random.randint(0, 30)
            order_date = end_date - timedelta(days=days_ago)
            
            # Create order
            order = {
                'order_id': str(uuid.uuid4()),
                'user_id': random.randint(1, 50),  # 50 users
                'order_date': order_date.isoformat(),
                'status': 'COMPLETED'
            }
            orders.append(order)
            
            # Each order has 2-5 items
            num_items = random.randint(2, 5)
            order_products = random.sample(product_ids, num_items)
            
            for product_id in order_products:
                order_item = {
                    'order_item_id': str(uuid.uuid4()),
                    'order_id': order['order_id'],
                    'product_id': product_id,
                    'quantity': random.randint(1, 3)
                }
                order_items.append(order_item)
        
        # Store in Redis
        redis_client.set('mock_orders', json.dumps(orders))
        redis_client.set('mock_order_items', json.dumps(order_items))
        
        # Create a transaction table for Apriori (order_id -> [product_ids])
        transactions = {}
        for order in orders:
            order_id = order['order_id']
            # Get all products in this order
            items = [item['product_id'] for item in order_items if item['order_id'] == order_id]
            transactions[order_id] = items
        
        # Store transactions in Redis
        redis_client.set('apriori_transactions', json.dumps(transactions))
        
        logger.info(f"Generated {len(orders)} mock orders with {len(order_items)} order items")
        logger.info(f"Created {len(transactions)} transactions for Apriori algorithm")
        
        return True
    except Exception as e:
        logger.error(f"Error generating mock order transactions: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Get last update time from Redis
    last_update = None
    if redis_client:
        try:
            last_update = redis_client.get('recommendation_rules:last_update')
        except Exception:
            pass
    
    return jsonify({
        "status": "healthy" if models_available and apriori_model and cf_model else "degraded", 
        "service": "recommendation-api",
        "available_models": ["apriori", "collaborative_filtering"] if models_available and apriori_model and cf_model else [],
        "database": "connected" if models_available and apriori_model and cf_model else "disconnected",
        "redis": "connected" if redis_available else "disconnected",
        "last_update": last_update
    })

@app.route('/recommendations/product/<int:product_id>', methods=['GET'])
def get_product_recommendations(product_id):
    """Get product recommendations based on product_id"""
    try:
        if not models_available or not apriori_model:
            # If models are not available, try to use the fake relations from Redis
            if redis_available:
                key = f"product_rules:{product_id}"
                if redis_client.exists(key):
                    recommendations = json.loads(redis_client.get(key))
                    return jsonify({
                        "product_id": product_id,
                        "model": "fake_relations",
                        "recommendations": recommendations
                    })
            
            return jsonify({"error": "Recommendation models not available"}), 503
            
        # Get optional query parameters
        limit = request.args.get('limit', default=5, type=int)
        threshold = request.args.get('threshold', default=0.1, type=float)
        model_type = request.args.get('model', default='apriori')
        
        # Choose model based on parameter
        if model_type == 'collaborative_filtering' and cf_model:
            recommendations = cf_model.get_similar_items(product_id, limit)
        else:
            recommendations = apriori_model.get_recommendations(product_id, limit, threshold)
        
        return jsonify({
            "product_id": product_id,
            "model": model_type,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting recommendations for product {product_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations/user/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    """Get product recommendations for a user"""
    try:
        if not models_available or not cf_model:
            return jsonify({"error": "Collaborative filtering model not available"}), 503
            
        # Get optional query parameters
        limit = request.args.get('limit', default=10, type=int)
        include_rated = request.args.get('include_rated', default='false').lower() == 'true'
        
        # Get recommendations from collaborative filtering model
        recommendations = cf_model.get_item_recommendations(user_id, limit, include_rated=include_rated)
        
        return jsonify({
            "user_id": user_id,
            "model": "collaborative_filtering",
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations/generate', methods=['POST'])
def generate_recommendations():
    """Generate recommendations by running algorithms"""
    try:
        if not models_available or not apriori_model or not cf_model:
            return jsonify({"error": "Recommendation models not available"}), 503
            
        # Get parameters from request
        data = request.get_json(silent=True) or {}
        min_support = data.get('min_support', APRIORI_MIN_SUPPORT)
        min_confidence = data.get('min_confidence', APRIORI_MIN_CONFIDENCE)
        model_type = data.get('model', 'all')
        
        results = {}
        
        # Generate rules based on model type
        if model_type in ['all', 'apriori']:
            try:
                apriori_result = apriori_model.generate_rules(min_support, min_confidence)
                results['apriori'] = apriori_result
                logger.info(f"Apriori model trained: {apriori_result}")
            except Exception as e:
                logger.error(f"Error training Apriori model: {e}")
                results['apriori'] = {'status': 'error', 'message': str(e)}
        
        if model_type in ['all', 'collaborative_filtering']:
            try:
                cf_result = cf_model.train_model(min_interactions=1)
                results['collaborative_filtering'] = cf_result
                logger.info(f"Collaborative filtering model trained: {cf_result}")
            except Exception as e:
                logger.error(f"Error training collaborative filtering model: {e}")
                results['collaborative_filtering'] = {'status': 'error', 'message': str(e)}
        
        # Store last update time in Redis
        if redis_client:
            try:
                redis_client.set('recommendation_rules:last_update', datetime.now().isoformat())
            except Exception as e:
                logger.error(f"Error updating Redis: {e}")
        
        return jsonify({
            "status": "success",
            "message": "Recommendation rules generated successfully",
            "results": results
        })
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations/clear', methods=['POST'])
def clear_recommendations():
    """Clear all recommendation data from Redis"""
    try:
        if not models_available or not apriori_model or not cf_model:
            return jsonify({"error": "Recommendation models not available"}), 503
            
        data = request.get_json(silent=True) or {}
        model_type = data.get('model', 'all')
        
        if model_type in ['all', 'apriori']:
            apriori_model.clear_rules()
        
        if model_type in ['all', 'collaborative_filtering']:
            cf_model.clear_model_data()
        
        return jsonify({
            "status": "success",
            "message": f"Recommendation rules cleared for model(s): {model_type}"
        })
    except Exception as e:
        logger.error(f"Error clearing recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/relations/generate', methods=['POST'])
def generate_fake_relations():
    """Generate fake product relations from the API data"""
    try:
        if not redis_available:
            return jsonify({"error": "Redis not available"}), 503
            
        success = fetch_products_and_create_relations()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Fake product relations generated successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to generate fake relations"
            }), 500
    except Exception as e:
        logger.error(f"Error generating fake relations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/relations/product/<int:product_id>', methods=['GET'])
def get_product_relations(product_id):
    """Get fake relations for a product"""
    try:
        if not redis_available:
            return jsonify({"error": "Redis not available"}), 503
            
        key = f"product_rules:{product_id}"
        if not redis_client.exists(key):
            return jsonify({
                "product_id": product_id,
                "recommendations": []
            })
            
        recommendations = json.loads(redis_client.get(key))
        
        return jsonify({
            "product_id": product_id,
            "model": "fake_relations",
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting fake relations for product {product_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/transactions/generate', methods=['POST'])
def generate_transactions():
    """Generate mock order transactions for Apriori algorithm"""
    try:
        if not redis_available:
            return jsonify({"error": "Redis not available"}), 503
            
        success = generate_mock_order_transactions()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Mock order transactions generated successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to generate mock order transactions"
            }), 500
    except Exception as e:
        logger.error(f"Error generating mock order transactions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/transactions', methods=['GET'])
def get_transactions():
    """Get mock order transactions"""
    try:
        if not redis_available:
            return jsonify({"error": "Redis not available"}), 503
            
        if not redis_client.exists('apriori_transactions'):
            return jsonify({
                "transactions": []
            })
            
        transactions = json.loads(redis_client.get('apriori_transactions'))
        
        return jsonify({
            "transactions": transactions
        })
    except Exception as e:
        logger.error(f"Error getting transactions: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app on port 5000")
    
    # Try to generate fake relations on startup if Redis is available
    if redis_available:
        try:
            fetch_products_and_create_relations()
            generate_mock_order_transactions()
        except Exception as e:
            logger.warning(f"Could not generate fake data on startup: {e}")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 