"""
Simple mock recommendation server for testing the integration
"""

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "mock-recommendation-api",
        "available_models": ["mock"],
        "message": "This is a mock server for testing"
    })

@app.route('/recommendations/product/<int:product_id>', methods=['GET'])
def get_product_recommendations(product_id):
    """Mock product recommendations"""
    # Get optional query parameters
    limit = request.args.get('limit', default=5, type=int)
    
    # Generate mock recommendations
    mock_recommendations = []
    for i in range(1, min(limit+1, 6)):
        mock_recommendations.append({
            'product_id': product_id + i,
            'confidence': round(0.9 - (i * 0.1), 2)
        })
    
    return jsonify({
        "product_id": product_id,
        "model": "mock",
        "recommendations": mock_recommendations
    })

@app.route('/recommendations/user/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    """Mock user recommendations"""
    # Get optional query parameters
    limit = request.args.get('limit', default=10, type=int)
    
    # Generate mock recommendations
    mock_recommendations = []
    for i in range(1, min(limit+1, 11)):
        mock_recommendations.append({
            'product_id': 100 + i,
            'score': round(0.95 - (i * 0.05), 2),
            'name': f"Mock Product {100 + i}",
            'price': 10.0 + i
        })
    
    return jsonify({
        "user_id": user_id,
        "model": "mock",
        "recommendations": mock_recommendations
    })

@app.route('/recommendations/generate', methods=['POST'])
def generate_recommendations():
    """Mock recommendation generation"""
    return jsonify({
        "status": "success",
        "message": "Mock recommendation rules generated successfully",
        "results": {
            "mock": {
                "rules_count": 100,
                "product_count": 50
            }
        }
    })

@app.route('/recommendations/clear', methods=['POST'])
def clear_recommendations():
    """Mock clear recommendations"""
    return jsonify({
        "status": "success",
        "message": "Mock recommendation rules cleared successfully"
    })

if __name__ == '__main__':
    print("Starting mock recommendation server on port 5000...")
    print("This is a mock server for testing the integration without dependencies.")
    print("Press Ctrl+C to stop the server.")
    app.run(host='0.0.0.0', port=5000, debug=True) 