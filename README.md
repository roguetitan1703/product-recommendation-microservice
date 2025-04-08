# Product Recommendation Microservice

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-Apriori-informational.svg)]()
[![Datastore](https://img.shields.io/badge/Datastore-HBase%20&%20Redis-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Replace -->

This microservice generates "frequently bought together" product recommendations for the **QuickCommerce** platform. It leverages the **Apriori algorithm** to mine association rules from historical transaction data.

## 🎯 Objective

To enhance user experience and increase sales by suggesting relevant products based on items currently being viewed or added to the cart.

## 🧠 Methodology: Association Rule Mining

*   **Algorithm:** Apriori.
*   **Goal:** Discover rules of the form `{Product A} => {Product B}`, indicating that customers who buy Product A also tend to buy Product B.
*   **Process:**
    1.  Identify frequent itemsets (sets of products often purchased together) in historical transactions.
    2.  Generate association rules from these frequent itemsets.
    3.  Filter rules based on metrics like *support*, *confidence*, and *lift* to ensure relevance and strength.
*   **Data Source:** Transactional data (order history) stored in Apache HBase.
*   **Caching:** Generated recommendations are cached in Redis for fast retrieval.

## 🛠 Technology Stack

| Category              | Technology / Library                            | Role                                              |
| :-------------------- | :---------------------------------------------- | :------------------------------------------------ |
| **Language**          | Python 3.x                                      | Core development language                         |
| **Algorithm Lib**     | [Specify: e.g., `mlxtend`, Custom Spark Impl.]  | Implementing Apriori                              |
| **API Framework**     | [Specify: e.g., Flask or FastAPI]               | Serving recommendations via REST API              |
| **Data Handling**     | Pandas, NumPy                                   | Data manipulation (in API/batch)                  |
| **Batch Processing**  | Apache Spark / PySpark                          | Running Apriori on large datasets (HBase -> Redis) |
| **Data Storage**      | Apache HBase                                    | Storing raw transaction data                      |
| **Caching**           | Redis                                           | Storing pre-calculated recommendations            |
| **HBase Client**      | [Specify: e.g., `happybase`]                    | Connecting to HBase (if needed in API/batch)      |
| **Redis Client**      | [Specify: e.g., `redis-py`]                     | Connecting to Redis                               |

*(**Note:** Fill in the `[Specify: ...]` placeholders.)*

## 🔄 Data & Processing Flow

1.  **Data Storage:** Customer orders (containing sets of `productIds`) are persisted into an **Apache HBase** table optimized for transaction lookups.
2.  **Batch Job (Periodic):** An **Apache Spark** application runs at regular intervals (e.g., every 20-30 minutes):
    *   Connects to HBase and reads recent transaction data.
    *   Executes the Apriori algorithm to mine association rules.
    *   Transforms strong rules into recommendation lists (e.g., `product_A -> [product_B, product_C]`).
    *   Connects to **Redis** and populates/updates the cache (e.g., `SET recommendation:product_A '["product_B", "product_C"]'`).
3.  **API Serving:** The Python API server (Flask/FastAPI):
    *   Receives requests for recommendations for a specific `productId`.
    *   Connects to **Redis** and performs a quick lookup using the `productId` as the key.
    *   Returns the cached list of recommended `productIds`.

> **Key:** The heavy computation (Apriori) is done offline in the batch job, while the API server provides fast, low-latency responses by reading from the cache.

## 🔌 API Endpoint

Provides recommendations based on a given product.

http
POST /api/v1/recommend
Content-Type: application/json

{
  "productId": "string" // The product ID to get recommendations for
  // "userId": "string" // Optional: For potential future personalization
}


**Success Response (Example):**

http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "productId": "prodXYZ",
  "recommended_products": [
    "prodABC",
    "prodDEF",
    "prodGHI"
  ]
}


## ⚙ Local Setup & Running (API Server)

> **Prerequisites:** Ensure **Redis** and **HBase** services are running and accessible from where you run this application. Connection details might need configuration via environment variables or config files.

1.  **Clone the repository:**
    bash
    git clone [URL_OF_THIS_RECOMMENDATION_REPO]
    cd product-recommendation-microservice
    
2.  **Create and Activate Python Virtual Environment:**
    bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate.bat # Windows CMD
    # venv\Scripts\Activate.ps1 # Windows PowerShell
    
3.  **Install Dependencies:**
    bash
    pip install -r requirements.txt
    
4.  **Configure Connections:**
    Set environment variables or update configuration files to point to your running Redis instance (and HBase if the API interacts directly, though typically it only needs Redis).
    bash
    export REDIS_HOST='localhost' # Example
    export REDIS_PORT=6379       # Example
    # set REDIS_HOST=localhost   # Windows example
    

5.  **Run the API Server:**
    *   **Using Flask:**
        bash
        export FLASK_APP=app.py # Linux/macOS
        # set FLASK_APP=app.py # Windows
        flask run --host=0.0.0.0 --port=5002 # Adjust port if needed
        
    *   **Using FastAPI (with Uvicorn):**
        bash
        uvicorn app:app --host 0.0.0.0 --port 5002 --reload
        
    > The API server will listen on the specified port (e.g., `http://localhost:5002`).

## ⏱ Batch Processing Job (Apriori on Spark)

> **Note:** This is a separate process responsible for generating the recommendations stored in Redis.

*   The Spark application (`batch_recommendations.py` or similar) needs to be executed independently.
*   It requires access to the **HBase** cluster (for input data) and the **Redis** instance (for outputting results).
*   This job should be scheduled to run periodically using tools like `cron`, Apache Airflow, or a dedicated job scheduler.
*   Execution typically involves submitting the PySpark script to a Spark cluster:
    bash
    # Example spark-submit command (details vary based on cluster setup)
    # spark-submit --master <your_spark_master> \
    #   --deploy-mode <client_or_cluster> \
    #   --packages <hbase_connector_package>,<redis_connector_package> \
    #   batch_recommendations.py \
    #   --hbase-table <transactions_table> \
    #   --redis-host <redis_host> --redis-port <redis_port>
    
*   Consult the specific scripts/documentation within a `batch/` or `spark_jobs/` directory for detailed instructions.
