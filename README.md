README

Title: A tutorial to set safety stock under guaranteed-service time by dynamic programming

Overview:
This Python script implements an inventory management algorithm based on the concept of safety stock optimization. The algorithm aims to optimize inventory levels by considering factors such as demand variability, lead times, and holding costs.
There are 3 python scripts with data provided in the article, and python script to input your data.

Usage:
1. Input Data: The script requires input data regarding the relationships between different stages in the supply chain, delivery stages, and specific parameters for each stage such as costs, lead times, and maximum replenishment levels. This data can be provided via a CSV file following a specific format.

2. Execution: Once the input data is provided, execute the Python script. It will prompt for necessary inputs and then perform calculations to determine optimal inventory levels.

3. Output: The script generates two outputs:
   - Printed results showing optimal inventory levels, safety stock costs, and related metrics.
   - A CSV file named "results.csv" containing the detailed results.

Input Data Format:
- The script expects input data in CSV format with the following columns: 'stage', 'c_i', 't_i', 'C_i', 'M_i', 'sigma_i'. Here's a brief description of each column:
  - 'stage': The stage number in the supply chain.
  - 'c_i': Cost associated with each unit of inventory.
  - 't_i': Lead time for replenishment.
  - 'C_i': Cumulative cost at each stage.
  - 'M_i': Maximum replenishment level.
  - 'sigma_i': Standard deviation representing demand variability.

Safety Stock Calculation:
The algorithm calculates safety stock levels based on the input parameters and the relationships between different stages in the supply chain. Safety stock is determined to ensure that sufficient inventory is available to meet demand during lead times while considering variability in demand and lead times.

Output Explanation:
The printed results display the optimal inventory levels, safety stock costs, and related metrics for each stage in the supply chain. Additionally, the total safety stock cost is provided.

Note:
- Ensure that the input CSV file is correctly formatted with valid data.
- Review the output carefully to understand the implications of inventory management decisions.

Example Usage:
1. Prepare input data in CSV format.
2. Execute the Python script and provide necessary inputs.
3. Review the printed results and the generated CSV file for detailed insights into inventory management.
