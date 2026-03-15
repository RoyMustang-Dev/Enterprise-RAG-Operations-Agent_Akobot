# Verification Report

## Langfuse Import
- status: ok

## RAG Query
- file: test-files\new-flow-test\Automaatte.txt
- query: Summarize the attached document in 5 bullet points.
- status: 200
- time_s: 24.74
- response: {"session_id":"b58bc601-5820-4e7d-930e-ee2bbe171f76","answer":"```json\n{\n  \"answer\": \"Here is a summary of the attached document in 5 bullet points, focusing on the main points and key information:\\n\\n*   **Automaatte** is an AI Automation Agency providing services and solutions to Indian individuals and corporates, focusing on automating various use cases [Doc Automaatte.txt].\\n*   The agency offers core services in **AI Research**, covering areas such as vacation, education, investment, insurance, and video shoot research [Doc Automaatte.txt].\\n*   Following research, they provide **AI Planning** services for the same domains, helping clients plan according to their research findings [Doc Automaatte.txt].\\n*   Clients also benefit from **free services** including text, video, and document summarization, document QnA, and language translations [Doc Automaatte.txt].\\n*   **Special services**, some of which are in the research phase, include various recommendations (skincare, movie, games), custom chatbots, social media automations (YouTube, Instagram, LinkedIn, Blog, Facebook), and automation services for IT professionals (Excel formulas, database queries, Power BI funct

## RAG Stress Query
- file: test-files\new-flow-test\Automaatte.txt
- query: From the knowledge base, provide a structured brief: 1) Core product purpose, 2) Top 5 features, 3) Target users, 4) Any limitations/unknowns, 5) Provide 3 source-backed bullet points. If information is missing, explicitly say so.
- status: 200
- time_s: 29.88
- response: {"session_id":"a9f356eb-568a-4b1f-a2b9-a35c2acf5158","answer":"```json\n{\n  \"answer\": \"Automaatte is an AI Automation Agency that provides various AI automation services and solutions.\\n\\n**1) Core Product Purpose:**\\nAutomaatte's core purpose is to provide AI automation services and solutions for a number of use cases, catering to both individuals and corporates. [Doc Automaatte.txt]\\n\\n**2) Top 5 Features:**\\n1.  **AI Research Services:** Including AI Vacation Research, AI Education Research, AI Investment Research, AI Insurance Research, and AI Video Shoot Research. [Doc Automaatte.txt]\\n2.  **AI Planning Services:** Including AI Vacation Planning, AI Education Planning, AI Investment Planning, AI Insurance Planning, and AI Video Shoot Planning. [Doc Automaatte.txt]\\n3.  **Summarisation:** Offering Text Summarisation, Video Summarisation, and Document Summarisation. [Doc Automaatte.txt]\\n4.  **Document QnA:** Providing question and answer capabilities for documents. [Doc Automaatte.txt]\\n5.  **Language Translations:** Offering language translation services. [Doc Automaatte.txt]\\n\\n**3) Target Users:**\\nAutomaatte targets Indian audiences, including single indivi

## BA Query
- dataset: data\uploads\marketing_ecommerce_benchmark.csv
- query: Identify the future trends of all products based on current data for every region individually.
- status: 200
- time_s: 22.75
- response: {"status":"success","agent":"BUSINESS_ANALYST","data":{"summary_paragraph":"Analyzed 1199 rows, date range 2023-01-01 to 2026-04-13, 4 regions, 5 products. Forecasting used xgboost for near-term trend projection. Overall regional trend: down in 4/4, up in 0/4, flat in 0/4. Top movers by region: Asia - Smartphone (down, -28.07%), Tablet (down, -27.23%); Europe - Laptop (down, -20.16%), Monitor (down, -19.93%); North America - Tablet (down, -14.25%), Smartphone (down, -12.48%); South America - Laptop (down, -16.56%), Monitor (down, -14.76%).","per_region_summaries":[{"region":"Asia","top_products":[{"product":"Smartphone","forecast_delta_percent":-28.066595295569563,"direction":"down","reason":"Forecast moved from 349.72 to 251.57."},{"product":"Tablet","forecast_delta_percent":-27.231755390450473,"direction":"down","reason":"Forecast moved from 363.52 to 264.53."},{"product":"Laptop","forecast_delta_percent":-23.172817876553115,"direction":"down","reason":"Forecast moved from 301.88 to 231.93."}],"overall_region_trend":"down","notes":"3 products evaluated."},{"region":"Europe","top_products":[{"product":"Laptop","forecast_delta_percent":-20.15554719929366,"direction":"down","reason"

## BA Stress Query
- dataset: data\uploads\marketing_ecommerce_benchmark.csv
- query: Using the dataset, build a full region-by-region trend analysis. Include top 3 products per region by forecast delta, overall regional trend, backtest MAE, confidence interval, and highlight any anomalies or risks.
- status: 200
- time_s: 22.59
- response: {"status":"success","agent":"BUSINESS_ANALYST","data":{"summary_paragraph":"Analyzed 1199 rows, date range 2023-01-01 to 2026-04-13, 4 regions, 5 products. Forecasting used xgboost for near-term trend projection. Overall regional trend: down in 4/4, up in 0/4, flat in 0/4. Top movers by region: Asia - Smartphone (down, -28.07%), Tablet (down, -27.23%); Europe - Laptop (down, -20.16%), Monitor (down, -19.93%); North America - Tablet (down, -14.25%), Smartphone (down, -12.48%); South America - Laptop (down, -16.56%), Monitor (down, -14.76%).","per_region_summaries":[{"region":"Asia","top_products":[{"product":"Smartphone","forecast_delta_percent":-28.066595295569563,"direction":"down","reason":"Forecast moved from 349.72 to 251.57."},{"product":"Tablet","forecast_delta_percent":-27.231755390450473,"direction":"down","reason":"Forecast moved from 363.52 to 264.53."},{"product":"Laptop","forecast_delta_percent":-23.172817876553115,"direction":"down","reason":"Forecast moved from 301.88 to 231.93."}],"overall_region_trend":"down","notes":"3 products evaluated."},{"region":"Europe","top_products":[{"product":"Laptop","forecast_delta_percent":-20.15554719929366,"direction":"down","reason"

## SLA Metrics Snapshot
- status: 200
- time_s: 2.07
- response: {"tenant_id":"aditya-ds","samples":21,"latency_ms":{"avg":19260.229,"p95":39798.349,"p99":39969.255,"max":40900.347},"error_rate":0.1905,"by_agent":{"rag_agent":10,"smalltalk":7,"unknown":4}}
