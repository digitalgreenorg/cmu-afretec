# Overview

This project designs a Dataset Description and Tagging Tool which utilizes the existing langchain framework and OpenAI models to automatically generate descriptive summaries for datasets and suggest relevant tags for the data. This tool aims to streamline the process of creating meaningful dataset descriptions and improving dataset organization and discoverability. This will essentially help collaborators and researchers better share resource dataset and drive partnership.

## Features
* Automatic Description Generation: The tool uses advanced natural language processing (NLP) models to analyze the dataset and generate concise and informative descriptions automatically.
* Smart Tag Suggestions: By leveraging pre-trained AI models and machine learning algorithms, the tool suggests relevant tags that can be associated with the dataset, enhancing its searchability and categorization.
* Multiple Dataset Formats: The tool supports various dataset formats, such as CSV, JSON, and Excel, making it versatile and compatible with different data sources.

## Input requirements
This tool requires the following data sources:
- Dataset(CSV or Excel): This will be any dataset in the specified format for which a description will be generated and tags suggested

## How to Use
1. Clone repository: The first step is to clone the repository to your local device using the HTTPS git link since the repository is public 
2. Install Dependencies: Before running the tool, make sure you have Python 3.10 installed along with the required libraries. You can install the dependencies using the following command: `pip install -r requirements.txt`
3. Upload Dataset: To be able to get a description of your dataset, upload this dataset to the dataset folder in the project repository cloned to your local device.
4. Add your key: To be able to use OpenAI you will need an open API key; you can add this to the `.env.production` file and this key should be between quotes to be identified as a string
5. Launch the Tool: Change into the scripts directory containing the generate_description.py within the project folder and run the following command to execute the app: `python generate_description.py {Dataset_name}`. You need to replace Dataset_name with the name of the dataset uploaded to the dataset folder
6. Results: The generated descriptions and suggested tags will be printed to the console of the command line interface
 
## Acknowledgement
* This project makes use of the amazing open-source frameworks and pre-trained AI models developed by the community. We extend our gratitude to all the contributors.
* Special thanks to the developers and maintainers of the underlying libraries and tools that made this project possible.

## Disclaimer
The Dataset Description and Tagging Tool provides automated suggestions and generated content, but it may not always be 100% accurate. Users are advised to review and validate the generated descriptions and tags before using them in describing and tagging their dataset. The developers are not responsible for any inaccuracies or damages caused by the use of this tool.