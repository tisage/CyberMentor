# CyberMentor: Enhancing Cybersecurity Education with Generative AI

<p align="center">
  <img src="/pic/GUI.PNG" width="500"/>
</p>

## Overview
CyberMentor is an innovative framework designed to address key challenges faced by college students in cybersecurity education. Leveraging Generative Artificial Intelligence (AI) and Large Language Models (LLMs), CyberMentor aims to enhance accessibility, engagement, and academic success by providing personalized, contextually relevant learning experiences.

This project integrates Retrieval-Augmented Generation (RAG) techniques within an agentic workflow to intelligently allocate student needs to various AI tools. By doing so, CyberMentor ensures that students receive tailored guidance, skill-based training, and mentoring support—precisely when and how they need it. The system also includes multilingual support to reduce barriers for non-native English speakers and offers flexible, adaptive resources to meet the diverse needs of non-traditional students.

## Key Features
- Personalized Learning: Tailored curriculum engagement and AI-generated content recommendations.
- AI-Driven Mentorship: Automated mentoring system designed to combat student isolation and provide timely support.
- Retrieval-Augmented Generation (RAG): Accurate and relevant information retrieval for effective knowledge acquisition.
- Agentic Workflow using LangChain & LangGraph: Provides personalized learning tools, real-time feedback, automated grading, and interactive simulations, tailored to meet the specific needs of students.
- Multilingual Support: Helps students overcome language barriers, ensuring equitable learning opportunities.


## Repository Contents
- `db/`: Includes embeddings, vector stores, and database files crucial for information retrieval.
- `pic/`: Flowchart images illustrating the workflow and processes.
- `prompt/`: Houses the prompts used for the LLMs to guide their outputs.
- `eval/`: Evaluation data and results
- `app.py`: A Streamlit-based web application (localhost)
    - Supports chat history (Memory)
    - Chatbot with integrated Agent, Knowledge Base, and Tools
    - Manages multiple sessions
    - Uses caching to speed up response times
    - Support Source of RAG

## Installation
### Download the Repository
Download and unzip the repository files into your chosen folder.

### Configure the API Key:
In the root directory, create a `.env` file.
Add your OpenAI API key in the following format:
```bash
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Install the Required Packages
Install the necessary packages using the following command:
```bash
pip install python-dotenv langchain langchain-openai langchain-experimental langchainhub langgraph openai faiss-cpu pypdf streamlit
```

### Run the Application
To start the application, run the following command:
```bash
streamlit run app.py
```
To stop the application when the browser page (`http://localhost:8080`) is open, press `Ctrl + C` in the terminal.

Note:
The default port number is set to `8080`. If this port is already in use, you can modify the port number in the configuration file located at `.streamlit/config.toml` by changing the value on the line that reads `port = 8080`.


## License
This project is licensed under the MIT License. See the LICENSE file for details.

```
v 1.0
```

Since 6/2024
